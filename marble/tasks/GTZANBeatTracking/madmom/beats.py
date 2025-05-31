# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains beat tracking related functionality.

"""

from __future__ import absolute_import, division, print_function

import sys

import numpy as np

from .processors import OnlineProcessor


def threshold_activations(activations, threshold):
    """
    Threshold activations to include only the main segment exceeding the given
    threshold (i.e. first to last time/index exceeding the threshold).

    Parameters
    ----------
    activations : numpy array
        Activations to be thresholded.
    threshold : float
        Threshold value.

    Returns
    -------
    activations : numpy array
        Thresholded activations
    start : int
        Index of the first activation exceeding the threshold.

    Notes
    -----

    This function can be used to extract the main segment of beat activations
    to track only the beats where the activations exceed the threshold.

    """
    first = last = 0
    # use only the activations > threshold
    idx = np.nonzero(activations >= threshold)[0]
    if idx.any():
        first = max(first, np.min(idx))
        last = min(len(activations), np.max(idx) + 1)
    # return thresholded activations segment and first index
    return activations[first:last], first


class DBNBeatTrackingProcessor(OnlineProcessor):
    """
    Beat tracking with RNNs and a dynamic Bayesian network (DBN) approximated
    by a Hidden Markov Model (HMM).

    Parameters
    ----------
    min_bpm : float, optional
        Minimum tempo used for beat tracking [bpm].
    max_bpm : float, optional
        Maximum tempo used for beat tracking [bpm].
    num_tempi : int, optional
        Number of tempi to model; if set, limit the number of tempi and use a
        log spacing, otherwise a linear spacing.
    transition_lambda : float, optional
        Lambda for the exponential tempo change distribution (higher values
        prefer a constant tempo from one beat to the next one).
    observation_lambda : int, optional
        Split one beat period into `observation_lambda` parts, the first
        representing beat states and the remaining non-beat states.
    threshold : float, optional
        Threshold the observations before Viterbi decoding.
    correct : bool, optional
        Correct the beats (i.e. align them to the nearest peak of the beat
        activation function).
    fps : float, optional
        Frames per second.
    online : bool, optional
        Use the forward algorithm (instead of Viterbi) to decode the beats.

    Notes
    -----
    Instead of the originally proposed state space and transition model for
    the DBN [1]_, the more efficient version proposed in [2]_ is used.

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "A Multi-Model Approach to Beat Tracking Considering Heterogeneous
           Music Styles",
           Proceedings of the 15th International Society for Music Information
           Retrieval Conference (ISMIR), 2014.
    .. [2] Florian Krebs, Sebastian Böck and Gerhard Widmer,
           "An Efficient State Space Model for Joint Tempo and Meter Tracking",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    Examples
    --------
    Create a DBNBeatTrackingProcessor. The returned array represents the
    positions of the beats in seconds, thus the expected sampling rate has to
    be given.

    >>> proc = DBNBeatTrackingProcessor(fps=100)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.beats.DBNBeatTrackingProcessor object at 0x...>

    Call this DBNBeatTrackingProcessor with the beat activation function
    returned by RNNBeatProcessor to obtain the beat positions.

    >>> act = RNNBeatProcessor()('tests/data/audio/sample.wav')
    >>> proc(act)
    array([0.1 , 0.45, 0.8 , 1.12, 1.48, 1.8 , 2.15, 2.49])

    """
    MIN_BPM = 55.
    MAX_BPM = 215.
    NUM_TEMPI = None
    TRANSITION_LAMBDA = 100
    OBSERVATION_LAMBDA = 16
    THRESHOLD = 0
    CORRECT = True

    def __init__(self, min_bpm=MIN_BPM, max_bpm=MAX_BPM, num_tempi=NUM_TEMPI,
                 transition_lambda=TRANSITION_LAMBDA,
                 observation_lambda=OBSERVATION_LAMBDA, correct=CORRECT,
                 threshold=THRESHOLD, fps=None, online=False, **kwargs):
        # pylint: disable=unused-argument
        # pylint: disable=no-name-in-module
        from .beats_hmm import (BeatStateSpace, BeatTransitionModel,
                                RNNBeatTrackingObservationModel)
        from .hmm import HiddenMarkovModel
        # convert timing information to construct a beat state space
        min_interval = 60. * fps / max_bpm
        max_interval = 60. * fps / min_bpm
        self.st = BeatStateSpace(min_interval, max_interval, num_tempi)
        # transition model
        self.tm = BeatTransitionModel(self.st, transition_lambda)
        # observation model
        self.om = RNNBeatTrackingObservationModel(self.st, observation_lambda)
        # instantiate a HMM
        self.hmm = HiddenMarkovModel(self.tm, self.om, None)
        # save variables
        self.correct = correct
        self.threshold = threshold
        self.fps = fps
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        # keep state in online mode
        self.online = online
        # TODO: refactor the visualisation stuff
        if self.online:
            self.visualize = kwargs.get('verbose', False)
            self.counter = 0
            self.beat_counter = 0
            self.strength = 0
            self.last_beat = 0
            self.tempo = 0

    def reset(self):
        """Reset the DBNBeatTrackingProcessor."""
        # pylint: disable=attribute-defined-outside-init
        # reset the HMM
        self.hmm.reset()
        # reset other variables
        self.counter = 0
        self.beat_counter = 0
        self.strength = 0
        self.last_beat = 0
        self.tempo = 0

    def process_offline(self, activations, **kwargs):
        """
        Detect the beats in the given activation function with Viterbi
        decoding.

        Parameters
        ----------
        activations : numpy array
            Beat activation function.

        Returns
        -------
        beats : numpy array
            Detected beat positions [seconds].

        """
        # init the beats to return and the offset
        beats = np.empty(0, dtype=int)
        first = 0
        # use only the activations > threshold
        if self.threshold:
            activations, first = threshold_activations(activations,
                                                       self.threshold)
        # return no beats if no activations given / remain after thresholding
        if not activations.any():
            return beats
        # get the best state path by calling the viterbi algorithm
        path, _ = self.hmm.viterbi(activations)
        # also return no beats if no path was found
        if not path.any():
            return beats
        # correct the beat positions if needed
        if self.correct:
            # for each detection determine the "beat range", i.e. states where
            # the pointers of the observation model are 1
            beat_range = self.om.pointers[path]
            # get all change points between True and False
            idx = np.nonzero(np.diff(beat_range))[0] + 1
            # if the first frame is in the beat range, add a change at frame 0
            if beat_range[0]:
                idx = np.r_[0, idx]
            # if the last frame is in the beat range, append the length of the
            # array
            if beat_range[-1]:
                idx = np.r_[idx, beat_range.size]
            # iterate over all regions
            if idx.any():
                for left, right in idx.reshape((-1, 2)):
                    # pick the frame with the highest activations value
                    peak = np.argmax(activations[left:right]) + left
                    beats = np.hstack((beats, peak))
        else:
            # just take the frames with the smallest beat state values
            from scipy.signal import argrelmin
            beats = argrelmin(self.st.state_positions[path], mode='wrap')[0]
            # recheck if they are within the "beat range", i.e. the pointers
            # of the observation model for that state must be 1
            # Note: interpolation and alignment of the beats to be at state 0
            #       does not improve results over this simple method
            beats = beats[self.om.pointers[path[beats]] == 1]
        # convert the detected beats to seconds and return them
        return (beats + first) / float(self.fps)

    def process_online(self, activations, reset=True, **kwargs):
        """
        Detect the beats in the given activation function with the forward
        algorithm.

        Parameters
        ----------
        activations : numpy array
            Beat activation for a single frame.
        reset : bool, optional
            Reset the DBNBeatTrackingProcessor to its initial state before
            processing.

        Returns
        -------
        beats : numpy array
            Detected beat position [seconds].

        """
        # cast as 1-dimensional array
        # Note: in online mode, activations are just float values
        if not isinstance(activations, np.ndarray):
            activations = np.array(activations, ndmin=1)
        # reset to initial state
        if reset:
            self.reset()
        # use forward path to get best state
        fwd = self.hmm.forward(activations, reset=reset)
        # choose the best state for each step
        states = np.argmax(fwd, axis=1)
        # decide which time steps are beats
        beats = self.om.pointers[states] == 1
        # the positions inside the beats
        positions = self.st.state_positions[states]
        # visualisation stuff (only when called frame by frame)
        if self.visualize and len(activations) == 1:
            beat_length = 80
            display = [' '] * beat_length
            display[int(positions * beat_length)] = '*'
            # activation strength indicator
            strength_length = 10
            self.strength = int(max(self.strength, activations * 10))
            display.append('| ')
            display.extend(['*'] * self.strength)
            display.extend([' '] * (strength_length - self.strength))
            # reduce the displayed strength every couple of frames
            if self.counter % 5 == 0:
                self.strength -= 1
            # beat indicator
            if beats:
                self.beat_counter = 3
            if self.beat_counter > 0:
                display.append('| X ')
            else:
                display.append('|   ')
            self.beat_counter -= 1
            # display tempo
            display.append('| %5.1f | ' % self.tempo)
            sys.stderr.write('\r%s' % ''.join(display))
            sys.stderr.flush()
        # forward path often reports multiple beats close together, thus report
        # only beats more than the minimum interval apart
        beats_ = []
        for frame in np.nonzero(beats)[0]:
            cur_beat = (frame + self.counter) / float(self.fps)
            next_beat = self.last_beat + 60. / self.max_bpm
            # FIXME: this skips the first beat, but maybe this has a positive
            #        effect on the overall beat tracking accuracy
            if cur_beat >= next_beat:
                # update tempo
                self.tempo = 60. / (cur_beat - self.last_beat)
                # update last beat
                self.last_beat = cur_beat
                # append to beats
                beats_.append(cur_beat)
        # increase counter
        self.counter += len(activations)
        # return beat(s)
        return np.array(beats_)

    process_forward = process_online

    process_viterbi = process_offline

    @staticmethod
    def add_arguments(parser, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                      num_tempi=NUM_TEMPI, transition_lambda=TRANSITION_LAMBDA,
                      observation_lambda=OBSERVATION_LAMBDA,
                      threshold=THRESHOLD, correct=CORRECT):
        """
        Add DBN related arguments to an existing parser object.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        min_bpm : float, optional
            Minimum tempo used for beat tracking [bpm].
        max_bpm : float, optional
            Maximum tempo used for beat tracking [bpm].
        num_tempi : int, optional
            Number of tempi to model; if set, limit the number of tempi and use
            a log spacing, otherwise a linear spacing.
        transition_lambda : float, optional
            Lambda for the exponential tempo change distribution (higher values
            prefer a constant tempo over a tempo change from one beat to the
            next one).
        observation_lambda : float, optional
            Split one beat period into `observation_lambda` parts, the first
            representing beat states and the remaining non-beat states.
        threshold : float, optional
            Threshold the observations before Viterbi decoding.
        correct : bool, optional
            Correct the beats (i.e. align them to the nearest peak of the beat
            activation function).

        Returns
        -------
        parser_group : argparse argument group
            DBN beat tracking argument parser group

        """
        # pylint: disable=arguments-differ
        # add DBN parser group
        g = parser.add_argument_group('dynamic Bayesian Network arguments')
        # add a transition parameters
        g.add_argument('--min_bpm', action='store', type=float,
                       default=min_bpm,
                       help='minimum tempo [bpm, default=%(default).2f]')
        g.add_argument('--max_bpm', action='store', type=float,
                       default=max_bpm,
                       help='maximum tempo [bpm,  default=%(default).2f]')
        g.add_argument('--num_tempi', action='store', type=int,
                       default=num_tempi,
                       help='limit the number of tempi; if set, align the '
                            'tempi with a log spacing, otherwise linearly')
        g.add_argument('--transition_lambda', action='store', type=float,
                       default=transition_lambda,
                       help='lambda of the tempo transition distribution; '
                            'higher values prefer a constant tempo over a '
                            'tempo change from one beat to the next one '
                            '[default=%(default).1f]')
        # observation model stuff
        g.add_argument('--observation_lambda', action='store', type=float,
                       default=observation_lambda,
                       help='split one beat period into N parts, the first '
                            'representing beat states and the remaining '
                            'non-beat states [default=%(default)i]')
        g.add_argument('-t', dest='threshold', action='store', type=float,
                       default=threshold,
                       help='threshold the observations before Viterbi '
                            'decoding [default=%(default).2f]')
        # option to correct the beat positions
        if correct:
            g.add_argument('--no_correct', dest='correct',
                           action='store_false', default=correct,
                           help='do not correct the beat positions (i.e. do '
                                'not align them to the nearest peak of the '
                                'beat activation function)')
        else:
            g.add_argument('--correct', dest='correct',
                           action='store_true', default=correct,
                           help='correct the beat positions (i.e. align them '
                                'to the nearest peak of the beat activation'
                                'function)')
        # return the argument group so it can be modified if needed
        return g
