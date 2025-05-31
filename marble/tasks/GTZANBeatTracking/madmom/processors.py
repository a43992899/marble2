# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains all processor related functionality.

Notes
-----
All features should be implemented as classes which inherit from Processor
(or provide a XYZProcessor(Processor) variant). This way, multiple Processor
objects can be chained/combined to achieve the wanted functionality.

"""
string_types = str
integer_types = (int, np.integer)

from __future__ import absolute_import, division, print_function

import argparse
import itertools as it
import multiprocessing as mp
import os
import sys
from collections.abc import MutableSequence
import contextlib
import io as _io

import numpy as np


# overwrite the built-in open() to transparently apply some magic file handling
@contextlib.contextmanager
def open_file(filename, mode='r'):
    """
    Context manager which yields an open file or handle with the given mode
    and closes it if needed afterwards.

    Parameters
    ----------
    filename : str or file handle
        File (handle) to open.
    mode: {'r', 'w'}
        Specifies the mode in which the file is opened.

    Yields
    ------
        Open file (handle).

    """
    # check if we need to open the file
    if isinstance(filename, string_types):
        f = fid = _io.open(filename, mode)
    else:
        f = filename
        fid = None
    # yield an open file handle
    yield f
    # close the file if needed
    if fid:
        fid.close()


class Processor(object):
    """
    Abstract base class for processing data.

    """

    @classmethod
    def load(cls, infile):
        """
        Instantiate a new Processor from a file.

        This method un-pickles a saved Processor object. Subclasses should
        overwrite this method with a better performing solution if speed is an
        issue.

        Parameters
        ----------
        infile : str or file handle
            Pickled processor.

        Returns
        -------
        :class:`Processor` instance
            Processor.

        """
        import pickle
        # instantiate a new Processor and return it
        with open_file(infile, 'rb') as f:
            # Python 2 and 3 behave differently
            try:
                # Python 3
                obj = pickle.load(f, encoding='latin1')
            except TypeError:
                # Python 2 doesn't have/need the encoding
                obj = pickle.load(f)
        return obj

    def dump(self, outfile):
        """
        Save the Processor to a file.

        This method pickles a Processor object and saves it. Subclasses should
        overwrite this method with a better performing solution if speed is an
        issue.

        Parameters
        ----------
        outfile : str or file handle
            Output file for pickling the processor.

        """
        import pickle
        # dump the Processor to the given file
        # Note: for Python 2 / 3 compatibility reason use protocol 2
        with open_file(outfile, 'wb') as f:
            pickle.dump(self, f, protocol=2)

    def process(self, data, **kwargs):
        """
        Process the data.

        This method must be implemented by the derived class and should
        process the given data and return the processed output.

        Parameters
        ----------
        data : depends on the implementation of subclass
            Data to be processed.
        kwargs : dict, optional
            Keyword arguments for processing.

        Returns
        -------
        depends on the implementation of subclass
            Processed data.

        """
        raise NotImplementedError('Must be implemented by subclass.')

    def __call__(self, *args, **kwargs):
        # this magic method makes a Processor callable
        return self.process(*args, **kwargs)


class OnlineProcessor(Processor):
    """
    Abstract base class for processing data in online mode.

    Derived classes must implement the following methods:

    - process_online(): process the data in online mode,
    - process_offline(): process the data in offline mode.

    """

    def __init__(self, online=False):
        self.online = online

    def process(self, data, **kwargs):
        """
        Process the data either in online or offline mode.

        Parameters
        ----------
        data : depends on the implementation of subclass
            Data to be processed.
        kwargs : dict, optional
            Keyword arguments for processing.

        Returns
        -------
        depends on the implementation of subclass
            Processed data.

        Notes
        -----
        This method is used to pass the data to either `process_online` or
        `process_offline`, depending on the `online` setting of the processor.

        """
        if self.online:
            return self.process_online(data, **kwargs)
        return self.process_offline(data, **kwargs)

    def process_online(self, data, reset=True, **kwargs):
        """
        Process the data in online mode.

        This method must be implemented by the derived class and should process
        the given data frame by frame and return the processed output.

        Parameters
        ----------
        data : depends on the implementation of subclass
            Data to be processed.
        reset : bool, optional
            Reset the processor to its initial state before processing.
        kwargs : dict, optional
            Keyword arguments for processing.

        Returns
        -------
        depends on the implementation of subclass
            Processed data.

        """
        raise NotImplementedError('Must be implemented by subclass.')

    def process_offline(self, data, **kwargs):
        """
        Process the data in offline mode.

        This method must be implemented by the derived class and should process
        the given data and return the processed output.

        Parameters
        ----------
        data : depends on the implementation of subclass
            Data to be processed.
        kwargs : dict, optional
            Keyword arguments for processing.

        Returns
        -------
        depends on the implementation of subclass
            Processed data.

        """
        raise NotImplementedError('Must be implemented by subclass.')

    def reset(self):
        """
        Reset the OnlineProcessor.

        This method must be implemented by the derived class and should reset
        the processor to its initial state.

        """
        raise NotImplementedError('Must be implemented by subclass.')

