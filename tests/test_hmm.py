# tests/test_hmm.py

import pytest
import numpy as np

# 导入 Cython 版和 Numba 版的 HMM 模块
from marble.tasks.GTZANBeatTracking.madmom import hmm as cy_hmm_mod
from marble.tasks.GTZANBeatTracking.madmom import hmm_numba as nb_hmm_mod


# --------------------------------------------------------------------------------
# 1. TransitionModel.make_sparse / make_dense (Numba) / from_dense
# --------------------------------------------------------------------------------

def test_transition_model_make_sparse_and_make_dense_roundtrip_numba():
    """
    仅针对 Numba 版：
    先用 make_sparse 构造 CSR，然后用 make_dense 恢复，并验证内容一致。
    """
    # 构造一个简单的 dense list：
    #   prev_state=0 -> next_state=0 概率 0.6, -> next_state=1 概率 0.4
    #   prev_state=1 -> next_state=1 概率 1.0
    # 确保 states < num_states，即不越界
    states = np.array([0, 1, 1], dtype=np.uint32)
    prev_states = np.array([0, 0, 1], dtype=np.uint32)
    probs = np.array([0.6, 0.4, 1.0], dtype=np.float64)

    # Numba 版 make_sparse
    s_nb, p_nb, d_nb = nb_hmm_mod.TransitionModel.make_sparse(states, prev_states, probs)
    # pointers 长度应等于 num_states+1，其中 num_states = max(prev_states)+1 = 2
    assert len(p_nb) == (int(prev_states.max()) + 1) + 1  # 2 + 1 = 3
    # 类型检查
    assert isinstance(s_nb, np.ndarray) and isinstance(p_nb, np.ndarray) and isinstance(d_nb, np.ndarray)
    assert s_nb.dtype == np.uint32
    assert p_nb.dtype == np.uint32
    assert d_nb.dtype == np.float64

    # Numba 版 make_dense：返回 (recovered_states_nb, recovered_prev_states_nb, recovered_probs_nb)
    recovered_states_nb, recovered_prev_states_nb, recovered_probs_nb = \
        nb_hmm_mod.TransitionModel.make_dense(s_nb, p_nb, d_nb)

    # 因为 Numba make_dense 返回值是 (orig prev_states, orig states, orig probs)
    # 所以用 (prev_states, states, probs) 与返回值对应
    orig_set_nb = set(zip(prev_states.tolist(), states.tolist(), probs.tolist()))
    rec_set_nb = set(zip(recovered_states_nb.tolist(),
                         recovered_prev_states_nb.tolist(),
                         recovered_probs_nb.tolist()))
    assert orig_set_nb == rec_set_nb

    # Numba 版 from_dense
    tm_nb = nb_hmm_mod.TransitionModel.from_dense(states, prev_states, probs)
    assert np.array_equal(tm_nb.states, s_nb)
    assert np.array_equal(tm_nb.pointers, p_nb)
    assert np.allclose(tm_nb.probabilities, d_nb)


def test_transition_model_make_sparse_and_from_dense_cython():
    """
    仅针对 Cython 版：
    测试 make_sparse 与 make_dense、一致性验证。
    """
    states = np.array([0, 1, 1], dtype=np.uint32)
    prev_states = np.array([0, 0, 1], dtype=np.uint32)
    probs = np.array([0.6, 0.4, 1.0], dtype=np.float64)

    # Cython 版 make_sparse
    s_cy, p_cy, d_cy = cy_hmm_mod.TransitionModel.make_sparse(states, prev_states, probs)

    # 用 make_dense 恢复原始 (states, prev_states, probabilities)
    rec_states_cy, rec_prev_states_cy, rec_probs_cy = \
        cy_hmm_mod.TransitionModel.make_dense(s_cy, p_cy, d_cy)

    orig_set_cy = set(zip(states.tolist(), prev_states.tolist(), probs.tolist()))
    rec_set_cy = set(zip(rec_states_cy.tolist(),
                         rec_prev_states_cy.tolist(),
                         rec_probs_cy.tolist()))
    assert orig_set_cy == rec_set_cy

    # Cython 版 from_dense
    tm_cy = cy_hmm_mod.TransitionModel.from_dense(states, prev_states, probs)
    assert np.array_equal(tm_cy.states, s_cy)
    assert np.array_equal(tm_cy.pointers, p_cy)
    assert np.allclose(tm_cy.probabilities, d_cy)


def test_transition_model_make_sparse_invalid_probability():
    """
    测试：当某个 prev_state 的转移概率之和 != 1 时，make_sparse/from_dense 应抛 ValueError。
    """
    states = np.array([0, 1], dtype=np.uint32)
    prev_states = np.array([0, 0], dtype=np.uint32)
    probs = np.array([0.3, 0.3], dtype=np.float64)

    with pytest.raises(ValueError):
        nb_hmm_mod.TransitionModel.make_sparse(states, prev_states, probs)
    with pytest.raises(ValueError):
        cy_hmm_mod.TransitionModel.make_sparse(states, prev_states, probs)
    with pytest.raises(ValueError):
        nb_hmm_mod.TransitionModel.from_dense(states, prev_states, probs)
    with pytest.raises(ValueError):
        cy_hmm_mod.TransitionModel.from_dense(states, prev_states, probs)


def test_transition_model_make_dense_empty_raises():
    """
    测试：当 states 和 probabilities 都为空时，make_dense 会抛出 ValueError（Numba、Cython 都一样）。
    """
    states = np.array([], dtype=np.uint32)
    pointers = np.array([0, 0], dtype=np.uint32)  # 一个 state 但无 transitions
    probabilities = np.array([], dtype=np.float64)

    with pytest.raises(ValueError):
        cy_hmm_mod.TransitionModel.make_dense(states, pointers, probabilities)
    with pytest.raises(ValueError):
        nb_hmm_mod.TransitionModel.make_dense(states, pointers, probabilities)


# --------------------------------------------------------------------------------
# 2. DiscreteObservationModel.densities / log_densities
# --------------------------------------------------------------------------------

def test_discrete_observation_model_densities_and_logdensities():
    """
    测试：正确的 observation_probabilities 下，densities 和 log_densities 行为一致。
    """
    obs_probs = np.array([
        [0.1, 0.6, 0.3],  # state 0: P(obs=0,1,2)
        [0.7, 0.2, 0.1],  # state 1: P(obs=0,1,2)
    ], dtype=np.float64)

    om_c = cy_hmm_mod.DiscreteObservationModel(obs_probs)
    om_n = nb_hmm_mod.DiscreteObservationModel(obs_probs)

    observations = np.array([0, 2, 1, 2], dtype=np.uint32)

    dens_c = om_c.densities(observations)
    dens_n = om_n.densities(observations)
    expected = np.vstack([
        [obs_probs[0, 0], obs_probs[1, 0]],
        [obs_probs[0, 2], obs_probs[1, 2]],
        [obs_probs[0, 1], obs_probs[1, 1]],
        [obs_probs[0, 2], obs_probs[1, 2]],
    ])
    assert dens_c.shape == (4, 2)
    assert dens_n.shape == (4, 2)
    assert np.allclose(dens_c, expected)
    assert np.allclose(dens_n, expected)

    log_c = om_c.log_densities(observations)
    log_n = om_n.log_densities(observations)
    assert np.allclose(log_c, np.log(expected))
    assert np.allclose(log_n, np.log(expected))


def test_discrete_observation_model_invalid_probability():
    """
    测试：当传入的 observation_probabilities 每行加和不为 1，应抛 ValueError。
    """
    bad_probs = np.array([
        [0.2, 0.2, 0.2],  # 行和 0.6 != 1
        [0.5, 0.5, 0.1],  # 行和 1.1 != 1
    ], dtype=np.float64)
    with pytest.raises(ValueError):
        cy_hmm_mod.DiscreteObservationModel(bad_probs)
    with pytest.raises(ValueError):
        nb_hmm_mod.DiscreteObservationModel(bad_probs)


# --------------------------------------------------------------------------------
# 3. HiddenMarkovModel: viterbi / forward / forward_generator / reset / initial_distribution
# --------------------------------------------------------------------------------

@pytest.fixture
def simple_2state_hmm():
    """
    Fixture: 构造一个简单的 2-state HMM。
    - transition: 0->0=0.7, 0->1=0.3; 1->0=0.4, 1->1=0.6
    - observation: state 0: [0.2, 0.8]; state 1: [0.5, 0.5]
    - initial_distribution: 默认为均匀
    """
    states = np.array([0, 1, 0, 1], dtype=np.uint32)
    prev_states = np.array([0, 0, 1, 1], dtype=np.uint32)
    probs = np.array([0.7, 0.3, 0.4, 0.6], dtype=np.float64)

    obs_probs = np.array([
        [0.2, 0.8],  # state 0: P(obs=0, obs=1)
        [0.5, 0.5],  # state 1: P(obs=0, obs=1)
    ], dtype=np.float64)

    tm_c = cy_hmm_mod.TransitionModel.from_dense(states, prev_states, probs)
    om_c = cy_hmm_mod.DiscreteObservationModel(obs_probs)
    hmm_c = cy_hmm_mod.HiddenMarkovModel(tm_c, om_c)

    tm_n = nb_hmm_mod.TransitionModel.from_dense(states, prev_states, probs)
    om_n = nb_hmm_mod.DiscreteObservationModel(obs_probs)
    hmm_n = nb_hmm_mod.HiddenMarkovModel(tm_n, om_n)

    return hmm_c, hmm_n


def test_viterbi_simple_hmm(simple_2state_hmm):
    """
    测试 viterbi 输出在 Cython & Numba 下保持一致。
    """
    hmm_c, hmm_n = simple_2state_hmm
    obs_seq = np.array([0, 1, 1, 0], dtype=np.uint32)

    path_c, logp_c = hmm_c.viterbi(obs_seq)
    path_n, logp_n = hmm_n.viterbi(obs_seq)

    assert np.array_equal(path_c, path_n)
    assert np.isfinite(logp_c) and np.isfinite(logp_n)
    assert np.isclose(logp_c, logp_n, atol=1e-12)
    assert path_c.shape == obs_seq.shape


def test_forward_simple_hmm(simple_2state_hmm):
    """
    测试 forward 输出在 Cython & Numba 下保持一致，以及 reset=False 分支行为。
    """
    hmm_c, hmm_n = simple_2state_hmm
    obs_seq = np.array([0, 1, 1, 0], dtype=np.uint32)

    fwd_c = hmm_c.forward(obs_seq, reset=True)
    fwd_n = hmm_n.forward(obs_seq, reset=True)
    assert fwd_c.shape == fwd_n.shape
    assert np.allclose(fwd_c, fwd_n, atol=1e-12)

    # reset=False 下可直接运行，形状一致
    fwd_c2 = hmm_c.forward(obs_seq, reset=True)
    fwd_c3 = hmm_c.forward(obs_seq, reset=False)
    assert fwd_c3.shape == fwd_c2.shape


def test_forward_generator_simple_hmm(simple_2state_hmm):
    """
    测试 forward_generator：不同 block_size 下，Cython & Numba 输出保持一致。
    """
    hmm_c, hmm_n = simple_2state_hmm
    obs_seq = np.array([0, 1, 1, 0, 0, 1], dtype=np.uint32)

    # block_size=None
    gen_c1 = hmm_c.forward_generator(obs_seq, block_size=None)
    gen_n1 = hmm_n.forward_generator(obs_seq, block_size=None)
    for _ in range(len(obs_seq)):
        vec_c = next(gen_c1)
        vec_n = next(gen_n1)
        assert np.allclose(vec_c, vec_n, atol=1e-12)

    # block_size=2
    gen_c2 = hmm_c.forward_generator(obs_seq, block_size=2)
    gen_n2 = hmm_n.forward_generator(obs_seq, block_size=2)
    for _ in range(len(obs_seq)):
        vec_c = next(gen_c2)
        vec_n = next(gen_n2)
        assert np.allclose(vec_c, vec_n, atol=1e-12)


def test_reset_and_initial_distribution_errors(simple_2state_hmm):
    """
    测试 HiddenMarkovModel 构造时 initial_distribution 检验：
    - Cython 版只校验 sum；Numba 版同时校验长度和 sum。
    """
    hmm_c, _ = simple_2state_hmm
    tm = hmm_c.transition_model
    om = hmm_c.observation_model

    # 1) initial_distribution 长度错误：
    bad_length = np.array([1.0, 0.0, 0.0])
    # Cython 版：只校验 sum，因此不抛异常
    cy_hmm_mod.HiddenMarkovModel(tm, om, initial_distribution=bad_length)
    # Numba 版：校验长度，应抛 ValueError
    with pytest.raises(ValueError):
        nb_hmm_mod.HiddenMarkovModel(tm, om, initial_distribution=bad_length)

    # 2) initial_distribution sum !=1：
    bad_sum = np.array([0.6, 0.6])
    with pytest.raises(ValueError):
        cy_hmm_mod.HiddenMarkovModel(tm, om, initial_distribution=bad_sum)
    with pytest.raises(ValueError):
        nb_hmm_mod.HiddenMarkovModel(tm, om, initial_distribution=bad_sum)

    # 3) reset() 后，forward(reset=False) 以均匀初始分布计算第一帧
    obs_seq = np.array([0, 1, 0, 1], dtype=np.uint32)
    _ = hmm_c.forward(obs_seq, reset=True)
    hmm_c.reset()
    # 对于 obs=[0]：
    # initial=[0.5,0.5]
    # state0: (0.5*0.7 + 0.5*0.4)*0.2 = 0.11
    # state1: (0.5*0.3 + 0.5*0.6)*0.5 = 0.225
    # 归一化 => [0.32835821, 0.67164179]
    expected_first = np.array([0.3283582089552239, 0.6716417910447761])
    new_fwd = hmm_c.forward(np.array([0], dtype=np.uint32), reset=False)
    assert new_fwd.shape == (1, 2)
    assert np.allclose(new_fwd[0], expected_first, atol=1e-8)


def test_viterbi_no_valid_path_returns_empty_and_negative_inf():
    """
    测试：如果给定的观测序列在 HMM 中无任何可能路径（概率全为 0），viterbi 返回空 path 且 log_prob = -inf。
    """
    # 构造一个全零转移的 TransitionModel
    states = np.array([], dtype=np.uint32)
    pointers = np.array([0, 0, 0], dtype=np.uint32)  # 2 states，但无 transitions
    probabilities = np.array([], dtype=np.float64)
    tm = cy_hmm_mod.TransitionModel(states, pointers, probabilities)

    obs_probs = np.array([[1.0], [1.0]], dtype=np.float64)
    om = cy_hmm_mod.DiscreteObservationModel(obs_probs)

    # Cython 版
    hmm_c = cy_hmm_mod.HiddenMarkovModel(tm, om)
    obs_seq = np.array([0, 0, 0], dtype=np.uint32)
    path_c, logp_c = hmm_c.viterbi(obs_seq)
    assert path_c.shape == (0,)
    assert logp_c == float("-inf")

    # Numba 版
    tm_n = nb_hmm_mod.TransitionModel(states, pointers, probabilities)
    om_n = nb_hmm_mod.DiscreteObservationModel(obs_probs)
    hmm_n = nb_hmm_mod.HiddenMarkovModel(tm_n, om_n)
    path_n, logp_n = hmm_n.viterbi(obs_seq)
    assert path_n.shape == (0,)
    assert logp_n == float("-inf")


def test_forward_all_zero_observation_probabilities(simple_2state_hmm):
    """
    测试：如果观测序列包含超出观测概率范围的索引，应抛 IndexError。
    """
    hmm_c, hmm_n = simple_2state_hmm

    # obs_probs 只有两列（obs ∈ {0,1}），这里用 obs=2 来触发越界
    obs_seq = np.array([0, 2, 0], dtype=np.uint32)

    with pytest.raises(IndexError):
        _ = hmm_c.forward(obs_seq)
    with pytest.raises(IndexError):
        _ = hmm_n.forward(obs_seq)


def test_hidden_markov_model_reset_behavior(simple_2state_hmm):
    """
    测试 reset() 后，forward(reset=False) 以均匀初始分布计算第一帧与预期一致。
    """
    hmm_c, _ = simple_2state_hmm

    # 先跑一段序列，让 _prev_forward 更新
    obs_seq = np.array([0, 1, 0, 1], dtype=np.uint32)
    _ = hmm_c.forward(obs_seq, reset=True)

    # 调用 reset() 恢复到初始均匀分布
    hmm_c.reset()

    # 对于 obs=[0]：
    # initial=[0.5,0.5]
    # state0: (0.5*0.7 + 0.5*0.4)*0.2 = 0.11
    # state1: (0.5*0.3 + 0.5*0.6)*0.5 = 0.225
    # 归一化 => [0.32835821, 0.67164179]
    expected = np.array([0.3283582089552239, 0.6716417910447761])
    new_fwd = hmm_c.forward(np.array([0], dtype=np.uint32), reset=False)
    assert new_fwd.shape == (1, 2)
    assert np.allclose(new_fwd[0], expected, atol=1e-8)
