# tests/test_hmm_speed.py

import time
import numpy as np

from marble.tasks.GTZANBeatTracking.madmom import hmm as cy_hmm_mod
from marble.tasks.GTZANBeatTracking.madmom import hmm_numba as nb_hmm_mod


def generate_random_hmm(num_states: int, num_obs_types: int, seed: int = 42):
    """
    生成一个随机的 HMM，用于 speed benchmark。
    - num_states: 状态数目
    - num_obs_types: 离散观测类型数目
    """
    rng = np.random.RandomState(seed)

    # 1) 随机生成 transition matrix：每个 prev_state 都要归一化
    states_list = []
    prev_states_list = []
    probs_list = []
    for prev in range(num_states):
        # 随机在每个 prev_state 上挑一些 next_state
        # 这里让每个 prev_state 都能转到所有 state（一般稠密），概率随机分配并归一化
        next_probs = rng.rand(num_states)
        next_probs = next_probs / next_probs.sum()
        for nxt in range(num_states):
            states_list.append(nxt)
            prev_states_list.append(prev)
            probs_list.append(next_probs[nxt])

    states = np.array(states_list, dtype=np.uint32)
    prev_states = np.array(prev_states_list, dtype=np.uint32)
    probs = np.array(probs_list, dtype=np.float64)

    # 2) 随机生成 observation probabilities：num_states x num_obs_types，逐行归一化
    obs_probs = rng.rand(num_states, num_obs_types)
    obs_probs = obs_probs / obs_probs.sum(axis=1, keepdims=True)

    # 3) 随机生成一个观测序列长度 T
    T = 2000  # 保证足够长，以观察运行差别
    obs_seq = rng.randint(0, num_obs_types, size=T, dtype=np.uint32)

    # 构造 Cython 版 HMM
    tm_c = cy_hmm_mod.TransitionModel.from_dense(states, prev_states, probs)
    om_c = cy_hmm_mod.DiscreteObservationModel(obs_probs)
    hmm_c = cy_hmm_mod.HiddenMarkovModel(tm_c, om_c)

    # 构造 Numba 版 HMM
    tm_n = nb_hmm_mod.TransitionModel.from_dense(states, prev_states, probs)
    om_n = nb_hmm_mod.DiscreteObservationModel(obs_probs)
    hmm_n = nb_hmm_mod.HiddenMarkovModel(tm_n, om_n)

    return hmm_c, hmm_n, obs_seq


def benchmark_viterbi(hmm_c, hmm_n, obs_seq):
    print("\n=== Viterbi Benchmark ===")
    # 预热调用
    _ = hmm_c.viterbi(obs_seq)
    _ = hmm_n.viterbi(obs_seq)

    # Cython 版计时
    start = time.perf_counter()
    path_c, logp_c = hmm_c.viterbi(obs_seq)
    dur_c = time.perf_counter() - start

    # Numba 版计时
    start = time.perf_counter()
    path_n, logp_n = hmm_n.viterbi(obs_seq)
    dur_n = time.perf_counter() - start

    # 简单验证两者结果一致
    assert np.array_equal(path_c, path_n)
    assert np.isclose(logp_c, logp_n, atol=1e-8)

    print(f"Cython viterbi time: {dur_c:.4f} s")
    print(f"Numba  viterbi time: {dur_n:.4f} s")
    print(f"Numba/Cython ratio:  {dur_n/dur_c:.2f}")


def benchmark_forward(hmm_c, hmm_n, obs_seq):
    print("\n=== Forward Benchmark ===")
    # 预热
    _ = hmm_c.forward(obs_seq)
    _ = hmm_n.forward(obs_seq)

    # Cython 版计时
    start = time.perf_counter()
    fwd_c = hmm_c.forward(obs_seq)
    dur_c = time.perf_counter() - start

    # Numba 版计时
    start = time.perf_counter()
    fwd_n = hmm_n.forward(obs_seq)
    dur_n = time.perf_counter() - start

    # 验证结果一致
    assert fwd_c.shape == fwd_n.shape
    assert np.allclose(fwd_c, fwd_n, atol=1e-12)

    print(f"Cython forward time: {dur_c:.4f} s")
    print(f"Numba  forward time: {dur_n:.4f} s")
    print(f"Numba/Cython ratio:  {dur_n/dur_c:.2f}")


def benchmark_forward_generator(hmm_c, hmm_n, obs_seq):
    print("\n=== Forward Generator Benchmark (block_size=50) ===")
    # 预热
    _ = list(hmm_c.forward_generator(obs_seq, block_size=50))
    _ = list(hmm_n.forward_generator(obs_seq, block_size=50))

    # Cython 版计时
    start = time.perf_counter()
    gen_c = hmm_c.forward_generator(obs_seq, block_size=50)
    for _ in gen_c:
        pass
    dur_c = time.perf_counter() - start

    # Numba 版计时
    start = time.perf_counter()
    gen_n = hmm_n.forward_generator(obs_seq, block_size=50)
    for _ in gen_n:
        pass
    dur_n = time.perf_counter() - start

    # 简单验证一次采样（前 10 帧）一致
    gen_c2 = hmm_c.forward_generator(obs_seq, block_size=50)
    gen_n2 = hmm_n.forward_generator(obs_seq, block_size=50)
    for i in range(10):
        v_c = next(gen_c2)
        v_n = next(gen_n2)
        assert np.allclose(v_c, v_n, atol=1e-12)

    print(f"Cython forward_generator time: {dur_c:.4f} s")
    print(f"Numba  forward_generator time: {dur_n:.4f} s")
    print(f"Numba/Cython ratio:            {dur_n/dur_c:.2f}")


def main():
    print("Preparing random HMM and observation sequence...")
    hmm_c, hmm_n, obs_seq = generate_random_hmm(num_states=50, num_obs_types=20)

    benchmark_viterbi(hmm_c, hmm_n, obs_seq)
    benchmark_forward(hmm_c, hmm_n, obs_seq)
    benchmark_forward_generator(hmm_c, hmm_n, obs_seq)


if __name__ == "__main__":
    main()
