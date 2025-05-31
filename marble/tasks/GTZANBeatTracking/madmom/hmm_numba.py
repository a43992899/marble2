# encoding: utf-8
"""
marble/tasks/GTZANBeatTracking/madmom/hmm_numba.py
使用 Numba 重写的 Hidden Markov Model (HMM) 模块
"""

import warnings
import numpy as np
from numba import njit

# ------------------------------------------------------------------------------
# TransitionModel: 与原实现一致，保持 Python 实现，主要用于保存稀疏矩阵格式
# ------------------------------------------------------------------------------

class TransitionModel:
    """
    TransitionModel 类：存储 HMM 的转移概率矩阵，采用 CSR（压缩稀疏行）格式。
    attributes:
        states:    uint32 数组，所有可能的“前状态”索引。
        pointers:  uint32 数组，length = num_states + 1，
                   pointers[s]~pointers[s+1] 范围内存储转移到状态 s 的“前状态”索引。
        probabilities: float 数组，与 states 对应，存储相应的转移概率。
    """

    def __init__(self, states: np.ndarray, pointers: np.ndarray, probabilities: np.ndarray):
        self.states = states.astype(np.uint32)
        self.pointers = pointers.astype(np.uint32)
        self.probabilities = probabilities.astype(np.float64)

    @property
    def num_states(self) -> int:
        return len(self.pointers) - 1

    @property
    def num_transitions(self) -> int:
        return self.probabilities.shape[0]

    @property
    def log_probabilities(self) -> np.ndarray:
        return np.log(self.probabilities)

    @staticmethod
    def make_dense(states: np.ndarray, pointers: np.ndarray, probabilities: np.ndarray):
        """
        将 CSR 格式转换为密集表示 (states_list, prev_states_list, probabilities_list)。
        """
        from scipy.sparse import csr_matrix
        # 转成 CSR
        csr = csr_matrix((probabilities.astype(np.float64),
                          states.astype(np.int64),
                          pointers.astype(np.int64)))
        # nonzero 返回 (row_indices, col_indices)，其中 row_indices=destination states, col_indices=previous states
        prev_states, new_states = csr.nonzero()
        # new_states 存放所有 destination state，prev_states 存放对应 previous state
        # 但为了与原接口保持一致，这里返回 states, prev_states, probabilities
        return new_states.astype(np.uint32), prev_states.astype(np.uint32), probabilities.astype(np.float64)

    @staticmethod
    def make_sparse(states: np.ndarray, prev_states: np.ndarray, probabilities: np.ndarray):
        """
        从密集表示(states, prev_states, probabilities) 生成 CSR 格式 (states, pointers, probabilities)。
        """
        from scipy.sparse import csr_matrix
        states = np.asarray(states, dtype=np.uint32)
        prev_states = np.asarray(prev_states, dtype=np.uint32)
        probabilities = np.asarray(probabilities, dtype=np.float64)

        # 检查是否为合法的概率分布（每个 prev_state 的转移概率之和是否为 1）
        binc = np.bincount(prev_states, weights=probabilities)
        if not np.allclose(binc, 1.0):
            raise ValueError("Not a probability distribution for transition from each state.")

        num_states = int(prev_states.max()) + 1
        # 构造一个 num_states x num_states 的稀疏矩阵 (row=destination, col=previous)
        csr = csr_matrix((probabilities, (states.astype(np.int64), prev_states.astype(np.int64))),
                         shape=(num_states, num_states))
        # 提取 CSR 格式
        new_states = csr.indices.astype(np.uint32)       # 所有 row 对应的 col，即在 CSR 中保存的 colIndices
        pointers = csr.indptr.astype(np.uint32)           # indptr 存储每一行在 indices 数组的起止
        data = csr.data.astype(np.float64)
        return new_states, pointers, data

    @classmethod
    def from_dense(cls, states: np.ndarray, prev_states: np.ndarray, probabilities: np.ndarray):
        """
        从密集格式(states, prev_states, probabilities) 创建 TransitionModel 实例。
        """
        s, p, pr = cls.make_sparse(states, prev_states, probabilities)
        return cls(s, p, pr)


# ------------------------------------------------------------------------------
# ObservationModel：保留抽象类与离散观测模型实现
# ------------------------------------------------------------------------------

class ObservationModel:
    """
    ObservationModel 抽象基类，定义接口：
      pointers 属性 用于映射状态到观测维度索引。
      子类需实现 log_densities(observations) 方法（返回形状为 (T, num_states) 的对数概率矩阵）。
    """

    def __init__(self, pointers: np.ndarray):
        self.pointers = pointers.astype(np.uint32)

    def log_densities(self, observations: np.ndarray) -> np.ndarray:
        raise NotImplementedError("子类需实现 log_densities")

    def densities(self, observations: np.ndarray) -> np.ndarray:
        # 默认实现：exp(log_densities)，子类可覆盖以加速
        return np.exp(self.log_densities(observations))


class DiscreteObservationModel(ObservationModel):
    """
    离散观测模型。给定一个形状为 (num_states, num_obs_types) 的矩阵
    observation_probabilities[i, j] = P(obs_type=j | state=i)
    pointers 对应为 np.arange(num_states)
    """

    def __init__(self, observation_probabilities: np.ndarray):
        # observation_probabilities: shape = (num_states, num_obs_types) 或 (num_obs_types, num_states)？
        # 为了与原实现保持一致，这里假设传入矩阵 shape=(num_states, num_obs_types)，
        # 且某些地方会索引为 observation_probabilities[state, obs].
        op = np.asarray(observation_probabilities, dtype=np.float64)
        # 检查按行求和是否都为 1
        if not np.allclose(op.sum(axis=1), 1.0):
            raise ValueError("Not a probability distribution in DiscreteObservationModel.")
        # pointers 简单地把每个状态映射到对应的行
        pointers = np.arange(op.shape[0], dtype=np.uint32)
        super().__init__(pointers)
        self.observation_probabilities = op

    def densities(self, observations: np.ndarray) -> np.ndarray:
        """
        observations: 1D 整数数组，取值范围是 [0, num_obs_types-1]
        返回 shape = (T, num_states) 的概率矩阵
        """
        T = observations.shape[0]
        num_states = self.observation_probabilities.shape[0]
        out = np.zeros((T, num_states), dtype=np.float64)
        for t in range(T):
            obs = observations[t]
            # 第 t 个观测类型对应所有状态下的概率分布取出作为列
            # observation_probabilities[:, obs] 如果观测矩阵是 (num_states, num_obs_types)，
            # 则要写成 op[:, obs]。这里 op 的 shape=(num_states, num_obs_types)，
            # 所以第一维是状态，第二维是观测类型
            out[t, :] = self.observation_probabilities[:, obs]
        return out

    def log_densities(self, observations: np.ndarray) -> np.ndarray:
        return np.log(self.densities(observations))


# ------------------------------------------------------------------------------
# Numba 加速的核心算法：viterbi, forward, forward_generator
# ------------------------------------------------------------------------------

@njit
def _viterbi_kernel(tm_states, tm_pointers, tm_log_probs,
                    om_pointers, om_log_densities, init_log_dist,
                    num_states, num_obs):
    """
    Numba 内部函数，实现 Viterbi 算法的核心。
    参数：
      tm_states:         uint32[ num_transitions ]，
      tm_pointers:       uint32[ num_states+1 ]，
      tm_log_probs:      float64[ num_transitions ]，对数转移概率
      om_pointers:       uint32[ num_states ]，指向观测概率矩阵列索引
      om_log_densities:  float64[ num_obs, num_states ]，对数观测概率
      init_log_dist:     float64[ num_states ]，初始分布的对数概率
      num_states:        uint32，状态数
      num_obs:           uint32，观测序列长度
    返回：
      path:     uint32[ num_obs ]，最优状态序列
      log_prob: float64，最优路径对数概率
    """
    # 分配临时数组
    curr_v = np.empty(num_states, dtype=np.float64)
    prev_v = np.empty(num_states, dtype=np.float64)
    # backtracking 矩阵，存储每一时刻每个 state 下最好的前一个 state
    bt = np.empty((num_obs, num_states), dtype=np.uint32)

    # 初始化 prev_v 为初始对数概率
    for s in range(num_states):
        prev_v[s] = init_log_dist[s]

    # 对每个时间步执行
    for t in range(num_obs):
        # 对每个状态 s，计算来自所有可能 prev_state 的最大值
        for s in range(num_states):
            curr_v[s] = -np.inf
        for s in range(num_states):
            # 取出观测对数概率
            d = om_log_densities[t, om_pointers[s]]
            # 遍历所有可从 prev_state 转移到 s
            start = tm_pointers[s]
            end = tm_pointers[s+1]
            best_val = -np.inf
            best_prev = np.uint32(0)
            for idx in range(start, end):
                ps = tm_states[idx]             # previous state
                val = prev_v[ps] + tm_log_probs[idx] + d
                if val > best_val:
                    best_val = val
                    best_prev = ps
            curr_v[s] = best_val
            bt[t, s] = best_prev
        # 将 curr_v 复制到 prev_v，以备下次迭代
        for s in range(num_states):
            prev_v[s] = curr_v[s]

    # 寻找终止时刻的最优状态
    last_state = np.uint32(0)
    best_final = -np.inf
    for s in range(num_states):
        if prev_v[s] > best_final:
            best_final = prev_v[s]
            last_state = np.uint32(s)

    # 如果概率为 -inf，说明没有可行路径
    if best_final == -np.inf:
        return np.empty(0, dtype=np.uint32), best_final

    # 回溯得到最优路径
    path = np.empty(num_obs, dtype=np.uint32)
    curr_state = last_state
    for t in range(num_obs-1, -1, -1):
        path[t] = curr_state
        curr_state = bt[t, curr_state]

    return path, best_final


@njit
def _forward_kernel(tm_states, tm_pointers, tm_probs,
                    om_pointers, om_densities, init_dist,
                    num_states, num_obs):
    """
    Numba 内部函数，实现 Forward 算法 (按时刻归一化的版本)。
    参数：
      tm_states:     uint32[ num_transitions ]
      tm_pointers:   uint32[ num_states+1 ]
      tm_probs:      float64[ num_transitions ]，非对数转移概率
      om_pointers:   uint32[ num_states ]
      om_densities:  float64[ num_obs, num_states ]，观测概率
      init_dist:     float64[ num_states ]，初始分布（未对数化）
      num_states:    uint32
      num_obs:       uint32
    返回：
      forward_probs: float64[ num_obs, num_states ]，按时刻归一化后的前向概率矩阵
    """
    fwd = np.zeros((num_obs, num_states), dtype=np.float64)
    prev_f = np.empty(num_states, dtype=np.float64)
    # 初始化 prev_f 为初始分布
    for s in range(num_states):
        prev_f[s] = init_dist[s]

    # 按时刻迭代
    for t in range(num_obs):
        # 先全部置零
        for s in range(num_states):
            fwd[t, s] = 0.0
        # 计算未归一化的前向概率并累积 sum
        norm_sum = 0.0
        for s in range(num_states):
            # 计算来自所有 prev_state 累积的概率
            tmp = 0.0
            start = tm_pointers[s]
            end = tm_pointers[s+1]
            for idx in range(start, end):
                ps = tm_states[idx]
                tmp += prev_f[ps] * tm_probs[idx]
            # 乘以观测概率
            val = tmp * om_densities[t, om_pointers[s]]
            fwd[t, s] = val
            norm_sum += val
        # 归一化，并更新 prev_f
        if norm_sum == 0.0:
            # 整个矩阵都为零，\sum 为 0，要避免除以 0，这里跳过
            # 但在实际应用里，这表明概率完全为 0，需要警告
            for s in range(num_states):
                prev_f[s] = 0.0
            continue
        inv_norm = 1.0 / norm_sum
        for s in range(num_states):
            fwd[t, s] *= inv_norm
            prev_f[s] = fwd[t, s]

    return fwd


@njit
def _forward_generator_kernel(
    tm_states, tm_pointers, tm_probs,
    om_pointers, om_densities_block,
    init_dist, num_states,
    rel_idx   # 块内相对索引（0 ≤ rel_idx < block_size）
):
    """
    Numba 内部函数：按块计算并返回“当前 rel_idx 时刻”的归一化 forward 概率。
    om_densities_block: float64[ block_size, num_states ]
    rel_idx:            块内相对行索引，指示要计算本块的第 rel_idx 帧。
    """
    # 先拷贝上一时刻的 forward 分布
    prev_f = np.empty(num_states, dtype=np.float64)
    for s in range(num_states):
        prev_f[s] = init_dist[s]

    # 本时刻用到的归一化概率
    cur_f = np.zeros(num_states, dtype=np.float64)
    norm_sum = 0.0

    # 逐个状态累加
    for s in range(num_states):
        tmp = 0.0
        # 累加从所有可能的前一个状态 ps -> s 的转移
        for idx in range(tm_pointers[s], tm_pointers[s+1]):
            ps = tm_states[idx]
            tmp += prev_f[ps] * tm_probs[idx]
        # **关键**：用块内第 rel_idx 行的观测概率（而不是 t-start_idx）
        val = tmp * om_densities_block[rel_idx, om_pointers[s]]
        cur_f[s] = val
        norm_sum += val

    # 归一化
    if norm_sum > 0.0:
        inv_norm = 1.0 / norm_sum
        for s in range(num_states):
            cur_f[s] *= inv_norm

    return prev_f, cur_f, norm_sum


# ------------------------------------------------------------------------------
# HiddenMarkovModel 类：暴露给用户的接口，内部调用上述 Numba 加速内核
# ------------------------------------------------------------------------------

class HiddenMarkovModel:
    """
    Hidden Markov Model，使用 Numba 加速 Viterbi 和 Forward 算法。
    参数：
      transition_model: TransitionModel 实例
      observation_model: ObservationModel 子类实例
      initial_distribution: 1D float64 数组，长度为 num_states，和为 1
    """

    def __init__(self, transition_model: TransitionModel,
                 observation_model: ObservationModel,
                 initial_distribution: np.ndarray = None):
        self.transition_model = transition_model
        self.observation_model = observation_model

        if initial_distribution is None:
            # 默认均匀分布
            num_s = self.transition_model.num_states
            initial_distribution = np.ones(num_s, dtype=np.float64) / num_s
        else:
            initial_distribution = np.asarray(initial_distribution, dtype=np.float64)
            if initial_distribution.ndim != 1 or \
               initial_distribution.shape[0] != self.transition_model.num_states or \
               not np.allclose(initial_distribution.sum(), 1.0):
                raise ValueError("Initial distribution must be length=num_states and sum to 1.")
        self.initial_distribution = initial_distribution
        # 保存对数形式的初始分布
        self._init_log_dist = np.log(self.initial_distribution)

        # 用于 forward_generator 状态维护
        self._prev_forward = self.initial_distribution.copy()

    def viterbi(self, observations: np.ndarray):
        """
        用 Viterbi 算法解码最优状态序列，返回 (path, log_prob)。
        observations: 1D uint32 数组，每个值代表一个离散观测类型索引。
        """
        # 提取 TransitionModel 所需的数组
        tm = self.transition_model
        tm_states = tm.states
        tm_pointers = tm.pointers
        tm_log_probs = tm.log_probabilities

        # 提取 ObservationModel 对数观测概率矩阵
        om = self.observation_model
        om_pointers = om.pointers
        # 先计算所有观测时刻的对数观测概率：形状 (T, num_states)
        om_log_densities = om.log_densities(observations)

        num_states = np.uint32(tm.num_states)
        num_obs = np.uint32(len(observations))

        path, logp = _viterbi_kernel(tm_states, tm_pointers, tm_log_probs,
                                     om_pointers, om_log_densities,
                                     self._init_log_dist,
                                     num_states, num_obs)
        if logp == -np.inf:
            warnings.warn("Viterbi 解码得到 -inf，表示无可行路径。", RuntimeWarning)
        return path, logp

    def forward(self, observations: np.ndarray, reset: bool = True):
        """
        计算标准的前向算法，返回归一化后的前向概率矩阵 shape=(T, num_states)。
        若 reset=True，则从初始分布开始；否则接着上次状态。
        observations: 2D numpy array 或 1D，具体看 ObservationModel 的 densities 接口。
        对于离散模型，应传入 1D 数组；对于连续模型，应传入相应的多维数组。
        """
        # 先获取观测概率：若 ObservationModel 重写了 densities，直接调用
        # 这里假设 observation_model.densities 返回形状 (T, num_states) 的矩阵
        om = self.observation_model
        obs_probs = om.densities(observations)  # float64 型，shape = (T, num_states)
        T, S = obs_probs.shape

        if reset:
            self._prev_forward = self.initial_distribution.copy()

        tm = self.transition_model
        tm_states = tm.states
        tm_pointers = tm.pointers
        tm_probs = tm.probabilities
        om_pointers = om.pointers

        num_states = np.uint32(S)
        num_obs = np.uint32(T)

        fwd_matrix = _forward_kernel(tm_states, tm_pointers, tm_probs,
                                     om_pointers, obs_probs,
                                     self._prev_forward, num_states, num_obs)
        # 更新 _prev_forward 为最后一步结果
        if T > 0:
            self._prev_forward = fwd_matrix[-1].copy()
        return fwd_matrix

    def forward_generator(self, observations: np.ndarray, block_size: int = None):
        """
        逐帧计算归一化的前向概率。修复后版本：
          - om_densities_block 取自 obs_probs[t:t+bs]
          - 直接传入 rel_idx 而非 start_idx，让内核精确索引。
        """
        om = self.observation_model
        # observations 是 1D 数组（离散观测），用 densities 得到 (T, S)
        obs_probs = om.densities(observations)  # float64, shape = (T, num_states)

        prev_fwd = self.initial_distribution.copy()
        T, S = obs_probs.shape
        if block_size is None:
            block_size = T

        t = 0
        while t < T:
            bs = min(block_size, T - t)
            # 本块的观测概率，shape=(bs, S)
            obs_block = obs_probs[t : t + bs, :]

            for i in range(bs):
                init_arr = prev_fwd.copy()
                # 传 rel_idx=i，代表使用 obs_block[i] 这一行
                _, cur_fwd, _ = _forward_generator_kernel(
                    self.transition_model.states,
                    self.transition_model.pointers,
                    self.transition_model.probabilities,
                    om.pointers,
                    obs_block,
                    init_arr,
                    np.uint32(S),
                    np.uint32(i)   # *** 块内相对索引 ***
                )
                yield cur_fwd.copy()
                prev_fwd = cur_fwd
            t += bs

    def reset(self, initial_distribution: np.ndarray = None):
        """
        重置 HMM 的前向状态到初始分布。
        """
        if initial_distribution is None:
            self._prev_forward = self.initial_distribution.copy()
        else:
            initial_distribution = np.asarray(initial_distribution, dtype=np.float64)
            if initial_distribution.ndim != 1 or initial_distribution.shape[0] != self.transition_model.num_states:
                raise ValueError("initial_distribution 维度不正确。")
            if not np.allclose(initial_distribution.sum(), 1.0):
                raise ValueError("initial_distribution 必须是概率分布。")
            self._prev_forward = initial_distribution.copy()


# alias
HMM = HiddenMarkovModel
