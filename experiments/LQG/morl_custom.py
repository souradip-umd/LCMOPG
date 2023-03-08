import numpy as np
import gym
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union, Optional, Callable
from numpy.typing import ArrayLike, NDArray
from joblib import Parallel, delayed

import torch.nn as nn
import torch
from torch.distributions import Categorical, Beta
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.hv import Hypervolume
from sklearn.neighbors import KDTree



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if DEVICE == "cpu":
    print("No GPU found, using CPU.")



def is_dominated_by(x1: NDArray, x2: NDArray):
    assert len(x1) == len(x2)
    # bigger is better
    if np.allclose(x1, x2):
        return False
    else:
        out = True
        for k in range(len(x1)):
            if x1[k] > x2[k]:
                out = False
                break
        return out


def check_pf(Rs: ArrayLike):
    assert np.ndim(Rs) == 2
    result = []
    for i in range(len(Rs)):
        R = Rs[i]
        out = True
        for j in np.arange(len(Rs))[np.arange(len(Rs)) != i]:
            R_o = Rs[j]
            if is_dominated_by(R, R_o):
                out = False
                break
        result += [out]
    return result


def unique_PF_points(Rs: ArrayLike):
    _RR = np.array(Rs)[check_pf(Rs)]
    _P = _RR[deviation_from_pf(_RR) < 1e-3]
    _Punique = []
    for p in _P:
        if len(_Punique) == 0:
            _Punique += [p]
        else:
            for q in _Punique:
                if np.allclose(p, q):
                    break
            else:
                _Punique += [p]

    return np.asarray(_Punique)


def deviation_from_pf(Rs: ArrayLike):
    Rs = np.atleast_2d(Rs)
    pf = np.atleast_2d(Rs[check_pf(Rs)])
    ind = GDPlus(-pf)
    result = []
    for R in Rs:
        dist = [ind(-R)]
        for j in range(len(R)):
            dist += [np.max(pf[:, j]) - R[j]]
        result += [np.min(dist)]
    return np.array(result)


def feature_embedding(X: Union[torch.Tensor, NDArray], 
                      embed_dims: Union[ArrayLike, None],
                      device: torch.device) -> Union[torch.Tensor, NDArray]:
    """
    The elements of X are assumed to be roughly in the range [0, 1] but can go beyond it. 
    (Not a hard constraint.) 

    If only part of X's dimensions need to be embedded into a higher dimension, 
    use -1 to indicate un-embedded dimensions; 
    for example, let 
        embed_dims = [5, 5, -1, -1]
    to indicate that only the first two components need to be embedded; the last 
    two components will not be modified at all.
    """
    if embed_dims is not None:
        assert (np.asarray(embed_dims) == 1).any() == False, \
               "Embedding into 1 dimension is not accepted; set the dimension to -1 instead."

    if isinstance(X, np.ndarray) and np.ndim(X) == 1:
        if embed_dims is None:
            return X
        else:
            assert len(X) == len(embed_dims)
            assert len(embed_dims) > 1
            A = np.hstack([
                    [np.cos(k * np.pi * X[m]) for k in np.arange(1, 1 + embed_dims[m])]
                    if embed_dims[m] > 0 else [X[m]]
                    for m in range(len(embed_dims))
                ])
            return A
    else:
        if embed_dims is None:
            return X if isinstance(X, torch.Tensor) else torch.Tensor(X).to(device)
        else:
            if isinstance(X, np.ndarray):
                assert X.ndim == 2
                assert X.shape[1] == len(embed_dims)
                if max(embed_dims) == min(embed_dims):    # all elements are equal
                    K = embed_dims[0]
                    assert K > 1
                    A = torch.Tensor(np.hstack(
                            [np.cos(k * np.pi * X) for k in np.arange(1, 1 + K)]
                        )).to(device)
                    return A
                else:
                    A = torch.Tensor(np.hstack([np.hstack(
                            [np.cos(k * np.pi * X[:, m : m + 1]) for k in np.arange(1, 1 + embed_dims[m])]
                        ) if embed_dims[m] > 0 else X[:, m : m + 1] 
                        for m in range(len(embed_dims))])).to(device)
                    return A

            elif isinstance(X, torch.Tensor):
                assert X.ndim == 2
                assert X.size()[-1] == len(embed_dims)
                if max(embed_dims) == min(embed_dims):    # all elements are equal
                    K = embed_dims[0]
                    assert K > 1
                    A = torch.cat(
                            [torch.cos(k * np.pi * X) for k in np.arange(1, 1 + K)], dim=-1
                        )
                    return A
                else:
                    A = torch.cat([torch.cat(
                            [torch.cos(k * np.pi * X[:, m : m + 1]) for k in np.arange(1, 1 + embed_dims[m])],
                            dim=-1
                        ) if embed_dims[m] > 0 else X[:, m : m + 1] for m in range(len(embed_dims))], 
                        dim=-1)
                    return A
            else:
                raise NotImplementedError


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int,  hidden_dim: int, act_space: tuple, c_dim: int,
                 embed_dims: Union[ArrayLike, None]):
        """
        Example: 
        If act_space = ('discrete', 3), then the action can take values {0, 1, 2}.
        If act_space = ('continuous', 3), then the action is in [-1.0, 1.0]^3.

        When the action space is continuous, the action range is fixed to [-1.0, 1.0]^d. 
        The environment to which this policy is applied should be configured in advance 
        so that the full range of accepted action is [-1.0, 1.0]^d. 

        If embed_dims is None, no state embedding is performed.
        If embed_dims is a list of numbers, say [5, 7], then each dimension of a state vector 
        (which is 2-dimensional in this example) will be embedded into a 5- and 7-dimensional 
        space via cosine basis functions.
        """
        super().__init__()
        if embed_dims is not None:
            assert len(embed_dims) == state_dim
        assert act_space[0] in ['discrete', 'continuous']
        self.action_is_continuous = True if act_space[0] == 'continuous' else False
        assert hidden_dim % c_dim == 0
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.embed_dims = embed_dims
        
        if act_space[0] == 'continuous':
            self.output_dim = 2 * act_space[1]
        else:
            self.output_dim = act_space[1]    # logits
        
        self.c_dim = c_dim
        input_dim = self.state_dim if (self.embed_dims is None) \
                    else sum([x if x > 1 else 1 for x in self.embed_dims])
        self.lin_0 = nn.Linear(input_dim, self.hidden_dim)
        self.lin_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin_3 = nn.Linear(self.hidden_dim, self.output_dim)
        self.lin_A = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.act = nn.SELU()
    
    @staticmethod
    def _inv_sigmoid(x: float):
        return np.log(x / (1 - x))
    
    def forward(self, X: torch.Tensor, c: NDArray):
        assert X.ndim == 2
        assert 0 <= np.min(c) and np.max(c) <= 1
        c = np.atleast_2d(c)
        if len(c) == 1:
            c = np.repeat(c, len(X), axis=0)
        assert np.shape(c) == (len(X), self.c_dim)

        A = feature_embedding(c, embed_dims=[self.hidden_dim // self.c_dim] * self.c_dim, 
                              device=X.device)
        A = self.lin_A(A)
        A = nn.Tanh()(A)
        X = feature_embedding(X, embed_dims=self.embed_dims, device=X.device)
        X = self.act(self.lin_0(X))
        X = X * A
        X = self.act(self.lin_1(X))
        X = self.act(self.lin_2(X))
        X = self.lin_3(X)
        if self.action_is_continuous:
            X = nn.Sigmoid()(X + self._inv_sigmoid(1/8)) * 8.0
            alpha_s = X[:, :self.output_dim // 2]
            beta_s = X[:, self.output_dim // 2:]
            return alpha_s, beta_s
        else:
            return X    # logits
    
    def sample(self, X: torch.Tensor, c: NDArray):
        with torch.no_grad():
            if self.action_is_continuous:
                alpha_s, beta_s = self.forward(X, c)
                a = torch.zeros(len(X), self.output_dim // 2)
                for i in range(self.output_dim // 2):
                    a[:, i] = Beta(alpha_s[:, i], beta_s[:, i]).sample() * 2 - 1    # action is in the range [-1, +1]
                return a
            else:
                a = Categorical(logits=self.forward(X, c)).sample().item()
                return a
    
    def mean(self, X: torch.Tensor, c: NDArray):
        with torch.no_grad():
            if self.action_is_continuous:
                alpha_s, beta_s = self.forward(X, c)
                a = torch.zeros(len(X), self.output_dim // 2)
                for i in range(self.output_dim // 2):
                    a[:, i] = Beta(alpha_s[:, i], beta_s[:, i]).mean * 2 - 1    # action is in the range [-1, +1]
                return a
            else:
                Logit = self.forward(X, c)
                return torch.argmax(Logit, dim=-1).item()
    
    def log_prob(self, X: torch.Tensor, c: NDArray, a: torch.Tensor):
        assert len(X) == len(a)
        if self.action_is_continuous:
            assert - 1.0 <= a.min() and a.max() <= 1.0
            a_bare = torch.clamp(0.5 * (a + 1.0), min=1e-4, max=1-1e-4)
            alpha_s, beta_s = self.forward(X, c)
            log_pi = 0
            for i in range(self.output_dim // 2):
                log_pi += Beta(alpha_s[:, i], beta_s[:, i]).log_prob(a_bare[:, i])
            return log_pi
        else:
            assert a.dtype in (torch.int16, torch.int32, torch.int64)
            Logit = self.forward(X, c)
            return Categorical(logits=Logit).log_prob(a.flatten())


def make_policy_net(state_dim: int, hidden_dim: int, c_dim: int, 
                    act_space: tuple, seed: int, 
                    embed_dims: Union[ArrayLike, None], std: float = 0.2):
    assert std > 0.0
    torch.manual_seed(seed)
    policy = PolicyNet(state_dim=state_dim, hidden_dim=hidden_dim, act_space=act_space, 
                       c_dim=c_dim, embed_dims=embed_dims)
    
    def weights_init(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=std)
            torch.nn.init.normal_(m.bias, mean=0.0, std=std)

    policy.apply(weights_init)
    policy.to(DEVICE)
    return policy


def run_episode(policy: PolicyNet, env: gym.Env, c: NDArray, 
                deterministic: bool, n_obj: int, gamma: float):
    assert  0.0 < gamma <= 1.0
    trajectory = []
    env = deepcopy(env)
    s = env.reset()
    ep_ret = np.zeros(n_obj)
    count = 0
    with torch.no_grad():
        while True:
            if deterministic:
                a = policy.mean(X=torch.Tensor(np.atleast_2d(s)), c=c)
            else:
                a = policy.sample(X=torch.Tensor(np.atleast_2d(s)), c=c)
            
            if not isinstance(a, int):
                a = a.cpu().numpy().ravel()
                    
            trajectory += [np.hstack([s, a])]
            s, r, done, _ = env.step(a)
            ep_ret += r * gamma ** count
            count += 1
            if done:
                break
    return ep_ret, np.vstack(trajectory)


def group_evaluation(policy: PolicyNet, group_size: int, n_jobs: int, env: gym.Env, 
                     deterministic: bool, n_obj: int, gamma: float, seed: Optional[int] = None):
    if seed is None:
        cs = np.random.uniform(size=(group_size, policy.c_dim))
    else:
        cs = np.random.RandomState(seed).uniform(size=(group_size, policy.c_dim))
        
    c_policy = deepcopy(policy).to('cpu') if policy is not None else None
    record = Parallel(n_jobs=n_jobs)(delayed(run_episode)(*(c_policy, env, 
                                                            c, deterministic, n_obj, gamma)) for c in cs)
    return cs, record


def run_episode_REPEAT(policy: PolicyNet, env: gym.Env, c: float, 
                       deterministic: bool, n_obj: int, gamma: float, 
                       n_episode_repeat: int):
    Rs = []
    env = deepcopy(env)
    for _ in range(n_episode_repeat):
        s = env.reset()
        ep_ret = np.zeros(n_obj)
        count = 0
        with torch.no_grad():
            while True:
                if deterministic:
                    a = policy.mean(X=torch.Tensor(np.atleast_2d(s)), c=c)
                else:
                    a = policy.sample(X=torch.Tensor(np.atleast_2d(s)), c=c)
                if not isinstance(a, int):
                    a = a.cpu().numpy().ravel()
                    
                s, r, done, _ = env.step(a)
                ep_ret += r * gamma ** count
                count += 1
                if done:
                    break
        Rs += [ep_ret]
    return np.mean(Rs, axis=0)


def group_evaluation_REPEAT(policy: PolicyNet, group_size: int, n_jobs: int, env: gym.Env, 
                            deterministic: bool, n_obj: int, gamma: float,
                            n_episode_repeat: int, seed: Optional[int] = None):
    if seed is None:
        cs = np.random.uniform(size=(group_size, policy.c_dim))
    else:
        cs = np.random.RandomState(seed).uniform(size=(group_size, policy.c_dim))
        
    c_policy = deepcopy(policy).to('cpu') if policy is not None else None
    record = Parallel(n_jobs=n_jobs)(delayed(run_episode_REPEAT)\
                                     (*(c_policy, env, c, deterministic, n_obj, gamma,
                                        n_episode_repeat)) for c in cs)
    return cs, record


def HV(Rs: list, ref_point: NDArray):
    metric = Hypervolume(ref_point=-ref_point)
    hv = metric.do(-np.array(Rs))
    return hv    # higher is better


class FE_layer(nn.Module):
    # Feature embedding layer
    def __init__(self, embed_dims: Union[ArrayLike, None]):
        super().__init__()
        self.embed_dims = embed_dims

    def forward(self, X: torch.Tensor):
        A = feature_embedding(X, embed_dims=self.embed_dims, device=X.device)
        return A


class ValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, 
                 output_dim: int, embed_dims: Union[ArrayLike, None], std: float = 0.2):
        super().__init__()
        if embed_dims is None:
            IN_DIM = input_dim
        else:
            IN_DIM = np.sum([x if x > 0 else 1 for x in embed_dims])
        self.net = nn.Sequential(FE_layer(embed_dims), 
                                 nn.Linear(IN_DIM, hidden_size), nn.SELU(), 
                                 nn.Linear(hidden_size, hidden_size), nn.SELU(), 
                                 nn.Linear(hidden_size, output_dim)).to(DEVICE)
        self.output_dim = output_dim
        self.embed_dims = embed_dims
        def weights_init(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=std)
                torch.nn.init.normal_(m.bias, mean=0.0, std=std)
        
        self.net.apply(weights_init)
        self.opt = torch.optim.Adam(self.net.parameters())
    
    def fit(self, X: NDArray, y: NDArray, n_epochs: int, batch_size: int, 
            mask: Optional[NDArray] = None):
        assert (mask is not None and self.output_dim > 1) or (mask is None and self.output_dim == 1)
        
        if mask is None:
            assert len(X) == len(y) and self.output_dim == 1
            mask = torch.ones((len(X), 1))
        else:
            assert len(X) == len(y) == len(mask)
            mask = torch.Tensor(np.atleast_2d(mask))
            
        from torch.utils.data import DataLoader, TensorDataset
        idx = np.random.permutation(range(len(X)))
        
        self.dataloader = DataLoader(dataset=TensorDataset(torch.Tensor(X[idx]).to(DEVICE), 
                                                           torch.Tensor(y[idx]).to(DEVICE), 
                                                           mask[idx].to(DEVICE)),
                                     batch_size=batch_size, shuffle=True)
        self.loss = nn.MSELoss()
        
        for _ in range(n_epochs):
            for i, (x, y, mm) in enumerate(self.dataloader):
                self.opt.zero_grad()
                self.loss((self.net(x) * mm).sum(dim=1), y.flatten()).backward()
                self.opt.step()
                
    def predict(self, X: NDArray):
        with torch.no_grad():
            return self.net(torch.Tensor(X).to(DEVICE)).cpu().numpy()


def R_normalize(points: list, mode: str):
    ph = np.array(points)
            
    if mode == 'mean':
        ph -= np.mean(ph, axis=0)
        ww = np.std(ph, axis=0)
    elif mode == 'median':
        ph -= np.median(ph, axis=0)
        ww = np.quantile(ph, q=0.75, axis=0) - np.quantile(ph, q=0.25, axis=0)
    elif mode == 'maxmin':
        ph -= np.median(ph, axis=0)
        ww = np.max(ph, axis=0) - np.min(ph, axis=0)
    else:
        raise NotImplementedError

    for i in range(len(ww)):
        if ww[i] == 0.0:
            ww[i] = 1.0
    
    return ph / ww


def nn_distance(x: NDArray, k: int):
    assert np.ndim(x) == 2
    x = x + 1e-8 * np.random.random_sample(x.shape).astype(np.float32)
    tree = KDTree(x, leaf_size=10)
    dist = tree.query(x, k=k + 1, return_distance=True)[0][:, k]
    return dist


def learn(policy: PolicyNet, gradient_steps: int, n_grad_repeat: int, 
          group_size_train: int, group_size_test: int, 
          env: gym.Env, n_obj: int, 
          test_repeat: int, n_jobs: int, 
          ep_len_train: int, ep_len_test: int,
          use_QV: bool, QV_params: dict, nn_k: int, 
          period_to_record_R: int, 
          J_AU: NDArray, J_U: NDArray, gamma_train: float, gamma_test: float, 
          beta: float, mode: str, score_clipped_above_zero: bool, 
          print_unique_pf_points: Optional[bool] = False):
    
    assert mode in ['mean', 'median', 'maxmin']
    assert len(J_AU) == len(J_U) == n_obj
    from sklearn.preprocessing import OneHotEncoder
    
    if isinstance(env.action_space, gym.spaces.Discrete):
        act_space = ('discrete', env.action_space.n)
    elif isinstance(env.action_space, gym.spaces.Box):
        act_space = ('continuous', env.action_space.shape[0])
    
    if not use_QV:
        assert QV_params is None
    elif use_QV:
        B_hidden_size = QV_params['hidden_size']
        B_n_epochs = QV_params['n_epochs']
        B_batch_size = QV_params['batch_size']
    
    remaining_steps = gradient_steps
    optimizer = torch.optim.Adam(policy.parameters())
    history = []
    best_policy = None
    best_score = - np.inf

    if use_QV:
        if act_space[0] == 'continuous':
            QQ = ValueNet(input_dim=env.observation_space.shape[0] + act_space[1], 
                          output_dim=1, hidden_size=B_hidden_size, 
                          embed_dims=list(policy.embed_dims)+[-1]*act_space[1] if (policy.embed_dims is not None) else None)
            VV = ValueNet(input_dim=env.observation_space.shape[0], output_dim=1, 
                          hidden_size=B_hidden_size,
                          embed_dims=policy.embed_dims)
        elif act_space[0] == 'discrete':
            QQ = ValueNet(input_dim=env.observation_space.shape[0], output_dim=act_space[1], 
                          hidden_size=B_hidden_size,
                          embed_dims=policy.embed_dims)
            VV = ValueNet(input_dim=env.observation_space.shape[0], output_dim=1, 
                          hidden_size=B_hidden_size,
                          embed_dims=policy.embed_dims)
    
    evolution = dict({})
    iteration = 0
    
    # first evaluation ----------------------------------------------------------------------------
    Rs = group_evaluation_REPEAT(policy=policy, 
                                 group_size=group_size_test,
                                 n_jobs=n_jobs, 
                                 env=env.set_ep_len(ep_len_test),
                                 deterministic=True, n_obj=n_obj, gamma=gamma_test, 
                                 n_episode_repeat=test_repeat)[1]
    evolution[iteration] = np.array(Rs)
    Rs_normalized = [(R - J_AU) / (J_U - J_AU) for R in Rs]
    current_HV = HV(Rs=Rs_normalized, ref_point=np.zeros(n_obj))
    history += [current_HV]
    print('{} [{:.4}]'.format(0, current_HV), end=' ')
    # ---------------------------------------------------------------------------------------------
    
    while remaining_steps > 0:
        iteration += 1
        cs, record = group_evaluation(policy=policy, 
                                      group_size=group_size_train,
                                      env=env.set_ep_len(ep_len_train), 
                                      deterministic=False, n_obj=n_obj, n_jobs=n_jobs, gamma=gamma_train)
        Rs = [rec[0] for rec in record]
        R_norm = R_normalize(Rs, mode=mode)
        scores = - deviation_from_pf(R_norm)    # higher is better
        scores -= {'mean': np.mean, 'median': np.median, 'maxmin': np.median}[mode](scores)
        trajectories = [rec[1] for rec in record]
        
        bonus = nn_distance(R_norm, k=nn_k)
        mask = scores >= 0
        bonus = beta * bonus * mask
        
        if use_QV:
            y_train = np.repeat(scores + bonus, repeats=[len(traj) for traj in trajectories]).reshape(-1, 1)
            
            if act_space[0] == 'continuous':
                XA_train = np.vstack(trajectories)
                X_train = XA_train[:, :- act_space[1]]
                QQ.fit(X=XA_train, y=y_train, n_epochs=B_n_epochs, batch_size=B_batch_size)
                VV.fit(X=X_train, y=y_train, n_epochs=B_n_epochs, batch_size=B_batch_size)
                scores_p = QQ.predict(X=XA_train).ravel() - VV.predict(X=X_train).ravel()
            
            elif act_space[0] == 'discrete':
                X_train = np.vstack(trajectories)[:, :-1]
                a_train = np.vstack(trajectories)[:, -1].astype(int)
                m_train = OneHotEncoder(categories=[range(act_space[1])], sparse=False).fit_transform(a_train.reshape(-1, 1))
                QQ.fit(X=X_train, y=y_train, mask=m_train, n_epochs=B_n_epochs, batch_size=B_batch_size)
                VV.fit(X=X_train, y=y_train, mask=None, n_epochs=B_n_epochs, batch_size=B_batch_size)
                q_vals = QQ.predict(X=X_train)
                assert np.shape(q_vals) == (len(X_train), act_space[1])
                scores_p = np.array([_q_[a_train[j]] for (j, _q_) in enumerate(q_vals)]) - VV.predict(X=X_train).ravel()
            
            ep_L = [0] + list(np.cumsum([len(traj) for traj in trajectories]))
            scores_p = [scores_p[ep_L[j] : ep_L[j + 1]] for j in range(len(cs))]
        else:
            scores_p = scores + bonus
        
        for _ in range(n_grad_repeat):
            loss = 0
            for i in np.arange(len(cs)):
                c = cs[i]
                traj = trajectories[i]
                _s = traj[:, :env.observation_space.shape[0]]
                _a = traj[:, env.observation_space.shape[0]:]
                thres = 0.0 if score_clipped_above_zero == True else - 9999.9
                if use_QV:
                    sss = torch.clamp(torch.Tensor(scores_p[i]).to(DEVICE), thres)
                    assert len(sss) == len(traj)
                else:
                    sss = max(scores_p[i], thres)
                    
                loss -= (sss * policy.log_prob(X=torch.Tensor(_s).to(DEVICE), c=c, 
                                               a=torch.tensor(_a.astype(int if act_space[0]=='discrete' else float)).to(DEVICE))).sum()
                
            loss /= np.sum([len(traj) for traj in trajectories])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            remaining_steps -= 1
        
        # evaluation
        Rs = group_evaluation_REPEAT(policy=policy, 
                                     group_size=group_size_test,
                                     n_jobs=n_jobs, 
                                     env=env.set_ep_len(ep_len_test),
                                     deterministic=True, n_obj=n_obj, gamma=gamma_test, 
                                     n_episode_repeat=test_repeat)[1]
        
        if print_unique_pf_points:
            print(unique_PF_points(Rs))
            
        if iteration % period_to_record_R == 0:
            evolution[iteration] = np.array(Rs)        
        Rs_normalized = [(R - J_AU) / (J_U - J_AU) for R in Rs]
        current_HV = HV(Rs=Rs_normalized, ref_point=np.zeros(n_obj))
        history += [current_HV]
        
        if current_HV > best_score:
            best_policy = deepcopy(policy)
            best_score = current_HV
            
        print('{} [{:.4}]'.format(gradient_steps - remaining_steps, 
                                  current_HV), end=' ')
        
    return history, policy, best_policy, evolution





