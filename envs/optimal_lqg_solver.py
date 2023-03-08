import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import List, Optional, Union, Dict
from scipy.optimize import minimize


def Riccati_solver(A: NDArray, B: NDArray, Q: NDArray, R: NDArray, gamma: float):
    assert np.allclose(Q, Q.T)
    assert np.allclose(R, R.T)
    assert 0 <= gamma <= 1
    d = len(A)
    def S_maker(x: NDArray):
        assert len(x) == d * (d + 1) / 2
        if d == 2:
            L = np.array([
                [x[0], 0], 
                [x[1], x[2]]
            ])
        elif d == 3:
            L = np.array([
                [x[0], 0, 0], 
                [x[1], x[2], 0], 
                [x[3], x[4], x[5]]
            ])
        elif d == 4:
            L = np.array([
                [x[0], 0, 0, 0], 
                [x[1], x[2], 0, 0], 
                [x[3], x[4], x[5], 0], 
                [x[6], x[7], x[8], x[9]]
            ])
        else:
            raise NotImplementedError
        return L @ L.T
    
    def S_loss(x: NDArray):
        S = S_maker(x)
        Z = S @ B @ np.linalg.inv(gamma * B.T @ S @ B + R) @ B.T @ S
        RHS = A.T @ (gamma * S - gamma ** 2 * Z) @ A + Q
        return ((S - RHS) ** 2).sum()
    
    while True:
        sol = minimize(S_loss, x0=np.random.uniform(size=d * (d + 1) // 2) * 2 - 1, 
                       method='Powell')
        if sol.fun < 1e-5:
            break
    S = S_maker(sol.x)
    K = gamma * np.linalg.inv(gamma * B.T @ S @ B + R) @ B.T @ S @ A
    return K


def run_LQG_repeat(A: NDArray, B: NDArray, 
                   Qs: List[NDArray], Rs: List[NDArray], 
                   w: ArrayLike, gamma: float, x_init: NDArray, 
                   n_repeat: int, T_horizon: int, noise_std: float) -> NDArray:
    # w: relative weight of objectives (e.g., w = [0.5, 0.1, 0.4])
    assert noise_std >= 0
    assert len(A) == len(x_init)
    if noise_std == 0:
        assert n_repeat == 1
    assert len(w) == len(Qs) == len(Rs)
    assert np.abs(np.sum(w) - 1) < 1e-3
    assert len(x_init) == len(w)
    
    Q = sum([w[i] * Qs[i] for i in range(len(w))])
    R = sum([w[i] * Rs[i] for i in range(len(w))])
    
    K = Riccati_solver(A, B, Q, R, gamma)
    J_all = []
    for _ in range(n_repeat):
        x = x_init
        cost = np.zeros(len(w), dtype=float)
        for t in range(T_horizon):
            u = - K @ x
            cost += gamma ** t * np.asarray([x @ Qs[i] @ x + u @ Rs[i] @ u 
                                             for i in range(len(w))])
            v = np.random.randn(len(w)) * noise_std
            x = A @ x + B @ u + v
        J_all += [cost]
        
    return - np.asarray(J_all).mean(axis=0)    # average over all episodes


def MO_LQG_return(xi: float, w: ArrayLike, gamma: float, 
                  x_init: NDArray, n_repeat: int, 
                  T_horizon: int, noise_std: float):
    """
    - Example usage:
    
    ws = np.arange(0.01, 1, 0.01)
    ws = np.hstack([ws.reshape(-1, 1), 1 - ws.reshape(-1, 1)])

    Rs_all = np.asarray([MO_LQG_return(xi=0.1, gamma=0.9, w=w, x_init=[10.0]*2, 
                                       n_repeat=1, T_horizon=50, noise_std=0)
             for w in ws])
    """
    assert 0 <= xi <= 1
    d = len(w)
    assert d > 1
    Qs = [None] * d
    for i in range(d):
        qq = xi * np.eye(d)
        qq[i, i] = 1 - xi
        Qs[i] = qq
    Rs = [None] * d
    for i in range(d):
        rr = (1 - xi) * np.eye(d)
        rr[i, i] = xi
        Rs[i] = rr
        
    A = np.eye(d)
    B = np.eye(d)
    JJ = run_LQG_repeat(A=A, B=B, Qs=Qs, Rs=Rs, w=w, gamma=gamma, 
                        x_init=x_init, n_repeat=n_repeat, 
                        T_horizon=T_horizon, noise_std=noise_std)
    return JJ








