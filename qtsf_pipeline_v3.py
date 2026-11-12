#!/usr/bin/env python3
"""
QTSF-Meta v3: leakage-free, tuned and extended benchmark (v2 + forecast combinations)
========================================================
Changes relative to the first release (see manuscript, Sections 5-6):
  * hyper-parameter tuning added, using ONLY pre-cutoff information (origins C-3..C-1 with
    meta-training cut at C-3); the same grid is given to every tunable model;
  * new baselines: Theta, global pooled ridge-AR, classical MAML variants (5-p linear, 19-p with bias),
    VQC-scratch trained to convergence (Adam, 100 steps), Ridge-AR with standardised inputs and tuned penalty;
  * Holm correction, year-block bootstrap, results on common origins, by-origin and by-region tables.
Requires numpy, scipy, pandas, scikit-learn.  Usage:
  python qtsf_pipeline_v3.py --data bps_clean.csv --test_start 2017 --folds 5 --seeds 5 --out results_v2
"""
import argparse, json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings('ignore', category=Warning, module='sklearn')
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import wilcoxon

# ----------------------------------------------------------------------------------------
# 1. Exact 4-qubit state-vector simulator (batched over parameter sets and samples)
# ----------------------------------------------------------------------------------------
N_Q = 4
DIM = 2 ** N_Q


def _cnot_chain_inverse_perm(n=N_Q):
    """Sequential CNOT chain (0->1, 1->2, ..., n-2 -> n-1) as an index permutation."""
    mapping = np.zeros(2 ** n, dtype=int)
    for b in range(2 ** n):
        bits = [(b >> (n - 1 - q)) & 1 for q in range(n)]
        for c in range(n - 1):
            if bits[c]:
                bits[c + 1] ^= 1
        mapping[b] = sum(bit << (n - 1 - q) for q, bit in enumerate(bits))
    inv = np.zeros_like(mapping)
    inv[mapping] = np.arange(2 ** n)       # psi_new[:, mapping[b]] = psi[:, b]
    return inv


_INV = _cnot_chain_inverse_perm()
_Z0 = np.array([1 - 2 * ((b >> (N_Q - 1)) & 1) for b in range(DIM)], dtype=float)  # <Z_1> signs


def vqc_forward(thetas, X):
    """thetas: (M, L, n, 2) [Ry, Rz]; X: (N, n) in [-1, 1].  Returns (M, N) <Z_1>.

    Encoding: RY(arccos(x_j))|0>  (pure angle encoding, product state).
    Ansatz  : L x [ (x_j Rz(th_z) Ry(th_y)) -> CNOT chain ].
    Observable: Z on the first qubit only (local cost).
    """
    thetas = np.asarray(thetas, float)
    M, L, n, _ = thetas.shape
    a = np.arccos(np.clip(X, -1.0, 1.0))                       # (N, n)
    q = np.stack([np.cos(a / 2), np.sin(a / 2)], axis=-1)       # (N, n, 2)
    psi = q[:, 0]
    for j in range(1, n):
        psi = (psi[:, :, None] * q[:, j][:, None, :]).reshape(X.shape[0], -1)  # (N, 2^n)
    psi = np.broadcast_to(psi.astype(complex), (M,) + psi.shape).copy()        # (M, N, 2^n)
    for l in range(L):
        ty, tz = thetas[:, l, :, 0], thetas[:, l, :, 1]         # (M, n)
        c, s = np.cos(ty / 2), np.sin(ty / 2)
        ry = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)          # (M, n, 2, 2)
        z0 = np.exp(-0.5j * tz); z1 = np.exp(0.5j * tz); zero = np.zeros_like(z0)
        rz = np.stack([np.stack([z0, zero], -1), np.stack([zero, z1], -1)], -2)
        u = rz @ ry                                             # (M, n, 2, 2)
        layer = np.einsum('mab,mcd,mef,mgh->macegbdfh', u[:, 0], u[:, 1], u[:, 2], u[:, 3])
        layer = layer.reshape(M, DIM, DIM)
        psi = np.einsum('mij,mnj->mni', layer, psi)
        psi = psi[:, :, _INV]
    return (np.abs(psi) ** 2) @ _Z0                              # (M, N)


def vqc_grad(theta, X, y):
    """MSE loss + exact gradient via the parameter-shift rule (every parameter enters one gate)."""
    flat = theta.reshape(-1)
    P = flat.size
    shifts = np.zeros((2 * P + 1, P))
    for k in range(P):
        shifts[1 + 2 * k, k] = np.pi / 2
        shifts[2 + 2 * k, k] = -np.pi / 2
    allth = (flat[None] + shifts).reshape((2 * P + 1,) + theta.shape)
    out = vqc_forward(allth, X)                                   # (2P+1, N)
    f = out[0]
    df = 0.5 * (out[1::2] - out[2::2])                            # (P, N)
    r = f - y
    return float(np.mean(r ** 2)), (2.0 * df @ r / len(y)).reshape(theta.shape)


# ----------------------------------------------------------------------------------------
# 2. Parameter-matched classical network (16 params): 4 -> 3 tanh -> 1, no hidden bias
# ----------------------------------------------------------------------------------------
def nn_unpack(p, n=N_Q, h=3):
    return p[:n * h].reshape(n, h), p[n * h:n * h + h], p[-1]


def nn_forward(p, X):
    W1, w2, b2 = nn_unpack(p, X.shape[1])
    return np.tanh(np.tanh(X @ W1) @ w2 + b2)


def nn_grad(p, X, y):
    W1, w2, b2 = nn_unpack(p, X.shape[1])
    h = np.tanh(X @ W1)
    z = h @ w2 + b2
    f = np.tanh(z)
    r = f - y
    dz = 2 * r * (1 - f ** 2) / len(y)
    gw2 = h.T @ dz
    gb2 = dz.sum()
    dh = np.outer(dz, w2) * (1 - h ** 2)
    gW1 = X.T @ dh
    return float(np.mean(r ** 2)), np.concatenate([gW1.ravel(), gw2, [gb2]])


# ----------------------------------------------------------------------------------------
# 2b. Further classical parametric models + common Model interface
#     'nn'  : 4-3-1 tanh, no hidden bias, 16 params (parameter-matched to the VQC)
#     'nnb' : 4-3-1 tanh with hidden bias, 19 params
#     'lin' : linear AR(4) with intercept, 5 params
# ----------------------------------------------------------------------------------------
def nnb_unpack(p, n=N_Q, h=3):
    return p[:n * h].reshape(n, h), p[n * h:n * h + h], p[n * h + h:n * h + 2 * h], p[-1]


def nnb_forward(p, X):
    W1, b1, w2, b2 = nnb_unpack(p, X.shape[1])
    return np.tanh(np.tanh(X @ W1 + b1) @ w2 + b2)


def nnb_grad(p, X, y):
    W1, b1, w2, b2 = nnb_unpack(p, X.shape[1])
    h = np.tanh(X @ W1 + b1)
    f = np.tanh(h @ w2 + b2)
    r = f - y
    dz = 2 * r * (1 - f ** 2) / len(y)
    dh = np.outer(dz, w2) * (1 - h ** 2)
    return float(np.mean(r ** 2)), np.concatenate([(X.T @ dh).ravel(), dh.sum(0), h.T @ dz, [dz.sum()]])


def lin_forward(p, X):
    return X @ p[:-1] + p[-1]


def lin_grad(p, X, y):
    r = lin_forward(p, X) - y
    d = 2 * r / len(y)
    return float(np.mean(r ** 2)), np.concatenate([X.T @ d, [d.sum()]])


class Model:
    """Common interface: init(rng) -> params ; forward ; grad."""
    def __init__(self, kind, L=2):
        self.kind, self.L = kind, L

    def init(self, rng):
        if self.kind == 'vqc':
            return rng.uniform(-np.pi, np.pi, size=(self.L, N_Q, 2)) * 0.1
        if self.kind == 'nn':
            return rng.normal(0, 0.5, size=N_Q * 3 + 3 + 1)
        if self.kind == 'nnb':
            return rng.normal(0, 0.5, size=N_Q * 3 + 3 + 3 + 1)
        return rng.normal(0, 0.1, size=N_Q + 1)

    def n_params(self):
        return {'vqc': self.L * N_Q * 2, 'nn': N_Q * 3 + 3 + 1, 'nnb': N_Q * 3 + 3 + 3 + 1, 'lin': N_Q + 1}[self.kind]

    def forward(self, p, X):
        if self.kind == 'vqc':
            return vqc_forward(p[None], X)[0]
        return {'nn': nn_forward, 'nnb': nnb_forward, 'lin': lin_forward}[self.kind](p, X)

    def grad(self, p, X, y):
        if self.kind == 'vqc':
            return vqc_grad(p, X, y)
        return {'nn': nn_grad, 'nnb': nnb_grad, 'lin': lin_grad}[self.kind](p, X, y)


# ----------------------------------------------------------------------------------------
# 3. Data handling (no look-ahead)
# ----------------------------------------------------------------------------------------
def load_panel(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    piv = df.pivot(index='year', columns='region', values='value').sort_index()
    if piv.isna().any().any():
        raise ValueError('Panel has missing values; impute or drop regions explicitly (and report it).')
    if (piv <= 0).any().any():
        raise ValueError('Values must be strictly positive for log-growth modelling.')
    return piv


def demo_panel(n_regions=35, y0=2002, y1=2025, seed=0):
    """SYNTHETIC poverty-rate-like panel (declining, noisy, region-specific). For smoke tests only."""
    rng = np.random.default_rng(seed)
    years = np.arange(y0, y1 + 1)
    base = rng.uniform(8, 25, n_regions)
    drift = rng.normal(-0.03, 0.01, n_regions)
    cols = {}
    for r in range(n_regions):
        g = drift[r] + 0.6 * rng.normal(0, 0.02, len(years))
        g[0] = 0
        g[years == 2020] += 0.04
        cols[f'Region_{r:02d}'] = base[r] * np.exp(np.cumsum(g))
    return pd.DataFrame(cols, index=years)


def growth(v):
    return np.diff(np.log(v))


def make_windows(g, w):
    """g: growth sequence. Returns X (n_win, w), y (n_win,) and index of target growth."""
    if len(g) <= w:
        return np.zeros((0, w)), np.zeros(0)
    X = np.stack([g[i:i + w] for i in range(len(g) - w)])
    y = g[w:]
    return X, y


def scaler(g_train, k=3.0):
    mu = float(np.mean(g_train))
    sd = float(np.std(g_train)) + 1e-8
    return mu, k * sd


def to_unit(x, mu, s):
    return np.clip((x - mu) / s, -1.0, 1.0)


def from_unit(z, mu, s):
    return z * s + mu


# ----------------------------------------------------------------------------------------
# 4. Training routines
# ----------------------------------------------------------------------------------------
def adam_step(p, g, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    m, v, t = state
    t += 1
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
    return p - lr * mh / (np.sqrt(vh) + eps), (m, v, t)


def adapt(model, p0, X, y, steps, lr):
    p = p0.copy()
    for _ in range(steps):
        _, g = model.grad(p, X, y)
        p = p - lr * g
    return p


def region_tasks(panel, regions, cutoff, w):
    """Task data for meta-training: only target years <= cutoff. Split chronologically 60/40."""
    tasks = []
    for r in regions:
        v = panel.loc[panel.index <= cutoff, r].values
        g = growth(v)
        mu, s = scaler(g)
        X, y = make_windows(g, w)
        Xu, yu = to_unit(X, mu, s), to_unit(y, mu, s)
        k = max(2, int(0.6 * len(yu)))
        if len(yu) - k < 2:
            continue
        tasks.append((Xu[:k], yu[:k], Xu[k:], yu[k:]))
    return tasks


def maml_train(model, tasks, rng, iters, meta_bs, inner_steps, alpha, beta):
    """First-order MAML (FOMAML) with Adam on the meta-parameters."""
    p = model.init(rng)
    state = (np.zeros_like(p), np.zeros_like(p), 0)
    for _ in range(iters):
        idx = rng.choice(len(tasks), size=min(meta_bs, len(tasks)), replace=False)
        g_meta = np.zeros_like(p)
        for i in idx:
            Xs, ys, Xq, yq = tasks[i]
            pi = adapt(model, p, Xs, ys, inner_steps, alpha)
            _, gq = model.grad(pi, Xq, yq)
            g_meta += gq / len(idx)
        p, state = adam_step(p, g_meta, state, beta)
    return p


def pooled_train(model, tasks, rng, iters, lr):
    """Non-meta ablation: ordinary pooled training on the same meta-train windows."""
    X = np.concatenate([np.concatenate([t[0], t[2]]) for t in tasks])
    y = np.concatenate([np.concatenate([t[1], t[3]]) for t in tasks])
    p = model.init(rng)
    state = (np.zeros_like(p), np.zeros_like(p), 0)
    for _ in range(iters):
        _, g = model.grad(p, X, y)
        p, state = adam_step(p, g, state, lr)
    return p


# ----------------------------------------------------------------------------------------
# 5. Forecasters. Each returns h-step-ahead predicted LOG-GROWTHS for a series v[:T] (levels).
#    All are recursive in growth space => identical protocol.
# ----------------------------------------------------------------------------------------
def recursive(predict_one, g_hist, w, H):
    buf = list(g_hist[-w:])
    out = []
    for _ in range(H):
        nxt = float(predict_one(np.array(buf[-w:])[None])[0])
        out.append(nxt)
        buf.append(nxt)
    return np.array(out)


def fc_naive(v, H, **kw):
    return np.zeros(H)


def fc_drift(v, H, **kw):
    g = growth(v)
    return np.full(H, g.mean())


def fc_damped_holt(v, H, **kw):
    """Gardner-McKenzie damped additive trend on log-levels; parameters by SSE (scipy)."""
    y = np.log(v)

    def run(par, ret=False):
        a, b, phi = par
        l, bt = y[0], y[1] - y[0]
        sse = 0.0
        for t in range(1, len(y)):
            f = l + phi * bt
            e = y[t] - f
            sse += e * e
            ln = f + a * e
            bt = phi * bt + a * b * e
            l = ln
        if ret:
            return l, bt
        return sse

    best = None
    for x0 in ([0.5, 0.2, 0.9], [0.8, 0.1, 0.95]):
        r = minimize(run, x0, bounds=[(0.01, 0.99), (0.01, 0.99), (0.8, 0.98)], method='L-BFGS-B')
        if best is None or r.fun < best.fun:
            best = r
    l, bt = run(best.x, ret=True)
    phi = best.x[2]
    cum, out, prev = 0.0, [], y[-1]
    fc = []
    for h in range(1, H + 1):
        cum += phi ** h
        fc.append(l + cum * bt)
    fc = np.array(fc)
    return np.diff(np.concatenate([[y[-1]], fc]))


def fc_arima_drift(v, H, max_p=2, **kw):
    """ARIMA(p,1,0)+drift: AR(p) with intercept on log-growth; p by AICc (least squares)."""
    g = growth(v)
    best = None
    for p in range(0, max_p + 1):
        if len(g) - p < p + 3:
            continue
        if p == 0:
            c, res, coef = g.mean(), g - g.mean(), np.array([])
        else:
            A = np.column_stack([np.ones(len(g) - p)] + [g[p - k - 1:len(g) - k - 1] for k in range(p)])
            beta, *_ = np.linalg.lstsq(A, g[p:], rcond=None)
            res = g[p:] - A @ beta
            c, coef = beta[0], beta[1:]
        n, k = len(res), p + 2
        aicc = n * np.log(np.mean(res ** 2) + 1e-12) + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1)
        if best is None or aicc < best[0]:
            best = (aicc, c, coef)
    _, c, coef = best
    buf, out = list(g), []
    for _ in range(H):
        nxt = c + sum(coef[k] * buf[-k - 1] for k in range(len(coef)))
        out.append(nxt)
        buf.append(nxt)
    return np.array(out)


def fc_ridge_ar(v, H, w=4, alpha=1.0, **kw):
    g = growth(v)
    ww = min(w, max(1, len(g) - 4))
    X, y = make_windows(g, ww)
    mu = X.mean(0)
    A = np.column_stack([np.ones(len(X)), X - mu])
    reg = alpha * np.eye(A.shape[1]); reg[0, 0] = 0
    beta = np.linalg.solve(A.T @ A + reg, A.T @ y)
    return recursive(lambda Z: beta[0] + (Z - mu) @ beta[1:], g, ww, H)


def fc_model(model, p, v, H, w, **kw):
    """Forecast with a (adapted) parametric model working in the unit-scaled growth space."""
    g = growth(v)
    mu, s = scaler(g)
    pred = lambda Z: from_unit(model.forward(p, to_unit(Z, mu, s)), mu, s)
    return recursive(pred, g, w, H)


def adapt_on_task(model, p0, v, w, steps, lr):
    g = growth(v)
    mu, s = scaler(g)
    X, y = make_windows(g, w)
    if len(y) < 2:
        return p0
    return adapt(model, p0, to_unit(X, mu, s), to_unit(y, mu, s), steps, lr)


def fc_pooled_mlp(v_all_train_regions, v, H, w, seed):
    """Non-meta classical NN (sklearn MLP, 1 hidden layer of 8 units) pooled over training regions
    plus the test region's own history up to T. L2-regularised. Standardisation from training only."""
    from sklearn.neural_network import MLPRegressor
    Xs, ys = [], []
    for series in v_all_train_regions + [v]:
        g = growth(series)
        mu, s = scaler(g)
        X, y = make_windows(g, w)
        if len(y):
            Xs.append((X - mu) / s); ys.append((y - mu) / s)
    X, y = np.concatenate(Xs), np.concatenate(ys)
    m = MLPRegressor(hidden_layer_sizes=(8,), alpha=1e-2, max_iter=500, solver='lbfgs', random_state=seed)
    m.fit(X, y)
    g = growth(v)
    mu, s = scaler(g)
    return recursive(lambda Z: from_unit(m.predict((Z - mu) / s), mu, s), g, w, H)


# ----------------------------------------------------------------------------------------
# 5b. Additional / redefined forecasters (all recursive in log-growth space)
# ----------------------------------------------------------------------------------------
def fc_theta(v, H, **kw):
    """Standard Theta method (Assimakopoulos & Nikolopoulos 2000; Hyndman & Billah 2003) on log-levels:
    SES with drift = half the OLS slope."""
    y = np.log(v); n = len(y)

    def sse(a):
        l, s = y[0], 0.0
        for t in range(1, n):
            e = y[t] - l; s += e * e; l = l + a[0] * e
        return s

    a = float(minimize(sse, [0.5], bounds=[(0.01, 0.99)], method='L-BFGS-B').x[0])
    l = y[0]
    for t in range(1, n):
        l = l + a * (y[t] - l)
    b0 = np.polyfit(np.arange(1, n + 1), y, 1)[0]
    fc = np.array([l + 0.5 * b0 * (h - 1 + 1 / a - ((1 - a) ** n) / a) for h in range(1, H + 1)])
    return np.diff(np.concatenate([[y[-1]], fc]))


def _ridge_fit(X, y, alpha):
    A = np.column_stack([np.ones(len(X)), X])
    reg = alpha * np.eye(A.shape[1]); reg[0, 0] = 0
    return np.linalg.solve(A.T @ A + reg, A.T @ y)


def fc_ridge_ar(v, H, w=4, alpha=1.0, **kw):
    """Local ridge-AR on STANDARDISED growths (penalty is therefore scale-free and tunable)."""
    g = growth(v); mu, s = scaler(g)
    X, y = make_windows(g, w)
    beta = _ridge_fit((X - mu) / s, (y - mu) / s, alpha)
    return recursive(lambda Z: from_unit_free(beta[0] + ((Z - mu) / s) @ beta[1:], mu, s), g, w, H)


def from_unit_free(z, mu, s):
    return z * s + mu


def fc_global_ar(train_series, v, H, w, alpha=1.0, **kw):
    """Global linear AR: ridge on windows pooled over training regions (data <= cutoff) and the test region's own
    history up to T; each series standardised with its own training statistics (the linear counterpart of MLP-pooled)."""
    Xs, ys = [], []
    for series in train_series + [v]:
        g = growth(series); mu, s = scaler(g); X, y = make_windows(g, w)
        if len(y):
            Xs.append((X - mu) / s); ys.append((y - mu) / s)
    beta = _ridge_fit(np.concatenate(Xs), np.concatenate(ys), alpha)
    g = growth(v); mu, s = scaler(g)
    return recursive(lambda Z: from_unit_free(beta[0] + ((Z - mu) / s) @ beta[1:], mu, s), g, w, H)


def fc_pooled_mlp(v_all_train_regions, v, H, w, seed, alpha=1e-2):
    from sklearn.neural_network import MLPRegressor
    Xs, ys = [], []
    for series in v_all_train_regions + [v]:
        g = growth(series); mu, s = scaler(g); X, y = make_windows(g, w)
        if len(y):
            Xs.append((X - mu) / s); ys.append((y - mu) / s)
    m = MLPRegressor(hidden_layer_sizes=(8,), alpha=alpha, max_iter=500, solver='lbfgs', random_state=seed)
    m.fit(np.concatenate(Xs), np.concatenate(ys))
    g = growth(v); mu, s = scaler(g)
    return recursive(lambda Z: from_unit(m.predict((Z - mu) / s), mu, s), g, w, H)


def adapt_adam(model, p0, v, w, steps, lr):
    """Train to convergence on the test region's own history (used for the 'scratch (long)' ablation)."""
    g = growth(v); mu, s = scaler(g); X, y = make_windows(g, w)
    if len(y) < 2:
        return p0
    X, y = to_unit(X, mu, s), to_unit(y, mu, s)
    p, st = p0.copy(), (np.zeros_like(p0), np.zeros_like(p0), 0)
    for _ in range(steps):
        _, gr = model.grad(p, X, y)
        p, st = adam_step(p, gr, st, lr)
    return p


# ----------------------------------------------------------------------------------------
# 6. Tuning on pre-cutoff information only
# ----------------------------------------------------------------------------------------
MAML_GRID = dict(alpha=[0.03, 0.1, 0.3], k=[3, 5, 10])
RIDGE_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
MLP_GRID = [1e-3, 1e-2, 1e-1, 1.0]


def levels_from_growth(v_T, gpred):
    return v_T * np.exp(np.cumsum(gpred))


def mase_scale(v):
    return np.mean(np.abs(np.diff(v))) + 1e-12


def tune(panel, args, folds, log):
    """Select hyper-parameters on origins C-3..C-1 whose targets are <= C (C = first test origin).
    Meta-training for the tuning stage uses target years <= C-3 only. No year > C is touched."""
    regions = list(panel.columns); C = args.test_start - 1; Ct = C - 3
    H, w = args.horizon, args.window
    origins = list(range(Ct, C))

    def score(r, T, gp):
        v = panel.loc[panel.index <= T, r].values
        tr = panel.loc[(panel.index > T) & (panel.index <= min(T + H, C)), r].values
        lv = levels_from_growth(v[-1], gp[:len(tr)])
        return float(np.mean(np.abs(lv - tr)) / mase_scale(v))

    rows, best = [], {}
    kinds = {'VQC': Model('vqc', args.layers), 'NN16': Model('nn'), 'NN19': Model('nnb'), 'LIN5': Model('lin')}
    for name, model in kinds.items():
        for a in MAML_GRID['alpha']:
            for k in MAML_GRID['k']:
                sc = []
                for fi, test_idx in enumerate(folds):
                    test_regs = [regions[i] for i in test_idx]
                    train_regs = [r for r in regions if r not in test_regs]
                    tasks = region_tasks(panel, train_regs, Ct, w)
                    p = maml_train(model, tasks, np.random.default_rng(777 + fi), args.meta_iters, args.meta_bs, k, a, args.beta)
                    for r in test_regs:
                        for T in origins:
                            v = panel.loc[panel.index <= T, r].values
                            pa = adapt_on_task(model, p, v, w, k, a)
                            sc.append(score(r, T, fc_model(model, pa, v, H, w)))
                rows.append(dict(model=name, alpha=a, k=k, val_MASE=np.mean(sc)))
                log(f'  tune {name} alpha={a} k={k}: {np.mean(sc):.4f}')
        sub = [x for x in rows if x['model'] == name]
        b = min(sub, key=lambda x: x['val_MASE']); best[name] = (b['alpha'], b['k'])
    # ridge-AR (local) over all regions
    for al in RIDGE_GRID:
        sc = [score(r, T, fc_ridge_ar(panel.loc[panel.index <= T, r].values, H, w=w, alpha=al))
              for r in regions for T in origins]
        rows.append(dict(model='RidgeAR', alpha=al, k=np.nan, val_MASE=np.mean(sc)))
    best['RidgeAR'] = min([x for x in rows if x['model'] == 'RidgeAR'], key=lambda x: x['val_MASE'])['alpha']
    # global AR and pooled MLP (fold-dependent)
    for nm, grid in (('GlobalAR', RIDGE_GRID), ('MLP', MLP_GRID)):
        for al in grid:
            sc = []
            for fi, test_idx in enumerate(folds):
                test_regs = [regions[i] for i in test_idx]
                train_series = [panel.loc[panel.index <= Ct, r].values for r in regions if r not in test_regs]
                for r in test_regs:
                    for T in origins:
                        v = panel.loc[panel.index <= T, r].values
                        gp = (fc_global_ar(train_series, v, H, w, alpha=al) if nm == 'GlobalAR'
                              else fc_pooled_mlp(train_series, v, H, w, 0, alpha=al))
                        sc.append(score(r, T, gp))
            rows.append(dict(model=nm, alpha=al, k=np.nan, val_MASE=np.mean(sc)))
            log(f'  tune {nm} alpha={al}: {np.mean(sc):.4f}')
        best[nm] = min([x for x in rows if x['model'] == nm], key=lambda x: x['val_MASE'])['alpha']
    return best, pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# 7. Main evaluation
# ----------------------------------------------------------------------------------------
def run(panel, args, hp, log):
    t0 = time.time()
    regions = list(panel.columns); last = int(panel.index.max())
    test_origins = list(range(args.test_start - 1, last)); cutoff = args.test_start - 1
    H, w = args.horizon, args.window
    perm = np.random.default_rng(12345).permutation(len(regions))
    folds = np.array_split(perm, args.folds)
    vqc, nn16, nnb, lin = Model('vqc', args.layers), Model('nn'), Model('nnb'), Model('lin')
    aV, kV = hp['VQC']; a16, k16 = hp['NN16']; a19, k19 = hp['NN19']; aL, kL = hp['LIN5']
    rows = []
    for fi, test_idx in enumerate(folds):
        ck = os.path.join(args.out, f'partial_fold{fi}.csv')
        if os.path.exists(ck):
            rows.extend(pd.read_csv(ck).to_dict('records')); log(f'[fold {fi+1}] loaded checkpoint'); continue
        n0 = len(rows)
        test_regs = [regions[i] for i in test_idx]
        train_regs = [r for r in regions if r not in test_regs]
        train_series = [panel.loc[panel.index <= cutoff, r].values for r in train_regs]
        det = {}
        for r in test_regs:
            for T in test_origins:
                v = panel.loc[panel.index <= T, r].values
                det[(r, T)] = {'Naive': fc_naive(v, H), 'Drift': fc_drift(v, H), 'DampedHolt': fc_damped_holt(v, H),
                               'ARIMA(p,1,0)+drift': fc_arima_drift(v, H), 'Theta': fc_theta(v, H),
                               'RidgeAR': fc_ridge_ar(v, H, w=w, alpha=hp['RidgeAR']),
                               'GlobalAR': fc_global_ar(train_series, v, H, w, alpha=hp['GlobalAR'])}
        log(f'[fold {fi+1}/{len(folds)}] deterministic models done ({time.time()-t0:.0f}s)')
        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 * fi + seed)
            def meta(model, a, k):
                return maml_train(model, region_tasks(panel, train_regs, cutoff, w), rng, args.meta_iters,
                                  args.meta_bs, k, a, args.beta)
            p_v, p_16, p_19, p_l = meta(vqc, aV, kV), meta(nn16, a16, k16), meta(nnb, a19, k19), meta(lin, aL, kL)
            p_pool = pooled_train(vqc, region_tasks(panel, train_regs, cutoff, w), rng, args.meta_iters, args.beta)
            p_init = vqc.init(rng)
            for r in test_regs:
                for T in test_origins:
                    v = panel.loc[panel.index <= T, r].values
                    truth = panel.loc[(panel.index > T) & (panel.index <= T + H), r].values
                    h_eff = len(truth)
                    if h_eff == 0:
                        continue
                    sc = mase_scale(v)
                    fcs = dict(det[(r, T)])
                    fcs['MLP-pooled'] = fc_pooled_mlp(train_series, v, H, w, seed, alpha=hp['MLP'])
                    fcs['MAML-Linear-5p'] = fc_model(lin, adapt_on_task(lin, p_l, v, w, kL, aL), v, H, w)
                    fcs['MAML-Classical-16p'] = fc_model(nn16, adapt_on_task(nn16, p_16, v, w, k16, a16), v, H, w)
                    fcs['MAML-Classical-19p'] = fc_model(nnb, adapt_on_task(nnb, p_19, v, w, k19, a19), v, H, w)
                    fcs['VQC-scratch(no meta)'] = fc_model(vqc, adapt_on_task(vqc, p_init, v, w, kV, aV), v, H, w)
                    fcs['VQC-scratch(long)'] = fc_model(vqc, adapt_adam(vqc, p_init, v, w, 100, 0.05), v, H, w)
                    fcs['VQC-pooled(no MAML)'] = fc_model(vqc, adapt_on_task(vqc, p_pool, v, w, kV, aV), v, H, w)
                    fcs['QTSF-Meta(VQC+FOMAML)'] = fc_model(vqc, adapt_on_task(vqc, p_v, v, w, kV, aV), v, H, w)
                    fcs['Ens(QTSF+Theta)'] = (fcs['QTSF-Meta(VQC+FOMAML)'] + fcs['Theta']) / 2   # equal-weight forecast combinations
                    fcs['Ens(QTSF+Drift)'] = (fcs['QTSF-Meta(VQC+FOMAML)'] + fcs['Drift']) / 2
                    fcs['Ens(Theta+Drift)'] = (fcs['Theta'] + fcs['Drift']) / 2
                    for name, gp in fcs.items():
                        lv = levels_from_growth(v[-1], gp[:h_eff])
                        for h in range(h_eff):
                            e = lv[h] - truth[h]
                            rows.append(dict(fold=fi, seed=seed, region=r, origin=T, h=h + 1, model=name,
                                             ape=abs(e) / truth[h] * 100, ase=abs(e) / sc, se=e ** 2))
            log(f'[fold {fi+1}/{len(folds)}] seed {seed+1}/{args.seeds} done ({time.time()-t0:.0f}s)')
        pd.DataFrame(rows[n0:]).to_csv(ck, index=False)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# 8. Summaries
# ----------------------------------------------------------------------------------------
REF = 'QTSF-Meta(VQC+FOMAML)'


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run_max = 0.0
    for k, i in enumerate(order):
        run_max = max(run_max, (m - k) * p[i]); adj[i] = min(1.0, run_max)
    return adj


def paired(g, h, last, H, rng, common=True):
    sub = g[g.h == h]
    if common:
        sub = sub[sub.origin <= last - H]
    per_reg = sub.groupby(['model', 'region']).ase.mean().unstack(0)
    per_org = sub.groupby(['model', 'origin', 'region']).ase.mean().unstack(0)
    out = []
    for m in per_reg.columns:
        if m == REF:
            continue
        d = per_reg[REF] - per_reg[m]
        try:
            p = wilcoxon(d.values).pvalue
        except ValueError:
            p = np.nan
        boots = [rng.choice(d.values, len(d)).mean() for _ in range(5000)]
        do = (per_org[REF] - per_org[m]).groupby(level='origin').mean()
        bo = [rng.choice(do.values, len(do)).mean() for _ in range(5000)]
        out.append(dict(h=h, competitor=m, mean_MASE_diff=d.mean(), ci_low=np.percentile(boots, 2.5),
                        ci_high=np.percentile(boots, 97.5), ci_year_low=np.percentile(bo, 2.5),
                        ci_year_high=np.percentile(bo, 97.5), wilcoxon_p=p, n_regions=len(d),
                        qtsf_better_regions=int((d < 0).sum()), n_origins=len(do),
                        qtsf_better_origins=int((do < 0).sum())))
    res = pd.DataFrame(out); res['p_holm'] = holm(res.wilcoxon_p.values)
    return res


def summarise(df, out, last, H):
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, 'raw_forecast_errors.csv'), index=False)
    g = df.groupby(['model', 'h', 'region', 'origin', 'fold'], as_index=False)[['ape', 'ase', 'se']].mean()
    g['target'] = g.origin + g.h

    def acc(x):
        return x.groupby(['model', 'h']).agg(MAPE=('ape', 'mean'), MASE=('ase', 'mean'),
                                              RMSE=('se', lambda z: float(np.sqrt(np.mean(z)))), n=('ase', 'size')).reset_index()
    acc(g).to_csv(os.path.join(out, 'table_accuracy_by_horizon.csv'), index=False)          # all available origins
    acc(g[g.origin <= last - H]).to_csv(os.path.join(out, 'table_accuracy_common_origins.csv'), index=False)
    g[g.h == 1].groupby(['model', 'target']).ase.mean().unstack(0).to_csv(os.path.join(out, 'table_mase_by_target_year_h1.csv'))
    g[g.h == 1].groupby(['model', 'region']).ase.mean().unstack(0).to_csv(os.path.join(out, 'table_mase_by_region_h1.csv'))
    seed_sd = df.groupby(['model', 'seed']).ase.mean().groupby('model').agg(['mean', 'std'])
    seed_sd.columns = ['MASE_mean_over_seeds', 'MASE_sd_over_seeds']
    seed_sd.to_csv(os.path.join(out, 'table_seed_variability.csv'))
    ens = [m for m in g.model.unique() if m.startswith('Ens(')]
    gm = g[~g.model.isin(ens)]
    rng = np.random.default_rng(0)
    pd.concat([paired(gm, h, last, H, rng) for h in (1, 2, 3)]).to_csv(os.path.join(out, 'table_paired_tests_common.csv'), index=False)
    paired(gm, 1, last, H, rng, common=False).to_csv(os.path.join(out, 'table_paired_tests_h1_allorigins.csv'), index=False)
    ge = g[g.model.isin([REF] + ens)]
    rng2 = np.random.default_rng(1)
    pd.concat([paired(ge, h, last, H, rng2) for h in (1, 2, 3)]).to_csv(os.path.join(out, 'table_paired_tests_ensembles.csv'), index=False)
    t = acc(g[g.origin <= last - H])
    print('\n=== MASE on common origins ===')
    print(t.pivot(index='model', columns='h', values='MASE').round(3).sort_values(1).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--test_start', type=int, default=2017)
    ap.add_argument('--folds', type=int, default=5); ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--window', type=int, default=4); ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--horizon', type=int, default=3)
    ap.add_argument('--meta_iters', type=int, default=300); ap.add_argument('--meta_bs', type=int, default=8)
    ap.add_argument('--beta', type=float, default=0.02)
    ap.add_argument('--out', default='results_v2')
    ap.add_argument('--reuse_tuning', action='store_true', help='reuse table_tuning_grid.csv and per-fold checkpoints in --out')
    a = ap.parse_args()
    if a.window != N_Q:
        sys.exit(f'window must equal n_qubits={N_Q}')
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, 'run.log'), 'a')
    def log(msg):
        print(msg, flush=True); logf.write(msg + '\n'); logf.flush()
    panel = load_panel(a.data)
    regions = list(panel.columns)
    perm = np.random.default_rng(12345).permutation(len(regions))
    folds = np.array_split(perm, a.folds)
    log('== tuning (pre-cutoff information only) ==')
    tg = os.path.join(a.out, 'table_tuning_grid.csv')
    if a.reuse_tuning and os.path.exists(tg):
        tune_df = pd.read_csv(tg); hp = {}
        for nm, sub in tune_df.groupby('model'):
            b = sub.loc[sub.val_MASE.idxmin()]
            hp[nm] = (float(b.alpha), int(b.k)) if nm in ('VQC', 'NN16', 'NN19', 'LIN5') else float(b.alpha)
    else:
        hp, tune_df = tune(panel, a, folds, log)
        tune_df.to_csv(tg, index=False)
    log(f'selected: {hp}')
    log('== main evaluation ==')
    df = run(panel, a, hp, log)
    summarise(df, a.out, int(panel.index.max()), a.horizon)
    cfg = vars(a); cfg['selected_hyperparameters'] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in hp.items()}
    cfg['grids'] = dict(maml=MAML_GRID, ridge=RIDGE_GRID, mlp=MLP_GRID)
    json.dump(cfg, open(os.path.join(a.out, 'config.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
