# nonlin_v4.py -- full nonlinear volume + five-shape-moduli evolution on the v11 leading cosh background
# (constant V = 3 M_P^2 h^2, AAA Standard Model twisted Ewald force, shape-independent zero-mode energy xi).
# Written from scratch by agent shape-blind (R16 V4), 2026-10-15.
#
# Units: tau = h t, l = L/L_b, L_b^4 = C_b/(3 pi^2 M_P^2 h^2), C_b = (1-xi) Z_eff(1).
#   d ln l/dtau = u,  du/dtau = 2 - 2u^2 - S/6,  S = l^-6 Tr(G^-1 Y G^-1 Y)
#   dG/dtau = 2 l^-3 Y,  dY/dtau = 2 l^-3 Y G^-1 Y + (6/(C_b l)) [G Z_G G - Tr(G Z_G) G/3]
#   constraint 3u^2 = 3 - 3 (Z(G) - xi Z(1))/(C_b l^4) + S/2.
# Canonical data: normal coordinates beta_A (G = exp(2 sum beta_A X_A)), p_A = l^3 (g dbeta/dtau)_A, [beta,p] = i eps_q.
# Exact map: in the eigenbasis of B = sum beta X, Y'_ij = P'_ij e^{b_i+b_j}/sinhc(b_i-b_j)   ('canonical');
# v11 convention Y = G^{1/2} P G^{1/2} is available as 'v11'.
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import sys, json, time, argparse, hashlib
import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOTB = os.path.dirname(HERE)
RES = os.path.join(ROOTB, 'results')
SHARDS = os.path.join(RES, 'shards')
sys.path.insert(0, HERE)
from ewald_v4 import zeff, shape_basis, expm_sym  # noqa: E402
from linear_v4 import transfer, Gfun, floor_of, eps_q, hessian_along, TAU_INF  # noqa: E402

X = shape_basis()
TAU_END = 12.0
LMIN = 1e-3
DEV_SEED = 20261015
NDEV = 4096
DEVFILE = os.path.join(RES, 'deviates_seed%d.npy' % DEV_SEED)


def sym6(v):
    return np.array([[v[0], v[3], v[4]], [v[3], v[1], v[5]], [v[4], v[5], v[2]]])


def six(M):
    return np.array([M[0, 0], M[1, 1], M[2, 2], 0.5 * (M[0, 1] + M[1, 0]), 0.5 * (M[0, 2] + M[2, 0]), 0.5 * (M[1, 2] + M[2, 1])])


def sinhc(x):
    x = np.asarray(x, float)
    out = 1.0 + x * x / 6.0 + x ** 4 / 120.0
    m = np.abs(x) > 1e-3
    out[m] = np.sinh(x[m]) / x[m]
    return out


class Model:
    def __init__(self, xi=0.30):
        self.xi = xi
        self.Z1 = zeff(np.eye(3), grad=False)[0]
        self.Cb = (1.0 - xi) * self.Z1
        self.kE = 3.0 * hessian_along(X[0]) / self.Cb
        self.kT = 3.0 * hessian_along(X[2]) / self.Cb
        self.eps = eps_q(self.Z1, xi)[0]

    def rhs(self, tau, y):
        lnl, u = y[0], y[1]
        G = sym6(y[2:8])
        Y = sym6(y[8:14])
        l3i = np.exp(-3.0 * lnl)
        Gi = np.linalg.inv(G)
        GiY = Gi @ Y
        S = l3i * l3i * np.trace(GiY @ GiY)
        _, dZ = zeff(G)
        GdZ = G @ dZ
        force = GdZ @ G - (np.trace(GdZ) / 3.0) * G
        dY = 2.0 * l3i * (Y @ GiY) + (6.0 / (self.Cb * np.exp(lnl))) * force
        out = np.empty(14)
        out[0] = u
        out[1] = 2.0 - 2.0 * u * u - S / 6.0
        out[2:8] = six(2.0 * l3i * Y)
        out[8:14] = six(dY)
        return out

    def gates(self, y):
        lnl, u = y[0], y[1]
        G = sym6(y[2:8])
        Y = sym6(y[8:14])
        Gi = np.linalg.inv(G)
        GiY = Gi @ Y
        S = np.exp(-6.0 * lnl) * np.trace(GiY @ GiY)
        Z = zeff(G, grad=False)[0]
        cons = 3 * u * u - 3 + 3 * (Z - self.xi * self.Z1) / (self.Cb * np.exp(4 * lnl)) - S / 2
        return float(cons), float(np.linalg.det(G) - 1.0), float(np.trace(GiY))

    def initial(self, tx, beta, p, conv='canonical'):
        Bm = np.tensordot(beta, X, 1)
        Pm = np.tensordot(p, X, 1)
        b, O = np.linalg.eigh(Bm)
        G0 = (O * np.exp(2.0 * b)) @ O.T
        Pp = O.T @ Pm @ O
        fac = np.exp(b[:, None] + b[None, :])
        if conv == 'canonical':
            fac = fac / sinhc(b[:, None] - b[None, :])
        Y0 = O @ (Pp * fac) @ O.T
        Gi = np.linalg.inv(G0)
        sp = float(np.trace(Gi @ Y0 @ Gi @ Y0))
        Zh = (zeff(G0, grad=False)[0] - self.xi * self.Z1) / self.Cb
        u0 = np.tanh(2.0 * tx)
        c = 1.0 - u0 * u0
        x = np.sqrt(Zh / c)
        for _ in range(60):
            dx = (c * x ** 3 - Zh * x + sp / 6.0) / (3 * c * x * x - Zh)
            x -= dx
            if abs(dx) < 1e-16 * x:
                break
        y0 = np.concatenate([[0.5 * np.log(x), u0], six(G0), six(Y0)])
        return y0

    @staticmethod
    def beta_of(G):
        w, O = np.linalg.eigh(G)
        Bm = (O * (0.5 * np.log(w))) @ O.T
        return np.array([np.trace(Bm @ X[a]) for a in range(5)])

    def run(self, tx, beta, p, direction='fwd', rtol=1e-10, atol=1e-12, conv='canonical', tau_end=TAU_END):
        y0 = self.initial(tx, beta, p, conv)
        tend = tau_end if direction == 'fwd' else -tau_end
        sgn = 1.0 if direction == 'fwd' else -1.0

        def ev_lam(t, y):
            return np.linalg.eigvalsh(sym6(y[2:8]))[0] - LMIN
        ev_lam.terminal = True

        def ev_vol(t, y):
            return y[0] - np.log(0.3)
        ev_vol.terminal = True
        t0 = time.time()
        try:
            sol = solve_ivp(self.rhs, (tx, tend), y0, method='DOP853', rtol=rtol, atol=atol, events=[ev_lam, ev_vol])
            status, msg = sol.status, sol.message
            yend, tlast = sol.y[:, -1], float(sol.t[-1])
            lam_steps = float(min(np.linalg.eigvalsh(sym6(sol.y[2:8, j]))[0] for j in range(sol.y.shape[1])))
            nfev = int(sol.nfev)
            hit_lam = len(sol.t_events[0]) > 0
            hit_vol = len(sol.t_events[1]) > 0
            # sign changes of u along the path (turns of the volume)
            us = sol.y[1]
            nturn = int(np.sum(np.sign(us[1:]) != np.sign(us[:-1])))
        except Exception as exc:  # integrator blow-up counts as failure
            status, msg, yend, tlast, lam_steps, nfev, hit_lam, hit_vol, nturn = -9, repr(exc), y0, tx, np.nan, -1, False, False, -1
        uend = float(yend[1])
        reached = (status == 0) and abs(tlast - tend) < 1e-9
        if hit_lam:
            reason = 'collapse_lambda_min'
        elif hit_vol:
            reason = 'volume_recollapse'
        elif not reached:
            reason = 'integrator:' + str(msg)[:60]
        elif sgn * uend <= 0.9:
            reason = 'u_end'
        else:
            reason = 'ok'
        surv = (reason == 'ok') and (lam_steps > LMIN)
        try:
            bout = self.beta_of(sym6(yend[2:8])).tolist()
        except Exception:
            bout = [np.nan] * 5
        g = self.gates(yend) if reached else (np.nan, np.nan, np.nan)
        return dict(surv=bool(surv), reason=reason, t_last=tlast, u_end=uend, lnl_end=float(yend[0]),
                    beta_out=bout, lam_min=lam_steps, nturn=nturn, cons=g[0], det_err=g[1], tr_err=g[2],
                    nfev=nfev, wall=time.time() - t0, lnl0=float(y0[0]))


def deviates():
    if not os.path.exists(DEVFILE):
        os.makedirs(RES, exist_ok=True)
        np.save(DEVFILE, np.random.default_rng(DEV_SEED).standard_normal((NDEV, 10)))
    return np.load(DEVFILE)


def widths(model, tx):
    flE, _ = floor_of(model.kE)
    flT, _ = floor_of(model.kT)
    gE, gT = Gfun(model.kE, tx, flE), Gfun(model.kT, tx, flT)
    return gE, gT


def draw(model, z, sE, sT, scale=1.0):
    s = np.array([sE, sE, sT, sT, sT])
    beta = scale * np.sqrt(model.eps * s) * z[:5]
    p = scale * np.sqrt(model.eps / (4.0 * s)) * z[5:]
    return beta, p


def sha(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def shard_name(tag, tx, d, i0):
    return os.path.join(SHARDS, '%s_tx%+.3f_%s_%05d.json' % (tag, tx, d, i0))


def run_shard(task):
    tx, d, i0, i1, tag, xi, rtol, conv = (task[k] for k in ('tx', 'dir', 'i0', 'i1', 'tag', 'xi', 'rtol', 'conv'))
    path = shard_name(tag, tx, d, i0)
    if os.path.exists(path):
        return path, 0.0
    t0 = time.time()
    model = Model(xi)
    Z = np.load(DEVFILE)[i0:i1]
    gE, gT = widths(model, tx)
    rowE = gE['A_row'] if d == 'fwd' else gE['B_row']
    rowT = gT['A_row'] if d == 'fwd' else gT['B_row']
    recs = []
    for j, z in enumerate(Z):
        beta, p = draw(model, z, gE['s_opt'], gT['s_opt'])
        r = model.run(tx, beta, p, d, rtol=rtol, atol=rtol * 1e-2, conv=conv)
        lin = [(rowE if a < 2 else rowT)[0] * beta[a] + (rowE if a < 2 else rowT)[1] * p[a] for a in range(5)]
        r.update(i=i0 + j, beta0=beta.tolist(), p0=p.tolist(), beta_lin=lin)
        recs.append(r)
    meta = dict(tx=tx, dir=d, i0=i0, i1=i1, tag=tag, xi=xi, rtol=rtol, conv=conv, eps_q=model.eps,
                kE=model.kE, kT=model.kT, sE=gE['s_opt'], sT=gT['s_opt'], GE=gE['G'], dev_seed=DEV_SEED,
                code_sha256={f: sha(os.path.join(HERE, f)) for f in ('nonlin_v4.py', 'ewald_v4.py', 'linear_v4.py')},
                finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), wall=time.time() - t0)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(dict(meta=meta, recs=recs), fh, default=float)
    os.replace(tmp, path)
    return path, time.time() - t0


def cmd_test():
    sys.stdout.reconfigure(encoding='utf-8')
    deviates()
    m = Model(0.30)
    print('kE %.6f kT %.6f eps %.6e Cb %.6f' % (m.kE, m.kT, m.eps, m.Cb))
    t0 = time.time()
    G = expm_sym(2 * np.tensordot([0.05, -0.03, 0.01, 0.02, -0.01], X, 1))
    for _ in range(200):
        zeff(G)
    print('zeff time per call %.1f us' % ((time.time() - t0) / 200 * 1e6))
    Z = np.load(DEVFILE)
    for tx in (0.0, -0.38, 0.2):
        gE, gT = widths(m, tx)
        for d in ('fwd', 'bwd'):
            row = gE['A_row'] if d == 'fwd' else gE['B_row']
            rowT = gT['A_row'] if d == 'fwd' else gT['B_row']
            beta, p = draw(m, Z[0], gE['s_opt'], gT['s_opt'], scale=1e-5)
            r = m.run(tx, beta, p, d, rtol=1e-11, atol=1e-16)
            lin = np.array([(row if a < 2 else rowT)[0] * beta[a] + (row if a < 2 else rowT)[1] * p[a] for a in range(5)])
            rel = np.max(np.abs(np.array(r['beta_out']) - lin)) / np.max(np.abs(lin))
            print('small-amp tx %+.2f %s: max rel dev from linear %.2e  surv %s gates %.1e %.1e %.1e nfev %d wall %.2f s' % (
                tx, d, rel, r['surv'], r['cons'], r['det_err'], r['tr_err'], r['nfev'], r['wall']))
    for tx in (0.0, -0.2, -0.38, -0.5, -1.0):
        gE, gT = widths(m, tx)
        for i in range(3):
            beta, p = draw(m, Z[i], gE['s_opt'], gT['s_opt'])
            r = m.run(tx, beta, p, 'fwd')
            print('tx %+.2f draw %d: surv %s %s t_last %.3f u_end %.4f lam_min %.2e |b_out| %.3f nturn %d gates %.1e %.1e %.1e nfev %d wall %.2f' % (
                tx, i, r['surv'], r['reason'], r['t_last'], r['u_end'], r['lam_min'], np.linalg.norm(r['beta_out'][:2]),
                r['nturn'], r['cons'], r['det_err'], r['tr_err'], r['nfev'], r['wall']))


def cmd_run(a):
    sys.stdout.reconfigure(encoding='utf-8')
    deviates()
    os.makedirs(SHARDS, exist_ok=True)
    txs = [float(x) for x in a.tx.split(',')]
    dirs = a.dirs.split(',')
    tasks = []
    for d in dirs:
        for tx in txs:
            for i0 in range(a.n0, a.n1, a.shard):
                tasks.append(dict(tx=tx, dir=d, i0=i0, i1=min(a.n1, i0 + a.shard), tag=a.tag, xi=a.xi,
                                  rtol=a.rtol, conv=a.conv))
    tasks = [t for t in tasks if not os.path.exists(shard_name(t['tag'], t['tx'], t['dir'], t['i0']))]
    log = open(os.path.join(RES, 'run_%s.log' % a.tag), 'a', encoding='utf-8')
    log.write('%s start %d tasks pid %d args %s\n' % (time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), len(tasks), os.getpid(), vars(a)))
    log.flush()
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        for k, (path, wt) in enumerate(pool.imap_unordered(run_shard, tasks)):
            log.write('%s done %d/%d %s %.1fs\n' % (time.strftime('%H:%M:%S', time.gmtime()), k + 1, len(tasks), os.path.basename(path), wt))
            log.flush()
    log.write('%s ALL DONE\n' % time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    log.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['test', 'run'])
    ap.add_argument('--tx', default='-1,-0.5,-0.38,-0.2,0,0.2,0.5')
    ap.add_argument('--dirs', default='fwd')
    ap.add_argument('--n0', type=int, default=0)
    ap.add_argument('--n1', type=int, default=256)
    ap.add_argument('--shard', type=int, default=32)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--tag', default='A')
    ap.add_argument('--xi', type=float, default=0.30)
    ap.add_argument('--rtol', type=float, default=1e-10)
    ap.add_argument('--conv', default='canonical')
    a = ap.parse_args()
    if a.mode == 'test':
        cmd_test()
    else:
        cmd_run(a)
