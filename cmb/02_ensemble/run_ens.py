# PROVENANCE: 04_calculations/code/20261013_argon_codex_intake/cmb_ensemble/run_ens.py (copied 2026-10-15 by code-shape-cmb)
"""[Package note] C145 master driver: S1 regression/8-block checks, S2 E-plane builds + KL against an actual cube pool + Brent profile, S3 structure fit / edges / P(read|L0), S4 convergence and Planck-like noise; --smoke = reduced l <= 12, x_max = 60 configuration (path check), --refine = second round. Resumable.

cmb_ensemble master driver (C145/D117) -- WORKING COPY.

Orchestration derived from snapshot P5_CMB/high_cov.py (main/analyze; sha256 858F996C3C5CF02781EB5D836208DAAF29AB6EDCAB40A19F707F1218F5C95669)
and P5_CMB/compare_volume.py (sha256 E27805AE36AB399C037B925DFA0EE396D4EFEFFB0B691E99AB9945A84A80EE23);
kernels in ens_core.py (full provenance header there).  SNAPSHOT_INPUT is read-only; NEW_RUN_OUTPUT =
run01 (run01/smoke with --smoke).  Resumable: rerun the same command (cached covariances, inverses and
KL tables are reused, nothing finished is recomputed).
Stages: S1 regression / benchmark / 8-block verification at theta!=0; S2 E-plane shape builds, KL
against an actual cube pool, bracketed Brent profile; S3 D(A,theta,L0) structure fit, readability edges,
P(read|L0) and windows; S4 (runs in the Brent batch on idle workers) x320 / taper convergence and
Planck-like isotropic noise.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_v] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
import json, time, math, argparse, gc, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # package: chain root (cmb/)
import _cmb_paths as WP  # noqa: E402
RUN01 = WP.wdir('ai/cmb_ensemble/run01')
PY = sys.executable  # package
PI = math.pi
S2 = math.sqrt(2.0)
GEN = (1.3, 0.37)          # generic E-plane point for the 8-block exactness test (not a symmetry point)
DELTA_T2 = 0.0065          # sigma_T2 scale
EPS = 0.02                 # small amplitude for the E vs T2 Fisher ratio at the cube point
SIG = 0.1041
TH = {'obl': -PI / 6, 'tri': 0.0, 'pro': PI / 6}


def avail_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 30
    except Exception:
        return None


def task_wait_then(sp):
    sys.path.insert(0, str(HERE))
    import ens_core as ec
    t0 = time.time()
    while not all(Path(p).exists() for p in sp['wait_for']):
        if time.time() - t0 > sp.get('wait_timeout', 2400):
            return dict(ok=False, task=sp.get('task'), tid=sp.get('tid'), error='timeout waiting for inputs',
                        missing=[p for p in sp['wait_for'] if not Path(p).exists()])
        time.sleep(10)
    return ec.run_task(sp)


def make_cfg(smoke):
    if smoke:
        return dict(smoke=True, lmax=12, xmax=60, taper=0.25, sels=[('TE', 8), ('TE', 12), ('T', 12)],
                    prim='TE12', thresh=1.0, ver=dict(lmax=10, xmax=30, taper=0.25, s=1.13),
                    G=[1.0, 1.03, 1.06, 1.09, 1.12, 1.15, 1.18],
                    pts=[(0.5, 'pro', 1.03, 'a'), (1.0, 'tri', 1.06, 'a'), (1.0, 'tri', 1.12, 'b'),
                         (S2, 'obl', 1.12, 'a'), (S2, 'obl', 1.15, 'b'), (S2, 'tri', 1.12, 'a'),
                         (S2, 'tri', 1.15, 'b'), (2.0, 'pro', 1.09, 'a'), (2.0, 'pro', 1.15, 'c'),
                         (1.0, 'pro', 1.15, 'c'), (1.0, 'obl', 1.15, 'c'), (0.5, 'tri', 1.15, 'c'),
                         (1.0, 'obl', 1.09, 'a'), (0.5, 'obl', 1.06, 'a'), (0.5, 'tri', 1.06, 'a'),
                         (1.0, 'pro', 1.06, 'a'), (S2, 'pro', 1.06, 'a'), (2.0, 'obl', 1.18, 'a'),
                         (2.0, 'tri', 1.18, 'a')],
                    slices=[1.15], conv_xmax=80, brent_xtol=2e-3, brent_maxiter=4, n_mc=200000)
    pts = [
        # (a) shortest side u ~ 0.99 (just above the matched-circle bound 0.985)
        (0.5, 'obl', 1.03, 'a'), (0.5, 'tri', 1.03, 'a'), (0.5, 'pro', 1.015, 'a'),
        (1.0, 'obl', 1.075, 'a'), (1.0, 'tri', 1.075, 'a'), (1.0, 'pro', 1.03, 'a'),
        (S2, 'obl', 1.12, 'a'), (S2, 'tri', 1.105, 'a'), (S2, 'pro', 1.06, 'a'),
        (2.0, 'obl', 1.18, 'a'), (2.0, 'tri', 1.15, 'a'), (2.0, 'pro', 1.075, 'a'),
        # representative shape: frozen snapshot fossils (read only)
        (S2, 'tri', 1.13, 'snap'), (S2, 'tri', 1.135, 'snap'), (S2, 'tri', 1.14, 'snap'), (S2, 'tri', 1.16, 'snap'),
        # (b) u ~ 1.01-1.04 (edge bracket)
        (1.0, 'obl', 1.12, 'b'), (1.0, 'tri', 1.105, 'b'), (1.0, 'pro', 1.075, 'b'),
        (S2, 'obl', 1.15, 'b'), (S2, 'pro', 1.09, 'b'),
        (2.0, 'obl', 1.20, 'b'), (2.0, 'tri', 1.18, 'b'), (2.0, 'tri', 1.20, 'b'), (2.0, 'pro', 1.12, 'b'),
        # (c) fixed-L0 slices for the structure fit
        (0.5, 'obl', 1.14, 'c'), (0.5, 'tri', 1.14, 'c'), (0.5, 'pro', 1.14, 'c'),
        (1.0, 'obl', 1.14, 'c'), (1.0, 'tri', 1.14, 'c'), (1.0, 'pro', 1.14, 'c'),
        (S2, 'obl', 1.14, 'c'), (S2, 'pro', 1.14, 'c'), (2.0, 'pro', 1.14, 'c'),
        (1.0, 'tri', 1.15, 'c'), (S2, 'tri', 1.15, 'c'), (S2, 'pro', 1.15, 'c'), (2.0, 'pro', 1.15, 'c'),
    ]
    return dict(smoke=False, lmax=80, xmax=240, taper=0.25, sels=[('TE', 30), ('TE', 40), ('TE', 80), ('T', 80)],
                prim='TE80', thresh=4.5, ver=dict(lmax=40, xmax=160, taper=0.25, s=1.13),
                G=[1.00, 1.015, 1.03, 1.045, 1.06, 1.075, 1.09, 1.105, 1.12, 1.13, 1.14, 1.15, 1.16, 1.18, 1.20],
                pts=pts, slices=[1.13, 1.14, 1.15], conv_xmax=320, brent_xtol=1e-3, brent_maxiter=6,
                n_mc=2_000_000)


# ----------------------------------------------------------------------------- status / runner
class Status:
    def __init__(self, ec, out, cmd):
        self.ec = ec; self.path = out / 'STATUS.json'
        self.d = ec.load_json(self.path, {}) or {}
        self.d.update(agent='cmb-ensemble', task='C145/D117', pid=os.getpid(), next_command=cmd,
                      output_root=str(out), snapshot_input_read_only=str(ec.SNAPSHOT_INPUT))
        for k in ('completed', 'key_numbers', 'outputs'):
            self.d.setdefault(k, [] if k == 'completed' else {})
        self.d['errors'] = []
        self.last = 0.0; self.builds = []; self.W = 1

    def write(self):
        self.d['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        self.ec.save_json(self.path, self.d)

    def set(self, **kw):
        self.d.update(kw); self.write()

    def key(self, k, v):
        self.d['key_numbers'][k] = v; self.write()

    def out(self, k, v):
        self.d['outputs'][k] = str(v); self.write()

    def done(self, item):
        if item not in self.d['completed']:
            self.d['completed'].append(item)
        self.write()

    def task_done(self, label, r, ndone, n, npend, ninf):
        if r.get('build_seconds') and r.get('lmax') == 80 and r.get('xmax') == 240:
            self.builds.append(r['build_seconds'])
        if not r.get('ok'):
            self.d['errors'].append(dict(tid=r.get('tid'), error=r.get('error')))
        self.tick(label, ndone, n, npend, ninf, force=True)

    def tick(self, label, ndone, n, npend, ninf, force=False):
        if not force and time.time() - self.last < 30:
            return
        self.last = time.time()
        b = sorted(self.builds)
        med = b[len(b) // 2] if b else None
        pr = dict(batch=label, done=ndone, total=n, pending=npend, inflight=ninf,
                  median_l80_x240_build_s_single_thread=med, avail_GB=avail_gb())
        if med:
            pr['eta_batch_min_upper'] = round((npend + ninf) * med / max(1, self.W) / 60.0, 1)
        self.d['progress'] = pr
        self.write()


class Runner:
    def __init__(self, ec, W, st, log_path):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        self.ec = ec; self.W = W; self.st = st; self.log_path = log_path; st.W = W
        self.ex = ProcessPoolExecutor(max_workers=W, mp_context=mp.get_context('spawn'), initializer=ec.init_worker)

    def run(self, specs, label, on_result=None):
        from concurrent.futures import wait, FIRST_COMPLETED
        pending = list(specs); inflight = {}; results = []; n = len(specs)
        while pending or inflight:
            while pending and len(inflight) < self.W:
                a = avail_gb()
                if a is not None and a < 1.5 and inflight:
                    break
                sp = pending.pop(0)
                fn = task_wait_then if sp.get('wait_for') else self.ec.run_task
                inflight[self.ex.submit(fn, sp)] = sp
            done, _ = wait(list(inflight), timeout=60, return_when=FIRST_COMPLETED)
            for fu in done:
                sp = inflight.pop(fu)
                try:
                    r = fu.result()
                except Exception as e:
                    r = dict(ok=False, task=sp.get('task'), tid=sp.get('tid'), error=repr(e), tb=traceback.format_exc())
                results.append(r)
                with open(self.log_path, 'a', encoding='utf-8') as fh:
                    fh.write(json.dumps({k: v for k, v in r.items() if k != 'evals'}, default=str) + '\n')
                print(time.strftime('%H:%M:%S'), label, r.get('tid'), 'ok' if r.get('ok') else 'FAIL',
                      round(r.get('seconds') or 0, 1), r.get('error', ''), flush=True)
                if on_result:
                    on_result(r)
                self.st.task_done(label, r, len(results), n, len(pending), len(inflight))
            self.st.tick(label, len(results), n, len(pending), len(inflight))
        return results

    def close(self):
        self.ex.shutdown(wait=True, cancel_futures=True)


# ----------------------------------------------------------------------------- specs
def oct_path(ec, kind, s, lmax, xmax, taper, A=0.0, th=0.0, fresh=False):
    if kind == 'cube':
        return ec.find_cov('cube', s, lmax, xmax, taper)[0]
    return ec.CACHE / (('fresh_' if fresh else '') + ec.cov_name('shape', s, lmax, xmax, taper, A, th) + '.npz')


def par_path(ec, label, s, lmax, xmax, taper):
    return ec.CACHE / ('par_%s_s%.5f_l%d_x%g_t%g.npz' % (label, s, lmax, xmax, taper))


def s1_specs(ec, cfg):
    v = cfg['ver']; base = dict(lmax=v['lmax'], xmax=v['xmax'], taper=v['taper'], s=v['s'])
    L = [dict(task='fresh', tid='S1_fresh', lmax=cfg['lmax'], xmax=cfg['xmax'], taper=cfg['taper'], s=1.13,
              ref=11.412636)]
    if not cfg['smoke']:
        L.append(dict(task='regress', tid='S1_regress'))
    octs = [('oct_cube', dict(kind='cube')), ('oct_rep', dict(kind='shape', A=S2, theta=0.0, fresh=True)),
            ('oct_gen', dict(kind='shape', A=GEN[0], theta=GEN[1])), ('oct_d3', dict(kind='shape', A=S2, theta=PI / 3))]
    paths = {}
    for lab, kw in octs:
        L.append(dict(task='oct', tid='S1_' + lab, label=lab, **kw, **base))
        paths[lab] = str(oct_path(ec, kw['kind'], v['s'], v['lmax'], v['xmax'], v['taper'], kw.get('A', 0.0),
                                  kw.get('theta', 0.0), kw.get('fresh', False)))
    rep5 = [S2 * SIG, 0.0, 0.0, 0.0, 0.0]
    par = {'cube': [0.0] * 5, 'gen': [GEN[0] * SIG * math.cos(GEN[1]), GEN[0] * SIG * math.sin(GEN[1]), 0.0, 0.0, 0.0],
           'rep_d1': [S2 * SIG, 0.0, DELTA_T2, 0.0, 0.0], 'rep_d2': [S2 * SIG, 0.0, 2 * DELTA_T2, 0.0, 0.0],
           'eps_Xd': [EPS, 0.0, 0.0, 0.0, 0.0], 'eps_X12': [0.0, 0.0, EPS, 0.0, 0.0]}
    for lab, b5 in par.items():
        L.append(dict(task='par', tid='S1_par_' + lab, label=lab, beta5=b5, **base))
        paths['par_' + lab] = str(par_path(ec, lab, v['s'], v['lmax'], v['xmax'], v['taper']))
    L.append(dict(task='l40an', tid='S1_l40an', ver=v, paths=paths, gen=list(GEN), wait_for=list(paths.values())))
    return L


def cube_specs(ec, cfg):
    sels = [list(x) for x in cfg['sels']]
    snap = [s for s in cfg['G'] if ec.find_cov('cube', s, cfg['lmax'], cfg['xmax'], cfg['taper'])[1]]
    new = [s for s in cfg['G'] if s not in snap]
    return [dict(task='cov', tid='C_%.5f' % s, kind='cube', s=s, lmax=cfg['lmax'], xmax=cfg['xmax'],
                 taper=cfg['taper'], prep='inv', sels=sels) for s in snap + new]


def fossil_specs(ec, cfg):
    sels = [list(x) for x in cfg['sels']]; out = []; meta = {}
    for A, lab, s, role in cfg['pts']:
        th = TH[lab]
        if s < ec.s_low(A, th) - 1e-9:
            print('skip circle-excluded point', A, lab, s); continue
        tid = 'F_A%.4f_%s_s%.4f' % (A, lab, s)
        if tid in meta:
            meta[tid]['roles'].append(role); continue
        meta[tid] = dict(A=A, lab=lab, theta=th, s=s, roles=[role])
        out.append(dict(task='cov', tid=tid, kind='shape', A=A, theta=th, s=s, lmax=cfg['lmax'], xmax=cfg['xmax'],
                        taper=cfg['taper'], prep='ld', sels=sels))
    return out, meta


# ----------------------------------------------------------------------------- KL tables and profiles
def cross_kl(ec, cfg, cubes, fossils, table_path, st):
    keys = [ec.sel_key(s) for s in cfg['sels']]
    table = ec.load_json(table_path, {}) or {}
    cs = sorted(cubes)
    todo = [fn for fn in fossils if fn not in table or any(('%.5f' % s) not in table[fn] for s in cs)]
    if todo:
        inv = {s: ec.load_inv(cubes[s], keys) for s in cs}
        for i, fn in enumerate(todo):
            fr = fossils[fn]
            cc, _ = ec.load_cov(fr['path'])
            fp = ec.fossil_packs(cc, cfg['lmax'], cfg['sels']); del cc
            ld = ec.load_json(fr['ld'])
            row = table.get(fn, {})
            for s in cs:
                k = '%.5f' % s
                if k not in row:
                    row[k] = {key: list(ec.kl_fast(fp[key], ld[key]['logdet'], inv[s][key])) for key in keys}
            table[fn] = row
            del fp; gc.collect()
            ec.save_json(table_path, table)
            st.set(progress=dict(batch='cross_kl', done=i + 1, total=len(todo), avail_GB=avail_gb()))
        del inv; gc.collect()
    return table


def parab(xs, ys):
    i = min(range(len(ys)), key=lambda j: ys[j])
    res = dict(grid_s=xs[i], grid_D=ys[i], idx=i)
    if i == 0 or i == len(xs) - 1:
        res.update(s=xs[i], D=ys[i], kind='boundary'); return res
    x0, x1, x2 = xs[i - 1:i + 2]; y0, y1, y2 = ys[i - 1:i + 2]
    d = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / d
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / d
    c = (x1 * x2 * (x1 - x2) * y0 + x2 * x0 * (x2 - x0) * y1 + x0 * x1 * (x0 - x1) * y2) / d
    if a <= 0:
        res.update(s=x1, D=y1, kind='grid'); return res
    xv = -b / (2 * a); yv = c - b * b / (4 * a)
    if not (x0 <= xv <= x2):
        res.update(s=x1, D=y1, kind='grid'); return res
    res.update(s=xv, D=min(yv, y1), kind='parabolic', bracket=[x0, x1, x2])
    return res


def nsel(ec, cfg, key):
    f = key[:2] if key.startswith('TE') else key[:1]; l = int(key[len(f):])
    return sum(len(ec.sel_index(b['labels'], l, f)) for b in ec.layouts_c(cfg['lmax']))


def profiles(ec, cfg, table, fossils, fmeta):
    keys = [ec.sel_key(s) for s in cfg['sels']]
    nk = {k: nsel(ec, cfg, k) for k in keys}
    out = {}
    for fn, fr in fossils.items():
        m = fmeta[fr['tid']]; row = table[fn]
        ss = sorted(float(k) for k in row)
        f = ec.short_factor(m['A'], m['theta'])
        rec = dict(name=fn, tid=fr['tid'], A=m['A'], lab=m['lab'], theta=m['theta'], xi=math.sin(3 * m['theta']),
                   s=m['s'], roles=m['roles'], f=f, u=m['s'] * f, s_low=ec.CIRCLE / f, path=fr['path'])
        for key in keys:
            ys = [row['%.5f' % s][key][0] for s in ss]; aa = [row['%.5f' % s][key][1] for s in ss]
            pr = parab(ss, ys)
            yam = [y - 0.5 * nk[key] * (a - 1 - math.log(a)) for y, a in zip(ys, aa)]
            pa = parab(ss, yam)
            fx = row.get('%.5f' % m['s'])
            rec[key] = dict(D_fixed=fx[key][0] if fx else None, a_star_fixed=fx[key][1] if fx else None,
                            D_prof=pr['D'], s_prof=pr['s'], prof_kind=pr['kind'], D_grid=pr['grid_D'],
                            s_grid=pr['grid_s'], a_star_at_grid_min=aa[pr['idx']],
                            Damp_prof=pa['D'], curve=[[x, y] for x, y in zip(ss, ys)])
        out[fn] = rec
    return out


# ----------------------------------------------------------------------------- S3 analysis
def edge(pts, T):
    pts = sorted([p for p in pts if p[1] is not None and p[1] > 0])
    if len(pts) < 2:
        return None
    above = [p[1] >= T for p in pts]
    if all(above):
        (sa, Da), (sb, Db) = pts[-2], pts[-1]; flag = 'extrap_up'
    elif not any(above):
        (sa, Da), (sb, Db) = pts[0], pts[1]; flag = 'extrap_down'
    else:
        idx = next((i for i in range(1, len(pts)) if above[i - 1] and not above[i]), None)
        if idx is None:
            (sa, Da), (sb, Db) = pts[-2], pts[-1]; flag = 'nonmonotone'
        else:
            (sa, Da), (sb, Db) = pts[idx - 1], pts[idx]; flag = 'bracket'
    slope = (math.log(Db) - math.log(Da)) / (sb - sa)
    if slope >= 0:
        return dict(s_max=None, flag=flag + '_nonneg_slope', pts=pts)
    smax = sa + (math.log(T) - math.log(Da)) / slope
    smax_lin = sa + (T - Da) * (sb - sa) / (Db - Da) if Db != Da else smax
    mono = all(pts[i + 1][1] <= pts[i][1] for i in range(len(pts) - 1))
    return dict(s_max=smax, s_max_lin=smax_lin, flag=flag, bracket=[sa, sb], D_bracket=[Da, Db],
                dlnD_ds=slope, monotone_decreasing=mono, pts=pts)


def lsq_structure(rows):
    import numpy as np
    A = np.array([r['A'] for r in rows]); xi = np.array([r['xi'] for r in rows]); D = np.array([r['D'] for r in rows])
    X = np.column_stack([A ** 2, A ** 3 * xi, A ** 4])
    c, *_ = np.linalg.lstsq(X, D, rcond=None)
    pred = X @ c
    loo = []
    for i in range(len(D)):
        m = np.ones(len(D), bool); m[i] = False
        if m.sum() >= 3:
            ci, *_ = np.linalg.lstsq(X[m], D[m], rcond=None); loo.append(float(X[i] @ ci))
        else:
            loo.append(None)
    return dict(c2=float(c[0]), c3=float(c[1]), c4=float(c[2]),
                points=[dict(A=float(a), xi=float(x), D=float(d), fit=float(p), rel_resid=float(p / d - 1),
                             loo_pred=l, loo_rel=(l / d - 1) if l is not None else None)
                        for a, x, d, p, l in zip(A, xi, D, pred, loo)],
                rms_rel_resid=float(np.sqrt(np.mean((pred / D - 1) ** 2))))


def analysis(ec, cfg, prof, brent, noise, conv, st, RES):
    import numpy as np
    T = cfg['thresh']; prim = cfg['prim']; keys = [ec.sel_key(s) for s in cfg['sels']]
    out = dict(threshold=T, primary=prim, circle=ec.CIRCLE, sigma_E=ec.SIGMA_E)
    # noise profiles
    if noise:
        for fn, rows in noise.items():
            if fn not in prof:
                continue
            nk = [k for k in rows[0] if k != 's_prime']
            prof[fn]['noise'] = {}
            for k in nk:
                xs = [r['s_prime'] for r in rows]; ys = [r[k] for r in rows]
                o = sorted(range(len(xs)), key=lambda j: xs[j]); xs = [xs[j] for j in o]; ys = [ys[j] for j in o]
                pr = parab(xs, ys)
                fx = [r[k] for r in rows if abs(r['s_prime'] - prof[fn]['s']) < 1e-9]
                prof[fn]['noise'][k] = dict(D_prof=pr['D'], s_prof=pr['s'], prof_kind=pr['kind'],
                                            D_fixed=fx[0] if fx else None)
    getters = {}
    for k in keys:
        getters[k + '_prof'] = (lambda r, k=k: r[k]['D_prof'])
        getters[k + '_fixed'] = (lambda r, k=k: r[k]['D_fixed'])
    getters[prim + 'ampfree_prof'] = (lambda r: r[prim]['Damp_prof'])
    nkeys = sorted({k for r in prof.values() for k in r.get('noise', {})})
    for k in nkeys:
        getters['noise_' + k + '_prof'] = (lambda r, k=k: r.get('noise', {}).get(k, {}).get('D_prof'))
        getters['noise_' + k + '_fixed'] = (lambda r, k=k: r.get('noise', {}).get(k, {}).get('D_fixed'))
    shapes = {}
    for r in prof.values():
        shapes.setdefault((round(r['A'], 6), r['lab']), []).append(r)
    # structure slices
    struct = {}
    for L0 in cfg['slices']:
        rows = [r for r in prof.values() if abs(r['s'] - L0) < 1e-9]
        sl = dict(points=[dict(A=r['A'], lab=r['lab'], theta=r['theta'], xi=r['xi'], u=r['u'],
                               D_fixed=r[prim]['D_fixed'], D_prof=r[prim]['D_prof'], s_prof=r[prim]['s_prof'],
                               D_fixed_TE30=r.get('TE30', {}).get('D_fixed'), D_prof_TE30=r.get('TE30', {}).get('D_prof'))
                          for r in sorted(rows, key=lambda r: (r['A'], r['theta']))])
        for kind in ('D_fixed', 'D_prof'):
            pts = [dict(A=p['A'], xi=p['xi'], D=p[kind]) for p in sl['points'] if p[kind]]
            if len(pts) >= 4:
                sl['fit_' + kind] = lsq_structure(pts)
        struct['%.3f' % L0] = sl
    out['structure'] = struct
    # global u surrogate
    rr = [r for r in prof.values() if r[prim]['D_prof'] and 0.2 < r[prim]['D_prof'] < 80]
    if len(rr) >= 6:
        X = np.column_stack([np.ones(len(rr)), [math.log(r['A']) for r in rr], [r['u'] - 1 for r in rr], [r['xi'] for r in rr]])
        y = np.log([r[prim]['D_prof'] for r in rr])
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        out['u_surrogate'] = dict(model='ln D_prof = c0 + c1 ln A + c2 (u-1) + c3 xi', coef=c.tolist(),
                                  rms_ln_resid=float(np.sqrt(np.mean((X @ c - y) ** 2))), n=len(rr))
    # edges per variant
    edges = {}
    for vn, g in getters.items():
        ev = {}
        for (A, lab), rs in shapes.items():
            pts = [(r['s'], g(r)) for r in rs]
            e = edge(pts, T)
            if e is None or e.get('s_max') is None:
                ev['%s|%.5f' % (lab, A)] = dict(A=A, lab=lab, edge=e); continue
            f = ec.short_factor(A, TH[lab]); slo = ec.CIRCLE / f
            e.update(A=A, lab=lab, s_low=slo, u_max=e['s_max'] * f, span=math.log(e['s_max'] / slo),
                     span_lin=math.log(max(e['s_max_lin'], 1e-6) / slo))
            ev['%s|%.5f' % (lab, A)] = e
        edges[vn] = ev
    out['edges'] = edges
    # Brent profile check
    if brent:
        out['brent'] = []
        for b in brent:
            if not b.get('ok'):
                continue
            fn = b['fossil']; r = prof.get(fn)
            if r:
                out['brent'].append(dict(fossil=fn, A=r['A'], lab=r['lab'], s=r['s'], s_star=b['s_star'],
                                         D_brent=b['D_star'], D_parab=r[prim]['D_prof'], s_parab=r[prim]['s_prof'],
                                         D_grid=r[prim]['D_grid'], rel_parab_minus_brent=r[prim]['D_prof'] / b['D_star'] - 1,
                                         n_eval=b['n_unique_evaluations'], n_new=b['n_new_builds'], interior=b['interior']))
    # Monte Carlo ensemble (same seed/sampling as snapshot size_prior.py)
    rng = np.random.default_rng(20261013); n = cfg['n_mc']
    be = rng.normal(size=(n, 2)) * ec.SIGMA_E
    mag = np.hypot(be[:, 0], be[:, 1]); thb = np.arctan2(be[:, 1], be[:, 0])
    f = np.exp(np.min(math.sqrt(2 / 3) * mag[:, None] * np.cos(thb[:, None] - np.deg2rad([30, 150, 270])), axis=1))
    Am = mag / ec.SIGMA_E; xi = np.sin(3 * thb); slo = ec.CIRCLE / f
    L0 = np.round(np.arange(0.98, 1.4001, 0.0025), 5)
    P_nc = np.searchsorted(np.sort(slo), L0, side='right') / n
    out['L0_grid'] = L0.tolist(); out['P_no_circles'] = P_nc.tolist()
    curves = {}; spantab = {}
    dsel = None
    for vn, ev in edges.items():
        var = dict(A={}, span={})
        for lab in ('obl', 'tri', 'pro'):
            es = sorted([e for e in ev.values() if e.get('lab') == lab and e.get('span') is not None], key=lambda e: e['A'])
            var['A'][lab] = [e['A'] for e in es]; var['span'][lab] = [e['span'] for e in es]
        if any(len(var['A'][lab]) < 2 for lab in var['A']):
            continue
        spantab[vn] = var

        def pc(var, shift=0.0):
            sp = ec.span_eval(var, Am, xi) + shift
            ok = sp > 0
            shi = slo * np.exp(sp)
            Pj = (np.searchsorted(np.sort(slo[ok]), L0, side='right') - np.searchsorted(np.sort(shi[ok]), L0, side='left')) / n
            return Pj, float(np.mean(np.maximum(sp, 0)))
        Pj, mspan = pc(var)
        Pc = np.where(P_nc > 0, Pj / np.maximum(P_nc, 1e-300), 0.0)
        rep = ev.get('tri|%.5f' % round(S2, 6)) or ev.get('tri|%.5f' % S2)
        repw = [rep['s_low'], rep['s_max']] if rep and rep.get('s_max') else None

        def wins(P):
            o = {}
            for p in (0.5, 0.3, 0.2, 0.1, 0.05):
                idx = np.flatnonzero(P >= p)
                o[str(p)] = None if not len(idx) else dict(L0_min=float(L0[idx[0]]), L0_max=float(L0[idx[-1]]),
                                                            contiguous=bool(np.all(np.diff(idx) == 1)),
                                                            dlnL0=float(np.log(L0[idx[-1]] / L0[idx[0]])))
            return o
        cur = dict(P_joint=Pj.tolist(), P_cond=Pc.tolist(), mean_readable_lnL0_span=mspan,
                   rep_window=repw, rep_span=(math.log(repw[1] / repw[0]) if repw else None),
                   P_joint_max=float(Pj.max()), L0_at_P_joint_max=float(L0[int(Pj.argmax())]),
                   P_cond_max=float(Pc.max()), L0_at_P_cond_max=float(L0[int(Pc.argmax())]),
                   windows_joint=wins(Pj), windows_cond=wins(Pc), span_table=var)
        if vn == prim + '_prof':
            d_int = max([abs(e['span'] - e['span_lin']) for e in ev.values() if e.get('flag') == 'bracket'] or [0.0])
            d_pr = 0.0
            for b in out.get('brent', []):
                es = [e for e in ev.values() if e.get('lab') == b['lab'] and abs(e['A'] - b['A']) < 1e-6 and e.get('dlnD_ds')]
                if es:
                    d_pr = max(d_pr, abs(math.log(b['D_parab'] / b['D_brent'])) / abs(es[0]['dlnD_ds']) / b['s'])
            dsel = math.hypot(d_int, d_pr)
            lo, _ = pc(var, -dsel); hi, _ = pc(var, +dsel)
            cur.update(band_dspan=dict(interp=d_int, profile=d_pr, total=dsel), P_joint_lo=lo.tolist(), P_joint_hi=hi.tolist(),
                       P_cond_lo=np.where(P_nc > 0, lo / np.maximum(P_nc, 1e-300), 0).tolist(),
                       P_cond_hi=np.where(P_nc > 0, hi / np.maximum(P_nc, 1e-300), 0).tolist(),
                       windows_joint_lo=wins(lo), windows_joint_hi=wins(hi))
            # theta-interpolation check is not available (only canonical theta computed): flagged
        curves[vn] = cur
    out['curves'] = curves
    out['conv'] = conv
    ec.save_json(RES / 'S3_analysis.json', out)
    # compact csv of all fossils
    lines = ['name,A,lab,theta,xi,s,u,s_low,' + ','.join('%s_fixed,%s_prof,%s_sprof' % (k, k, k) for k in keys)]
    for r in sorted(prof.values(), key=lambda r: (r['lab'], r['A'], r['s'])):
        vals = []
        for k in keys:
            vals += ['%.6g' % r[k]['D_fixed'] if r[k]['D_fixed'] is not None else '', '%.6g' % r[k]['D_prof'], '%.5f' % r[k]['s_prof']]
        lines.append('%s,%.5f,%s,%.5f,%.4f,%.5f,%.5f,%.5f,' % (r['name'], r['A'], r['lab'], r['theta'], r['xi'], r['s'], r['u'], r['s_low']) + ','.join(vals))
    (ec.out_path(RES / 'fossil_table.csv')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    # figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for vn, sty in ((prim + '_prof', '-'), (prim + '_fixed', '--'), ('TE30_prof', ':'), ('T80_prof', '-.'),
                        ('noise_planck143_' + prim + '_prof', (0, (5, 1, 1, 1)))):
            if vn in curves:
                ax[0].plot(L0, curves[vn]['P_joint'], sty, label=vn)
                ax[1].plot(L0, curves[vn]['P_cond'], sty, label=vn)
        if prim + '_prof' in curves and 'P_joint_lo' in curves[prim + '_prof']:
            c = curves[prim + '_prof']
            ax[0].fill_between(L0, c['P_joint_lo'], c['P_joint_hi'], alpha=.25)
            ax[1].fill_between(L0, c['P_cond_lo'], c['P_cond_hi'], alpha=.25)
            if c['rep_window']:
                for a in ax:
                    a.axvspan(*c['rep_window'], color='grey', alpha=.15, label='rep. point window')
        ax[0].set_title('P(read and no circles | L0)'); ax[1].set_title('P(read | no circles, L0)')
        for a in ax:
            a.set_xlabel('L0 / d_LSS'); a.set_xlim(0.98, 1.35); a.grid(alpha=.3)
        ax[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(ec.out_path(RES / 'P_read.png'), dpi=130); plt.close(fig)
    except Exception as e:
        out['figure_error'] = repr(e)
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--workers', type=int, default=int(os.environ.get("DNEC_WORKERS", "6")))  # package: --workers
    ap.add_argument('--analysis-only', action='store_true')
    ap.add_argument('--no-stage4', action='store_true')
    ap.add_argument('--refine', action='store_true')
    args = ap.parse_args()
    if args.smoke:
        os.environ['CMB_ENS_OUT'] = str(RUN01 / 'smoke')
    sys.path.insert(0, str(HERE))
    import numpy as np
    import ens_core as ec
    cfg = make_cfg(args.smoke)
    if args.refine and not args.smoke:
        # Refinement round chosen from the first actual ensemble (batch-1 S2_profiles): bracket the small-A
        # edges (A=0.5 at u~1.02; A=0.25 near the circle bound), narrow the wide A=1 brackets, replace the
        # (2,obl)/(sqrt2,obl) upward extrapolations; extra actual cubes 0.985/0.9925 (no-circle bound) and 1.19/1.22.
        cfg['pts'] = cfg['pts'] + [
            (0.5, 'obl', 1.06, 'r'), (0.5, 'tri', 1.06, 'r'), (0.5, 'pro', 1.045, 'r'),
            (0.25, 'obl', 1.015, 'r'), (0.25, 'obl', 1.03, 'r'), (0.25, 'tri', 1.015, 'r'), (0.25, 'tri', 1.03, 'r'),
            (0.25, 'pro', 1.00, 'r'), (0.25, 'pro', 1.015, 'r'), (0.25, 'pro', 1.03, 'r'),
            (1.0, 'tri', 1.09, 'r'), (1.0, 'obl', 1.105, 'r'), (1.0, 'pro', 1.06, 'r'),
            (S2, 'obl', 1.16, 'r'), (S2, 'pro', 1.105, 'r'), (2.0, 'obl', 1.22, 'r'), (2.0, 'tri', 1.19, 'r')]
        cfg['G'] = sorted(set(cfg['G'] + [0.985, 0.9925, 1.19, 1.22]))
    OUT = ec.NEW_RUN_OUTPUT; RES = OUT / 'results'; LOGD = OUT / 'logs'
    ec.out_path(RES / 'x'); ec.out_path(LOGD / 'x')
    cmd = "& '%s' -B -u '%s'%s%s" % (PY, HERE / 'run_ens.py', ' --smoke' if args.smoke else '', ' --refine' if args.refine else '')
    st = Status(ec, OUT, cmd)
    st.set(state='running', stage='S1+S2 builds', workers=args.workers, avail_GB_at_start=avail_gb())
    runner = None
    try:
        fspecs, fmeta = fossil_specs(ec, cfg)
        if not args.analysis_only:
            runner = Runner(ec, args.workers, st, LOGD / 'tasks.jsonl')

            def on1(r):
                if r.get('tid') == 'S1_fresh' and r.get('ok'):
                    ec.save_json(RES / 'S1_fresh_build.json', r)
                    st.key('S1_fresh', {k: r.get(k) for k in ('build_seconds', 'max_whitened_diff_vs_snapshot', 'D_fresh',
                                                               'D_snapshot_cache', 'abs_diff_fresh_vs_cache', 'D_P5_table')})
                    st.done('S1 fresh ell80 build + benchmark')
                if r.get('tid') == 'S1_regress' and r.get('ok'):
                    ec.save_json(RES / 'S1_regression.json', r)
                    st.key('S1_regression_D80', {k: r['regression']['TE80'][k] for k in ('D_eig', 'ref', 'abs_err', 'fast_minus_eig', 'chol_minus_eig')})
                    st.done('S1 regression from snapshot cache (no recomputation)')
                if r.get('tid') == 'S1_l40an' and r.get('ok'):
                    ec.save_json(RES / 'S1_verify_l40.json', r)
                    st.key('S1_8block_generic', r.get('generic')); st.key('S1_D3', r.get('D3'))
                    st.key('S1_T2', {k: r['T2'][k] for k in ('dD_rep_d1', 'dD_rep_d2', 'quadratic_ratio_d2_over_d1', 'fisher_ratio_T2_over_E_at_cube')})
                    st.done('S1 8-block exactness at theta!=0, D3 and T2 checks (ell40)')
            res1 = runner.run(([] if args.refine else s1_specs(ec, cfg)) + cube_specs(ec, cfg) + fspecs, 'batch1', on1)
            ec.save_json(RES / 'batch1_results.json', [{k: v for k, v in r.items() if k not in ('evals',)} for r in res1])
        else:
            res1 = ec.load_json(RES / 'batch1_results.json', [])
        cubes = {r['s']: r['inv'] for r in res1 if r.get('ok') and r.get('task') == 'cov' and r.get('kind') == 'cube'}
        fossils = {r['name']: r for r in res1 if r.get('ok') and r.get('task') == 'cov' and r.get('kind') == 'shape'}
        st.done('S2 builds: %d cubes, %d fossils' % (len(cubes), len(fossils)))
        st.set(stage='S2 cross-KL')
        table = cross_kl(ec, cfg, cubes, fossils, RES / 'kl_table.json', st)
        prof = profiles(ec, cfg, table, fossils, fmeta)
        ec.save_json(RES / 'S2_profiles.json', prof)
        st.done('S2 fixed/profiled D on the actual cube pool'); st.out('profiles', RES / 'S2_profiles.json')
        prim = cfg['prim']; T = cfg['thresh']
        # Brent targets: representative fossil closest to threshold + two other shapes (distinct theta)
        cand = sorted(prof.values(), key=lambda r: abs(math.log(r[prim]['D_prof'] / T)))
        targets = []
        rep = [r for r in cand if abs(r['A'] - S2) < 1e-6 and r['lab'] == 'tri']
        if rep:
            targets.append(rep[0])
        for lab in ('obl', 'pro', 'tri'):
            c2 = [r for r in cand if r['lab'] == lab and r not in targets and r[prim]['prof_kind'] != 'boundary']
            if c2 and len(targets) < 3:
                targets.append(c2[0])
        bspecs = []
        for r in targets:
            cur = r[prim]['curve']; i = min(range(len(cur)), key=lambda j: cur[j][1])
            if 0 < i < len(cur) - 1:
                bspecs.append(dict(task='brent', tid='B_' + r['tid'], fossil_name=r['name'], fossil_path=r['path'],
                                   pool=cur, bracket=[cur[i - 1][0], cur[i][0], cur[i + 1][0]], lmax=cfg['lmax'],
                                   xmax=cfg['xmax'], taper=cfg['taper'], xtol=cfg['brent_xtol'], maxiter=cfg['brent_maxiter']))
        s4 = []
        if not args.no_stage4 and targets:
            r = targets[0]; sg = r[prim]['s_grid']; A, th, s = r['A'], r['theta'], r['s']
            X = cfg['conv_xmax']
            for kind, ss, AA, tt in (('shape', s, A, th), ('cube', s, 0.0, 0.0), ('cube', sg, 0.0, 0.0)):
                s4.append(dict(task='cov', tid='V_x%d_%s_%.4f' % (X, kind, ss), kind=kind, s=ss, A=AA, theta=tt,
                               lmax=cfg['lmax'], xmax=X, taper=cfg['taper']))
                s4.append(dict(task='taper', tid='V_t125_%s_%.4f' % (kind, ss), kind=kind, s=ss, A=AA, theta=tt,
                               lmax=cfg['lmax'], xmax=cfg['xmax']))
            sets = {}
            for nm, xm, tp in (('x240_t0.25', cfg['xmax'], cfg['taper']), ('x%d_t0.25' % X, X, cfg['taper']), ('x240_t0.125', cfg['xmax'], 0.125)):
                sets[nm] = dict(fossil=str(ec.find_cov('shape', s, cfg['lmax'], xm, tp, A, th)[0]),
                                cube_fixed=str(ec.find_cov('cube', s, cfg['lmax'], xm, tp)[0]),
                                cube_prof=str(ec.find_cov('cube', sg, cfg['lmax'], xm, tp)[0]))
            if args.smoke:
                sets = {k.replace('x240', 'x%d' % cfg['xmax']): v for k, v in sets.items()}
            noise_specs = []
            for fr in sorted(prof.values(), key=lambda r: abs(math.log(r[prim]['D_prof'] / T))):
                cur = fr[prim]['curve']; i = min(range(len(cur)), key=lambda j: cur[j][1])
                ss = sorted({cur[j][0] for j in range(max(0, i - 3), min(len(cur), i + 4))} | ({fr['s']} if any(abs(c[0] - fr['s']) < 1e-9 for c in cur) else set()))
                noise_specs.append(dict(task='noise', tid='N_' + fr['tid'], fossil_name=fr['name'], fossil_path=fr['path'],
                                        lmax=cfg['lmax'], sels=[[prim[:2], cfg['lmax']], ['TE', 30 if not args.smoke else 8]],
                                        levels=[dict(name='planck143', dT=33.0, dP=70.0, fwhm=7.3)],
                                        cubes=[[x, str(ec.find_cov('cube', x, cfg['lmax'], cfg['xmax'], cfg['taper'])[0])] for x in ss]))
            s4 = s4 + [dict(task='conv_an', tid='V_conv_an', lmax=cfg['lmax'], target=r['name'], s_prime=sg, settings=sets,
                            wait_for=[p for d in sets.values() for p in d.values()])] + noise_specs
        reuse = args.analysis_only or args.refine
        bres = ec.load_json(RES / 'S2_brent.json', None) if reuse else None
        conv = ec.load_json(RES / 'S4_convergence.json', None) if reuse else None
        noise = ec.load_json(RES / 'S4_noise.json', None) if reuse else None
        if args.refine and not args.analysis_only:
            noise = noise or {}
            todo_n = [sp for sp in s4 if sp['task'] == 'noise' and sp['fossil_name'] not in noise]
            st.set(stage='R noise for refinement fossils (%d)' % len(todo_n))
            res3 = runner.run(todo_n, 'refine_noise')
            noise.update({r['fossil']: r['rows'] for r in res3 if r.get('task') == 'noise' and r.get('ok')})
            ec.save_json(RES / 'S4_noise.json', noise)
            st.done('R refinement round: %d fossils total, noise for %d new' % (len(fossils), len(todo_n)))
        if not reuse:
            st.set(stage='S2 Brent profile + S4 convergence/noise (idle workers)')
            res2 = runner.run(bspecs + s4, 'batch2')
            bres = [r for r in res2 if r.get('task') == 'brent']
            ec.save_json(RES / 'S2_brent.json', bres)
            cv = [r for r in res2 if r.get('task') == 'conv_an' and r.get('ok')]
            conv = cv[0] if cv else None
            if conv:
                ec.save_json(RES / 'S4_convergence.json', conv)
            noise = {r['fossil']: r['rows'] for r in res2 if r.get('task') == 'noise' and r.get('ok')}
            ec.save_json(RES / 'S4_noise.json', noise)
            st.done('S2 bracketed Brent continuous profile (%d targets)' % len(bres))
            if conv:
                st.done('S4 x%d / taper .125 convergence at the representative edge' % cfg['conv_xmax'])
            st.done('S4 Planck-like noise D for %d fossils' % len(noise))
        st.set(stage='S3 analysis')
        out = analysis(ec, cfg, prof, bres, noise, conv, st, RES)
        c = out['curves'].get(prim + '_prof', {})
        st.key('S3_primary', {k: c.get(k) for k in ('P_joint_max', 'L0_at_P_joint_max', 'P_cond_max', 'L0_at_P_cond_max',
                                                    'mean_readable_lnL0_span', 'rep_window', 'rep_span', 'windows_joint',
                                                    'windows_cond', 'band_dspan')})
        st.key('S3_other', {vn: {k: cv.get(k) for k in ('P_joint_max', 'P_cond_max', 'mean_readable_lnL0_span', 'rep_window')}
                            for vn, cv in out['curves'].items() if vn != prim + '_prof'})
        st.key('S2_brent', out.get('brent'))
        st.key('S3_structure_fit_rms', {L: {k: v.get('rms_rel_resid') for k, v in sl.items() if k.startswith('fit_')} for L, sl in out['structure'].items()})
        st.done('S3 structure fit, edges, P(read|L0) curves and windows')
        st.out('analysis', RES / 'S3_analysis.json'); st.out('figure', RES / 'P_read.png')
        st.set(state='computation_complete', stage='S1-S3 done (S4 conv/noise included if listed); priors + D117 pending',
               next_command="priors: & '%s' -B -u '%s'; then write D117" % (PY, HERE / 'priors_ens.py'))
    except Exception as e:
        st.set(state='failed', error=repr(e), traceback=traceback.format_exc())
        raise
    finally:
        if runner:
            runner.close()


if __name__ == '__main__':
    main()
