# sm-blind R16 V1 -- own core code (independent of the production pipeline):
#  * alpha_s: 5-flavour running (4/5 loops) + 5->6 decoupling with on-shell top mass (up to 4 loops)
#  * mbar_t(mbar_t) from M_t (pure QCD, 4 loops)
#  * SM RGE: 3-loop BPV betas (generated from mr's rge.m, checked term-by-term vs Buttazzo et al. App.)
#            + 4-loop terms (own pure-QCD ones; full known set transcribed programmatically from SMDR betas.c)
#            + 5-loop pure QCD (g3, y_t)
#  * interface to mr_driver.exe (KPV two-loop matching) and smdr_driver.exe (SMDR fit) and coupling assembly
import os, sys, math, re, json, subprocess
import numpy as np
from scipy.integrate import solve_ivp

MY = r'D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\07_reviews\R16_v12_round3\sm_mapping_blind'
WORK = os.path.join(MY, 'work')
if WORK not in sys.path:
    sys.path.insert(0, WORK)
import rge_bpv3_gen as G

BUILD = os.path.join(WORK, 'build')
MR_EXE = os.path.join(BUILD, 'mr_driver.exe')
SMDR_EXE = os.path.join(BUILD, 'smdr_driver.exe')
SMDR_BETAS = os.path.join(MY, 'inputs', '20261014_smdr_extracted', 'SMDR-1.3', 'src', 'betas.c')
MINGW = r'D:\Python\envs\cbuild\Library\mingw-w64\bin'

PI = math.pi
Z2, Z3, Z4, Z5 = PI ** 2 / 6, 1.2020569031595942854, PI ** 4 / 90, 1.0369277551433699263
LI4, LI5, LN2 = 0.5174790616738993863, 0.5084005792422687075, math.log(2.0)
KAP = 1.0 / (16 * PI ** 2)

# ---------------- PDG 2026 inputs (fetched 2026-10-14, inputs/20261014_pdg) ----------------
PDG26 = dict(Mt=172.60, asMZ=0.1180, Mh=125.13, MW_PDG=80.3625, GW=2.14, MZ_PDG=91.1879, GZ=2.4955,
             GF=1.1663785e-5, mbmb=4.186, mcmc=1.2729, Mtau=1.77693)


def bw_to_pole(M, Gam):
    """PDG (running-width Breit-Wigner) mass -> pole mass Mbar with s_P = Mbar^2 - i Mbar Gbar."""
    return M / math.sqrt(1 + (Gam / M) ** 2)


# ---------------- QCD ----------------
def bQCD(nf):
    b0 = 11 - 2 / 3 * nf
    b1 = 102 - 38 / 3 * nf
    b2 = 2857 / 2 - 5033 / 18 * nf + 325 / 54 * nf ** 2
    b3 = (149753 / 6 + 3564 * Z3) - (1078361 / 162 + 6508 / 27 * Z3) * nf + (50065 / 162 + 6472 / 81 * Z3) * nf ** 2 \
        + 1093 / 729 * nf ** 3
    b4 = (8157455 / 16 + 621885 / 2 * Z3 - 88209 / 2 * Z4 - 288090 * Z5) \
        + nf * (-336460813 / 1944 - 4811164 / 81 * Z3 + 33935 / 6 * Z4 + 1358995 / 27 * Z5) \
        + nf ** 2 * (25960913 / 1944 + 698531 / 81 * Z3 - 10526 / 9 * Z4 - 381760 / 81 * Z5) \
        + nf ** 3 * (-630559 / 5832 - 48722 / 243 * Z3 + 1618 / 27 * Z4 + 460 / 9 * Z5) \
        + nf ** 4 * (1205 / 2916 - 152 / 81 * Z3)
    return [b0, b1, b2, b3, b4]


def gamma_m(nf):
    """d ln m / d ln mu^2 = - sum_i gamma_i (alpha_s/pi)^(i+1)."""
    g0 = 1.0
    g1 = (202 / 3 - 20 / 9 * nf) / 16
    g2 = (1249 + (-2216 / 27 - 160 / 3 * Z3) * nf - 140 / 81 * nf ** 2) / 64
    g3 = (4603055 / 162 + 135680 / 27 * Z3 - 8800 * Z5 + (-91723 / 27 - 34192 / 9 * Z3 + 880 * Z4 + 18400 / 9 * Z5) * nf
          + (5242 / 243 + 800 / 9 * Z3 - 160 / 3 * Z4) * nf ** 2 + (-332 / 243 + 64 / 27 * Z3) * nf ** 3) / 256
    g4 = 559.7069 - 143.6864 * nf + 7.4824 * nf ** 2 + 0.1083 * nf ** 3 - 0.000085 * nf ** 4  # BCK 1402.6611
    return [g0, g1, g2, g3, g4]


def run_as(als0, mu0, mu1, nf, loops):
    b = bQCD(nf)[:loops]
    if mu0 == mu1:
        return als0
    f = lambda t, y: [-sum(bi * y[0] ** (i + 2) for i, bi in enumerate(b))]
    s = solve_ivp(f, (2 * math.log(mu0), 2 * math.log(mu1)), [als0 / (4 * PI)], method='DOP853', rtol=1e-13, atol=1e-20)
    return s.y[0, -1] * 4 * PI


def dec_z_5to6(als5, mu, M, nl=5, order=3):
    """alpha_s^(6)(mu)/alpha_s^(5)(mu), on-shell heavy mass M (CKS hep-ph/0512060 eq.63 as coded in mr; the
    a^2 and a^3 constants were checked analytically against CKS 1997: +7/24 (as/pi)^2, +4.0112 (as/pi)^3)."""
    a = als5 / (4 * PI); L = 2 * math.log(mu / M); z = 1.0
    if order >= 1:
        z += a * L * (2 / 3)
    if order >= 2:
        z += a ** 2 * (14 / 3 + 38 / 3 * L + 4 / 9 * L ** 2)
    if order >= 3:
        z += a ** 3 * (58933 / 1944 + 80507 / 432 * Z3 + 128 / 3 * Z2 + 128 / 9 * LN2 * Z2 + nl * (-2479 / 486 - 64 / 9 * Z2)
                       + L * (8941 / 27 - 409 / 27 * nl) + 511 / 9 * L ** 2 + 8 / 27 * L ** 3)
    if order >= 4:
        z += a ** 4 * (592371712 / 382725 - 21814592 / 2835 * LI5 - 25433192 / 1701 * LI4 + 40596749 / 5670 * Z5
                       + 71102219 / 8505 * Z4 + 2408412383 / 340200 * Z3 + 11153936 / 1215 * Z2 - 46048 / 27 * Z2 * Z3
                       - 18636934 / 2835 * LN2 * Z4 - 131456 / 81 * LN2 * Z2 + 5826074 / 1701 * LN2 ** 2 * Z2
                       - 5453648 / 8505 * LN2 ** 3 * Z2 - 3179149 / 5103 * LN2 ** 4 + 2726824 / 42525 * LN2 ** 5)
        z += a ** 4 * nl * (-1773073 / 2916 - 692 / 81 * LI4 - 460 / 9 * Z5 + 697709 / 648 * Z4 - 4756441 / 3888 * Z3
                            - 71296 / 81 * Z2 - 5632 / 81 * LN2 * Z2 + 1709 / 81 * LN2 ** 2 * Z2 - 173 / 486 * LN2 ** 4)
        z += a ** 4 * nl ** 2 * (140825 / 5832 + 76 / 27 * Z3 + 1664 / 81 * Z2)
        z += a ** 4 * L * (21084715 / 2916 + 2922161 / 648 * Z3 + 8960 / 9 * Z2 + 8960 / 27 * LN2 * Z2)
        z += a ** 4 * L * nl * (-1140191 / 1458 - 132283 / 324 * Z3 - 6016 / 27 * Z2 - 512 / 27 * LN2 * Z2)
        z += a ** 4 * L * nl ** 2 * (1679 / 729 + 256 / 27 * Z2)
        z += a ** 4 * L ** 2 * (94078 / 27 - 18230 / 81 * nl + 493 / 81 * nl ** 2)
        z += a ** 4 * L ** 3 * (28298 / 81 - 428 / 27 * nl) + a ** 4 * L ** 4 * 16 / 81
    return z


def as6(mu, asMZ, Mt, MZ=91.1879, run_loops=4, dec_order=3, mu_dec=None):
    """alpha_s^(6)(mu) in the SM (pure-QCD running), decoupling at mu_dec (default M_t)."""
    mu_dec = Mt if mu_dec is None else mu_dec
    a5 = run_as(asMZ, MZ, mu_dec, 5, run_loops)
    a6 = a5 * dec_z_5to6(a5, mu_dec, Mt, 5, dec_order)
    return run_as(a6, mu_dec, mu, 6, run_loops)


C_POLE = dict(c1=4 / 3, c2=307 / 32 + 2 * Z2 + 2 / 3 * Z2 * LN2 - Z3 / 6 - 5 * (71 / 144 + Z2 / 3),
              c3=190.595 - 26.655 * 5 + 0.6527 * 25, c4=8.49 * PI ** 4)  # nl=5; c4: Marquard et al. 2015 (+-0.25 pi^4)


def mt_msbar(Mt, asMZ, MZ=91.1879, loops=4, run_loops=4, dec_order=3, c4=None):
    """mbar_t(mbar_t) (6-flavour MSbar, pure QCD) from the pole mass M_t."""
    c = [C_POLE['c1'], C_POLE['c2'], C_POLE['c3'], C_POLE['c4'] if c4 is None else c4][:loops]
    m = Mt / 1.06
    for _ in range(60):
        a = as6(m, asMZ, Mt, MZ, run_loops, dec_order) / PI
        m_new = Mt / (1 + sum(ci * a ** (i + 1) for i, ci in enumerate(c)))
        if abs(m_new - m) < 1e-12:
            break
        m = m_new
    return m_new


def mq_run(m0, mu0, mu1, asMZ, nf=5, loops=4, MZ=91.1879):
    """Run an MSbar quark mass with nf flavours (pure QCD)."""
    gm = gamma_m(nf)[:loops]; b = bQCD(nf)[:loops]
    def f(t, y):
        a = y[0]
        return [-sum(bi * a ** (i + 2) for i, bi in enumerate(b)), -y[1] * sum(gi * (4 * a) ** (i + 1) for i, gi in enumerate(gm))]
    a0 = run_as(asMZ, MZ, mu0, nf, loops) / (4 * PI)
    s = solve_ivp(f, (2 * math.log(mu0), 2 * math.log(mu1)), [a0, m0], method='DOP853', rtol=1e-12, atol=1e-20)
    return s.y[1, -1]


# ---------------- SM RGE ----------------
def _load_smdr_hl():
    src = open(SMDR_BETAS, encoding='utf-8', errors='replace').read()
    src = re.sub(r'/\*.*?\*/', ' ', src, flags=re.S)
    out = {}
    for nm in ['betayt4', 'betak4', 'betag34', 'betag4', 'betagp4', 'betayt5', 'betag35']:
        m = re.search(r'\b' + nm + r'\s*=\s*(.*?);', src, flags=re.S)
        out[nm] = compile(m.group(1).replace('Zeta3', 'Z3'), nm, 'eval')
    return out


SMDR_HL = _load_smdr_hl()
NAMES = ['a1', 'a2', 'as', 'at', 'ab', 'atau', 'alam']


def g_to_a(c):
    return np.array([5 / 3 * KAP * c['gY'] ** 2, KAP * c['g2'] ** 2, KAP * c['g3'] ** 2, KAP * c['yt'] ** 2,
                     KAP * c['yb'] ** 2, KAP * c['ytau'] ** 2, KAP * c['lam']])


def a_to_g(x):
    return dict(gY=math.sqrt(0.6 * x[0] / KAP), g1=math.sqrt(x[0] / KAP), g2=math.sqrt(x[1] / KAP), g3=math.sqrt(x[2] / KAP),
                yt=math.sqrt(x[3] / KAP), yb=math.sqrt(max(x[4], 0) / KAP), ytau=math.sqrt(max(x[5], 0) / KAP), lam=x[6] / KAP)


class RGE:
    """d a/d ln mu^2. nloop in 1..3 (full SM); four in {'none','qcd','full'}; five: 5-loop pure QCD g3, y_t."""

    def __init__(self, nloop=3, four='full', five=True):
        self.nloop, self.four, self.five = nloop, four, five
        self.b6 = bQCD(6); self.gm6 = gamma_m(6)

    def ns(self, x):
        a1, a2, as_, at, ab, atau, alam = x
        g3 = math.sqrt(as_ / KAP); g = math.sqrt(a2 / KAP); gp = math.sqrt(0.6 * a1 / KAP); yt = math.sqrt(at / KAP)
        yb = math.sqrt(max(ab, 0) / KAP); ytau = math.sqrt(max(atau, 0) / KAP); k = alam / KAP
        return dict(g3=g3, g=g, gp=gp, yt=yt, yb=yb, ytau=ytau, k=k, Z3=Z3, g32=g3 ** 2, g34=g3 ** 4, g36=g3 ** 6,
                    g38=g3 ** 8, g2=g ** 2, g4=g ** 4, g6=g ** 6, gp2=gp ** 2, gp4=gp ** 4, gp6=gp ** 6, yt2=yt ** 2,
                    yt4=yt ** 4, yt6=yt ** 6, yb2=yb ** 2, yb4=yb ** 4, yb6=yb ** 6, ytau2=ytau ** 2, ytau4=ytau ** 4,
                    ytau6=ytau ** 6, k2=k ** 2, k3=k ** 3)

    def __call__(self, t, x):
        d = np.zeros(7)
        for L in range(1, self.nloop + 1):
            d += np.array(G.betas(L, *x))
        as_, at = x[2], x[3]
        if self.four == 'qcd':
            d[2] += -self.b6[3] * as_ ** 5
            d[3] += -2 * self.gm6[3] * 4 ** 4 * at * as_ ** 4
        elif self.four == 'full':
            n = self.ns(x)
            ev = lambda nm: eval(SMDR_HL[nm], {}, n)
            d[2] += KAP ** 5 * n['g3'] * ev('betag34')
            d[1] += KAP ** 5 * n['g'] * ev('betag4')
            d[0] += 5 / 3 * KAP ** 5 * n['gp'] * ev('betagp4')
            d[3] += KAP ** 5 * n['yt'] * ev('betayt4')
            d[6] += 0.5 * KAP ** 5 * ev('betak4')
        if self.five:
            d[2] += -self.b6[4] * as_ ** 6
            d[3] += -2 * self.gm6[4] * 4 ** 5 * at * as_ ** 5
        return d

    def run(self, c0, mu0, mus, rtol=1e-11):
        """c0: dict of couplings at mu0; returns {mu: dict} for all mu in mus."""
        x0 = g_to_a(c0); out = {}
        up = sorted([m for m in mus if m >= mu0]); dn = sorted([m for m in mus if m < mu0], reverse=True)
        for grp in (up, dn):
            if not grp:
                continue
            ts = [2 * math.log(m) for m in grp]
            s = solve_ivp(self, (2 * math.log(mu0), ts[-1]), x0, method='DOP853', t_eval=ts, rtol=rtol, atol=1e-18)
            for i, m in enumerate(grp):
                out[m] = a_to_g(s.y[:, i])
        return out

    def lam_min(self, c0, mu0, lmu_lo=12.0, lmu_hi=21.0, n=181):
        """minimum of lambda(mu) for log10(mu) in [lo,hi]; returns (log10 mu_min, lambda_min) (parabolic refine)."""
        lm = np.linspace(lmu_lo, lmu_hi, n)
        r = self.run(c0, mu0, list(10 ** lm))
        lam = np.array([r[m]['lam'] for m in 10 ** lm])
        i = int(np.argmin(lam)); i = min(max(i, 1), n - 2)
        x = lm[i - 1:i + 2]; y = lam[i - 1:i + 2]
        A = np.polyfit(x, y, 2)
        xm = -A[1] / (2 * A[0])
        r2 = self.run(c0, mu0, [10 ** xm])
        return xm, r2[10 ** xm]['lam']


# ---------------- external matching codes ----------------
def _env(nthreads=4):
    e = os.environ.copy()
    e['PATH'] = MINGW + ';' + e['PATH']
    e['OMP_NUM_THREADS'] = str(nthreads)
    return e


MR_THREADS = 1  # TSIL may not be thread-safe inside mr's OpenMP sections; see test T3


def mr_eval(points):
    """points: list of dicts with Mb, MW, MZ, MH, Mt, Gf, as_mu, mu -> list of mr JSON records."""
    lines = ['%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g' % (p['Mb'], p['MW'], p['MZ'], p['MH'], p['Mt'], p['Gf'],
                                                                  p['as_mu'], p['mu']) for p in points]
    r = subprocess.run([MR_EXE], input='\n'.join(lines) + '\n', capture_output=True, text=True, env=_env(MR_THREADS), cwd=BUILD)
    recs = [json.loads(l) for l in r.stdout.splitlines() if l.startswith('{')]
    if len(recs) != len(points):
        raise RuntimeError('mr_driver: %d/%d records; stderr=%s' % (len(recs), len(points), r.stderr[-2000:]))
    return recs


ALL_ORDERS = ('x01', 'x02', 'x03', 'x04', 'x10', 'x11', 'x20')


def mr_couplings(rec, alpha='GF', orders=ALL_ORDERS, yb=None, ytau=0.009995):
    """Assemble MSbar couplings at mu from mr coefficients exactly as P2MS does (y-expansion, G_F normalisation)."""
    aQ = rec['as_mu'] / (4 * PI)
    aE = (rec['alpha_GF'] if alpha == 'GF' else rec['alpha_solve']) / (4 * PI)
    o = set(orders)
    def d(p, qcd=False):
        s = 1.0
        if qcd and 'x01' in o: s += aQ * rec[p + '_y01']
        if 'x10' in o: s += aE * rec[p + '_y10']
        if qcd and 'x02' in o: s += aQ ** 2 * rec[p + '_y02']
        if 'x11' in o: s += aE * aQ * rec[p + '_y11']
        if 'x20' in o: s += aE ** 2 * rec[p + '_y20']
        if qcd and 'x03' in o: s += aQ ** 3 * rec[p + '_y03']
        if qcd and 'x04' in o: s += aQ ** 4 * rec[p + '_y04']
        return s
    Gf = rec['Gf']
    gg = 2 ** 2.5 * Gf * rec['MW'] ** 2 * d('W')
    gz = 2 ** 2.5 * Gf * rec['MZ'] ** 2 * d('Z')
    c = dict(g2=math.sqrt(gg), gY=math.sqrt(gz - gg), g3=math.sqrt(4 * PI * rec['as_mu']),
             yt=2 ** 0.75 * math.sqrt(Gf) * rec['Mt'] * d('t', True), lam=Gf / math.sqrt(2) * rec['MH'] ** 2 * d('H'),
             ytau=ytau)
    c['yb'] = yb if yb is not None else 0.0155
    c['g1'] = math.sqrt(5 / 3) * c['gY']
    return c


def mr_eval_par(points, nproc=4):
    """Same as mr_eval but split over <=4 single-threaded mr_driver processes (OpenMP races avoided)."""
    if len(points) <= 1:
        return mr_eval(points)
    chunks = [points[i::nproc] for i in range(nproc)]
    procs = []
    for ch in chunks:
        if not ch:
            procs.append(None); continue
        lines = ['%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g' % (p['Mb'], p['MW'], p['MZ'], p['MH'], p['Mt'], p['Gf'],
                                                                      p['as_mu'], p['mu']) for p in ch]
        pr = subprocess.Popen([MR_EXE], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              env=_env(1), cwd=BUILD)
        procs.append((pr, '\n'.join(lines) + '\n'))
    res = []
    for item in procs:
        if item is None:
            res.append([]); continue
        pr, inp = item
        so, se = pr.communicate(inp)
        res.append([json.loads(l) for l in so.splitlines() if l.startswith('{')])
    out = [None] * len(points)
    for i in range(nproc):
        for j, rec in enumerate(res[i]):
            out[i + j * nproc] = rec
    if any(r is None for r in out):
        raise RuntimeError('mr_eval_par: missing records')
    return out


def smdr_cmds(cmds, timeout=3000):
    r = subprocess.run([SMDR_EXE], input='\n'.join(cmds) + '\n', capture_output=True, text=True, env=_env(1),
                       cwd=BUILD, timeout=timeout)
    return [json.loads(l) for l in r.stdout.splitlines() if l.startswith('{')], r.stderr


def mu_b_and_nodes(MPbar=2.435e18, div=722.0):
    mub = MPbar / div
    us = [0.5 * i for i in range(11)]
    return mub, us, [mub / math.sqrt(math.cosh(2 * u)) for u in us]
