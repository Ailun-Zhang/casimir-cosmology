# sm-blind R16 V1: build TSIL-1.3 + mr (GitHub master) [matching A], TSIL-1.46 + 3VIL + SMDR-1.3 [matching B]
# with MinGW-w64 gcc/g++ 5.3; <=4 parallel compile workers. Objects/libs go to work/build (not into inputs/).
# Usage: python -I build_all.py [libs|drivers|all]
import os, sys, subprocess, re, time, json, glob, hashlib, tarfile, io, gzip
from concurrent.futures import ThreadPoolExecutor
sys.stdout.reconfigure(encoding='utf-8')
MY = r'D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\07_reviews\R16_v12_round3\sm_mapping_blind'
INP = os.path.join(MY, 'inputs')
B = os.path.join(MY, 'work', 'build')
DRV = os.path.join(MY, 'work', 'drivers')
os.makedirs(B, exist_ok=True)
MINGW = r'D:\Python\envs\cbuild\Library\mingw-w64\bin'
USRB = r'D:\Python\envs\cbuild\Library\usr\bin'
env = os.environ.copy()
env['PATH'] = MINGW + ';' + USRB + ';' + env['PATH']
GCC, GXX, AR = [os.path.join(MINGW, x) for x in ('gcc.exe', 'g++.exe', 'ar.exe')]
MRREL = os.path.join(INP, '20261014_mr_release_extracted', 'mr-1.3.2')
MRSRC = os.path.join(INP, '20261014_mr_extracted', 'mr-master', 'mr')
TSIL13 = os.path.join(MRREL, 'tsil-1.3')
TSIL146 = os.path.join(INP, '20261014_tsil_extracted', 'TSIL-1.46')
SMDR = os.path.join(INP, '20261014_smdr_extracted', 'SMDR-1.3')
LOG = open(os.path.join(B, 'build_log.txt'), 'a', encoding='utf-8')
STUB = os.path.join(B, 'stub_include')  # MinGW lacks execinfo.h (glibc backtrace); no-op stub
os.makedirs(STUB, exist_ok=True)
with open(os.path.join(STUB, 'execinfo.h'), 'w') as _f:
    _f.write('#ifndef STUB_EXECINFO_H\n#define STUB_EXECINFO_H\n'
             'static inline int backtrace(void **b, int s) { (void)b; (void)s; return 0; }\n'
             'static inline char **backtrace_symbols(void *const *b, int s) { (void)b; (void)s; return 0; }\n'
             'static inline void backtrace_symbols_fd(void *const *b, int s, int fd) { (void)b; (void)s; (void)fd; }\n'
             '#endif\n')
CF = ['-O2', '-DTSIL_SIZE_LONG', '-D__USE_MINGW_ANSI_STDIO=1', '-D_USE_MATH_DEFINES', '-I' + STUB]
SKIP_EXISTING = True  # recompile only objects that failed / are missing


def log(s):
    LOG.write(s + '\n'); LOG.flush(); print(s, flush=True)


def run(cmd, cwd=B):
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, errors='replace')
    dt = time.time() - t0
    tag = 'OK ' if p.returncode == 0 else 'ERR'
    log('%s %6.1fs %s' % (tag, dt, ' '.join(os.path.basename(c) if i == 0 else c for i, c in enumerate(cmd))[-300:]))
    if p.returncode != 0:
        log('   STDERR: ' + p.stderr[-4000:])
    return p.returncode


def has_main(path):
    return re.search(r'\bint\s+main\s*\(', open(path, errors='ignore').read()) is not None


def jobs_c(srcdir, outdir, flags, srcs=None, cc=GCC, ext='.c'):
    os.makedirs(outdir, exist_ok=True)
    if srcs is None:
        srcs = [f for f in glob.glob(os.path.join(srcdir, '*' + ext)) if not has_main(f)]
    out = []
    for s in srcs:
        rel = os.path.relpath(s, srcdir).replace(os.sep, '_')
        o = os.path.join(outdir, rel.rsplit('.', 1)[0] + '.o')
        out.append((os.path.getsize(s), [cc] + flags + ['-c', s, '-o', o], o))
    return out


def extract_3vil():
    xd = os.path.join(INP, '20261014_3vil_extracted')
    if os.path.isdir(xd) and os.listdir(xd):
        return glob.glob(os.path.join(xd, '3vil*'))[0]
    os.makedirs(xd, exist_ok=True)
    src = glob.glob(os.path.join(SMDR, '3vil*.tar.gz'))[0]
    with tarfile.open(src) as t:
        mem = [m for m in t.getmembers() if not (m.issym() or m.islnk()) and
               os.path.realpath(os.path.join(xd, m.name)).startswith(os.path.realpath(xd))]
        t.extractall(xd, members=mem)
    rec = dict(file=os.path.basename(src), origin='bundled inside SMDR-1.3 tarball', extracted_to=xd,
               sha256=hashlib.sha256(open(src, 'rb').read()).hexdigest())
    json.dump([rec], open(os.path.join(xd, 'SOURCES.json'), 'w'), indent=1)
    return glob.glob(os.path.join(xd, '3vil*'))[0]


def build_libs():
    T3 = extract_3vil()
    groups = {}
    groups['tsil13'] = jobs_c(TSIL13, os.path.join(B, 'tsil13'), CF)
    groups['tsil146'] = jobs_c(TSIL146, os.path.join(B, 'tsil146'), CF)
    groups['3vil'] = jobs_c(T3, os.path.join(B, '3vil'), CF + ['-I' + TSIL146])
    smdr_srcs = [os.path.join(SMDR, 'src', s) for s in
                 'io.c arguments.c messages.c utilities.c fit_obs.c betas.c RGrun.c rk6_RG.c effpot.c GFermi.c Mh.c '
                 'MW.c MZ.c Mt.c Mlight.c decouple_thZW.c QCDQED_match.c betas_QCDQED.c RGrun_QCDQED.c'.split()]
    groups['smdr'] = jobs_c(os.path.join(SMDR, 'src'), os.path.join(B, 'smdr'),
                            CF + ['-fcommon', '-I' + TSIL146, '-I' + T3], srcs=smdr_srcs)
    mr_srcs = [f for f in glob.glob(os.path.join(MRSRC, '**', '*.cpp'), recursive=True) if not f.endswith('GL01.cpp')]  # GL01 not in Makefile.am
    groups['mr'] = jobs_c(MRSRC, os.path.join(B, 'mr'),
                          ['-std=gnu++14', '-fopenmp', '-D__USE_ISOC99'] + CF + ['-I' + MRSRC, '-I' + TSIL13, '-I' + MRREL],
                          srcs=mr_srcs, cc=GXX, ext='.cpp')
    alljobs = []
    for g, js in groups.items():
        for sz, cmd, o in js:
            if SKIP_EXISTING and os.path.exists(o) and os.path.getsize(o) > 0:
                continue
            alljobs.append((sz if g == 'mr' else sz / 4, g, cmd, o))
    alljobs.sort(key=lambda x: -x[0])  # big files first
    log('=== compiling %d objects with 4 workers' % len(alljobs))
    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        rcs = list(ex.map(lambda j: run(j[2]), alljobs))
    log('=== compile wall %.1fs, failures %d' % (time.time() - t0, sum(1 for r in rcs if r)))
    for g, js in groups.items():
        lib = os.path.join(B, 'lib%s.a' % g)
        if os.path.exists(lib):
            os.remove(lib)
        objs = [o for _, _, o in js if os.path.exists(o)]
        run([AR, 'rcs', lib] + objs)
    return T3


def build_drivers(T3):
    ok = True
    mrd = os.path.join(DRV, 'mr_driver.cpp')
    if os.path.exists(mrd):
        ok &= run([GXX, '-std=gnu++14', '-fopenmp', '-D__USE_ISOC99'] + CF + ['-I' + MRSRC, '-I' + TSIL13, '-I' + MRREL, mrd,
                   '-o', os.path.join(B, 'mr_driver.exe'), '-L' + B, '-lmr', '-ltsil13', '-lm']) == 0
    for c in glob.glob(os.path.join(DRV, 'smdr_*.c')):
        exe = os.path.join(B, os.path.basename(c)[:-2] + '.exe')
        ok &= run([GCC] + CF + ['-fcommon', '-I' + os.path.join(SMDR, 'src'), '-I' + TSIL146, '-I' + T3, c, '-o', exe,
                   '-L' + B, '-lsmdr', '-l3vil', '-ltsil146', '-lm']) == 0
    return ok


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    T3 = extract_3vil()
    if what in ('libs', 'all'):
        T3 = build_libs()
    if what in ('drivers', 'all'):
        build_drivers(T3)
    for f in sorted(glob.glob(os.path.join(B, '*.a')) + glob.glob(os.path.join(B, '*.exe'))):
        log('%s %d %s' % (os.path.basename(f), os.path.getsize(f), hashlib.sha256(open(f, 'rb').read()).hexdigest()[:16]))
    log('=== DONE ' + what)
