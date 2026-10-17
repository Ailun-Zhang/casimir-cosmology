# sm-blind R16 V1: download public codes/papers/PDG tables into new empty dirs under inputs/,
# record URL, date, SHA-256 in SOURCES.json; then extract archives into separate new dirs,
# and extract PDF text to work/txt (outside download dirs).
import os, sys, json, hashlib, urllib.request, datetime, re, tarfile, zipfile, io, gzip
os.environ['OMP_NUM_THREADS'] = '1'
sys.stdout.reconfigure(encoding='utf-8')
MY = r'D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\07_reviews\R16_v12_round3\sm_mapping_blind'
INP = os.path.join(MY, 'inputs')
TXT = os.path.join(MY, 'work', 'txt')
os.makedirs(TXT, exist_ok=True)
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) academic-research-download'
NOW = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def get(url, timeout=240):
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.geturl(), r.headers.get('Content-Type', '')


def newdir(name):
    d = os.path.join(INP, name)
    k = 0
    while os.path.exists(d) and os.listdir(d):
        k += 1
        d = os.path.join(INP, f'{name}_{k}')
    os.makedirs(d, exist_ok=True)
    return d


def save_group(name, items):
    d = newdir(name)
    recs = []
    for urls, fname in items:
        if isinstance(urls, str):
            urls = [urls]
        for url in urls:
            try:
                data, final, ctype = get(url)
                if len(data) < 3000 and b'<html' in data[:800].lower():
                    raise RuntimeError('html stub of %d bytes' % len(data))
                p = os.path.join(d, fname)
                with open(p, 'wb') as f:
                    f.write(data)
                recs.append(dict(file=fname, url=url, final_url=final, content_type=ctype, date_utc=NOW,
                                 bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
                print('OK  ', name, fname, len(data), url)
                break
            except Exception as e:
                print('FAIL', name, url, repr(e)[:160])
                recs.append(dict(file=fname, url=url, error=repr(e)[:300], date_utc=NOW))
    with open(os.path.join(d, 'SOURCES.json'), 'w', encoding='utf-8') as f:
        json.dump(recs, f, indent=1)
    return d, recs


def latest(index_url, pat):
    html, _, _ = get(index_url)
    names = sorted(set(re.findall(pat, html.decode('latin1'))),
                   key=lambda s: [int(x) for x in re.findall(r'\d+', s)])
    return names


report = {}
# 1. mr (Kniehl-Pikelner-Veretin) from GitHub
report['mr'] = save_group('20261014_mr_github', [([
    'https://github.com/apik/mr/archive/refs/heads/master.zip',
    'https://codeload.github.com/apik/mr/zip/refs/heads/master',
    'https://github.com/apik/mr/archive/refs/heads/main.zip'], 'mr-src.zip')])
# 2. SMDR and TSIL from S.P. Martin's pages
for key, idx, pat in [('smdr', 'https://www.niu.edu/spmartin/SMDR/', r'smdr-[\d.]+\.tar\.gz'),
                      ('tsil', 'https://www.niu.edu/spmartin/TSIL/', r'tsil-[\d.]+\.tar\.gz')]:
    try:
        names = latest(idx, pat)
        print(key, 'available:', names)
        nm = names[-1]
        report[key] = save_group('20261014_' + key, [(idx + nm, nm)])
    except Exception as e:
        print('FAIL index', key, repr(e)[:200])
# 3. arXiv PDFs
pdfs = [('1307.3536', 'Buttazzo2013'), ('1507.08833', 'BKPV2015'), ('1601.08143', 'mr_KPV2016'),
        ('1907.02500', 'SMDR_MR2019'), ('1205.6497', 'Degrassi2012')]
report['arxiv_pdf'] = save_group('20261014_arxiv_pdfs',
                                 [('https://arxiv.org/pdf/' + a, f'{n}_{a}.pdf') for a, n in pdfs])
# 4. arXiv LaTeX sources (for beta-function transcription)
srcs = [('1307.3536', 'Buttazzo2013'), ('1205.2892', 'ChetyrkinZoller2012'), ('1201.5868', 'MSS2012'),
        ('1601.08143', 'mr_KPV2016')]
report['arxiv_src'] = save_group('20261014_arxiv_src',
                                 [('https://arxiv.org/e-print/' + a, f'{n}_{a}.src') for a, n in srcs])
# 5. PDG 2026 (fallback 2025, 2024)
pdg = []
for stem in ['reviews/rpp{Y}-rev-phys-constants.pdf', 'tables/rpp{Y}-sum-gauge-higgs-bosons.pdf',
             'tables/rpp{Y}-sum-quarks.pdf', 'tables/rpp{Y}-sum-leptons.pdf']:
    urls = ['https://pdg.lbl.gov/%d/' % Y + stem.replace('{Y}', str(Y)) for Y in (2026, 2025, 2024)]
    pdg.append((urls, stem.split('/')[-1].replace('{Y}', 'YYYY')))
report['pdg'] = save_group('20261014_pdg', pdg)

# ---- extraction of archives into separate new dirs ----
def safe_members_tar(t, dest):
    out = []
    for m in t.getmembers():
        p = os.path.realpath(os.path.join(dest, m.name))
        if not p.startswith(os.path.realpath(dest)) or m.issym() or m.islnk():
            continue
        out.append(m)
    return out

ext = {}
d, recs = report['mr']
for r in recs:
    if 'sha256' in r:
        xd = newdir('20261014_mr_extracted')
        with zipfile.ZipFile(os.path.join(d, r['file'])) as z:
            for n in z.namelist():
                p = os.path.realpath(os.path.join(xd, n))
                if p.startswith(os.path.realpath(xd)):
                    z.extract(n, xd)
        ext['mr'] = xd
for key in ('smdr', 'tsil'):
    if key in report:
        d, recs = report[key]
        for r in recs:
            if 'sha256' in r:
                xd = newdir('20261014_%s_extracted' % key)
                with tarfile.open(os.path.join(d, r['file'])) as t:
                    t.extractall(xd, members=safe_members_tar(t, xd))
                ext[key] = xd
d, recs = report['arxiv_src']
xs = newdir('20261014_arxiv_src_extracted')
for r in recs:
    if 'sha256' not in r:
        continue
    raw = open(os.path.join(d, r['file']), 'rb').read()
    sub = os.path.join(xs, r['file'].replace('.src', ''))
    os.makedirs(sub, exist_ok=True)
    try:
        dat = gzip.decompress(raw)
    except Exception:
        dat = raw
    try:
        with tarfile.open(fileobj=io.BytesIO(dat)) as t:
            t.extractall(sub, members=safe_members_tar(t, sub))
    except Exception:
        with open(os.path.join(sub, 'main.tex'), 'wb') as f:
            f.write(dat)
ext['arxiv_src'] = xs

# ---- PDF text extraction ----
from pypdf import PdfReader
for key in ('arxiv_pdf', 'pdg'):
    d, recs = report[key]
    for r in recs:
        if 'sha256' not in r:
            continue
        try:
            rd = PdfReader(os.path.join(d, r['file']))
            with open(os.path.join(TXT, r['file'] + '.txt'), 'w', encoding='utf-8') as f:
                for i, pg in enumerate(rd.pages):
                    f.write('\n\n=== PAGE %d ===\n' % (i + 1))
                    f.write(pg.extract_text() or '')
            print('TXT', r['file'], len(rd.pages), 'pages')
        except Exception as e:
            print('PDF FAIL', r['file'], repr(e)[:150])

with open(os.path.join(MY, 'work', 'download_report.json'), 'w', encoding='utf-8') as f:
    json.dump({'time_utc': NOW, 'dirs': {k: v[0] for k, v in report.items()}, 'extracted': ext}, f, indent=1)
print('EXTRACTED', json.dumps(ext, indent=1))
# listing of extracted code trees (sizes)
for key in ('mr', 'smdr', 'tsil'):
    if key in ext:
        print('---- tree', key)
        n = 0
        for root, dirs, files in os.walk(ext[key]):
            for fn in files:
                p = os.path.join(root, fn)
                n += 1
                if n <= 140:
                    print('%9d %s' % (os.path.getsize(p), os.path.relpath(p, ext[key])))
        print('files:', n)
for root, dirs, files in os.walk(xs):
    for fn in files:
        p = os.path.join(root, fn)
        print('%9d %s' % (os.path.getsize(p), os.path.relpath(p, xs)))
