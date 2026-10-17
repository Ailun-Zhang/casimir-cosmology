# sm-blind: fetch mr GitHub release tarball(s) (may bundle tsil-1.3 + odeint subset) into a new empty dir
import os, sys, json, hashlib, urllib.request, datetime, tarfile, re
sys.stdout.reconfigure(encoding='utf-8')
MY = r'D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\07_reviews\R16_v12_round3\sm_mapping_blind'
INP = os.path.join(MY, 'inputs')
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) academic-research-download'
NOW = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': '*/*'})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read(), r.geturl()


def newdir(name):
    d = os.path.join(INP, name); k = 0
    while os.path.exists(d) and os.listdir(d):
        k += 1; d = os.path.join(INP, f'{name}_{k}')
    os.makedirs(d, exist_ok=True); return d

rel, _ = get('https://api.github.com/repos/apik/mr/releases')
rel = json.loads(rel)
assets = []
for r in rel:
    for a in r.get('assets', []):
        assets.append((r['tag_name'], a['name'], a['browser_download_url'], a.get('size')))
print('releases/assets:', json.dumps(assets, indent=0)[:3000])
# choose latest .tar.gz asset
tg = [a for a in assets if a[1].endswith('.tar.gz')]
d = newdir('20261014_mr_release')
recs = []
for tag, name, url, size in tg[:1]:
    data, final = get(url)
    open(os.path.join(d, name), 'wb').write(data)
    recs.append(dict(file=name, tag=tag, url=url, final_url=final, date_utc=NOW, bytes=len(data),
                     sha256=hashlib.sha256(data).hexdigest()))
    print('OK', name, len(data))
json.dump(recs, open(os.path.join(d, 'SOURCES.json'), 'w'), indent=1)
xd = newdir('20261014_mr_release_extracted')
for r in recs:
    with tarfile.open(os.path.join(d, r['file'])) as t:
        mem = [m for m in t.getmembers() if not (m.issym() or m.islnk()) and
               os.path.realpath(os.path.join(xd, m.name)).startswith(os.path.realpath(xd))]
        t.extractall(xd, members=mem)
# summarize top-level structure and whether tsil / boost bundled
for root, dirs, files in os.walk(xd):
    depth = os.path.relpath(root, xd).count(os.sep)
    if depth <= 2:
        print('%s/ (%d files, %d dirs)' % (os.path.relpath(root, xd), len(files), len(dirs)))
print('EXTRACTED', xd)
