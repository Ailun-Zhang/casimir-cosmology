# claims-blind: download one document into a download dir (SOURCES.json, SHA-256) and write page-tagged text.
# Usage: python -I fetch_one.py <download_dir> <txt_out_dir> <name> <url>
import sys, os, json, hashlib, time, urllib.request
sys.stdout.reconfigure(encoding="utf-8")
DL, TXT, name, url = sys.argv[1:5]
os.makedirs(DL, exist_ok=True); os.makedirs(TXT, exist_ok=True)
src_path = os.path.join(DL, "SOURCES.json")
sources = json.load(open(src_path, encoding="utf-8")) if os.path.exists(src_path) else []
pdf = os.path.join(DL, name + ".pdf")
if not os.path.exists(pdf):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research download; claims-blind)"})
    with urllib.request.urlopen(req, timeout=300) as r, open(pdf, "wb") as f:
        f.write(r.read())
    sha = hashlib.sha256(open(pdf, "rb").read()).hexdigest()
    sources.append(dict(file=os.path.basename(pdf), url=url, date_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), sha256=sha, bytes=os.path.getsize(pdf), downloaded_by="claims-blind"))
    json.dump(sources, open(src_path, "w", encoding="utf-8"), indent=1)
    print("downloaded", name, os.path.getsize(pdf), sha, flush=True)
from pypdf import PdfReader
rd = PdfReader(pdf)
with open(os.path.join(TXT, name + ".pages.txt"), "w", encoding="utf-8") as f:
    for i, pg in enumerate(rd.pages):
        try:
            t = pg.extract_text() or ""
        except Exception as e:
            t = f"[extract error {e}]"
        f.write(f"\n=====PDFPAGE {i+1}=====\n{t}\n")
print("extracted", name, len(rd.pages), "pages", flush=True)
