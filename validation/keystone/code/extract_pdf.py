import sys, hashlib
sys.stdout.reconfigure(encoding='utf-8')
from pypdf import PdfReader
src, dst = sys.argv[1], sys.argv[2]
h = hashlib.sha256(open(src, 'rb').read()).hexdigest()
r = PdfReader(src)
with open(dst, 'w', encoding='utf-8') as f:
    for i, p in enumerate(r.pages):
        f.write(f"\n===== PAGE {i+1} =====\n")
        f.write(p.extract_text() or "")
print("pages", len(r.pages), "sha256", h)
