import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
sys.stdout.reconfigure(encoding="utf-8")
from pypdf import PdfReader

src_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
for fn in sorted(os.listdir(src_dir)):
    if not fn.lower().endswith(".pdf"):
        continue
    r = PdfReader(os.path.join(src_dir, fn))
    parts = []
    for i, p in enumerate(r.pages):
        try:
            t = p.extract_text() or ""
        except Exception as e:
            t = f"[extract error {e}]"
        parts.append(f"\n===== PAGE {i+1} =====\n" + t)
    txt = "".join(parts)
    with open(os.path.join(out_dir, fn[:-4] + ".txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    first = txt[:300].replace("\n", " ")
    print(fn, len(r.pages), "pages |", first)
