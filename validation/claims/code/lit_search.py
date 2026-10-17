# claims-blind: page-tagged regex search in extracted text. Usage: python -I lit_search.py <pages.txt> <maxhits> <ctx> <regex> [<regex> ...]
import sys, re
sys.stdout.reconfigure(encoding="utf-8")
fn, maxhits, ctx = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
txt = open(fn, encoding="utf-8").read()
pages = re.split(r"\n=====PDFPAGE (\d+)=====\n", txt)
# pages = ['', '1', text1, '2', text2, ...]
P = [(int(pages[i]), pages[i + 1]) for i in range(1, len(pages) - 1, 2)]
for pat in sys.argv[4:]:
    rx = re.compile(pat, re.I | re.S)
    n = 0
    print(f"##### pattern: {pat}")
    for pno, t in P:
        for m in rx.finditer(t):
            a, b = max(0, m.start() - ctx), min(len(t), m.end() + ctx)
            s = t[a:b].replace("\n", " ")
            # printed page label: first short line that looks like a page number at end/start of page
            print(f"[pdf p.{pno}] ...{s}...")
            n += 1
            if n >= maxhits: break
        if n >= maxhits: break
    print()
