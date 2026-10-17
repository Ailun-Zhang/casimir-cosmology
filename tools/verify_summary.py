# PROVENANCE: new tool written 2026-10-15 by code-higgs (v12 code/data package).
"""Summarise all VERIFY.json files of the package (one row per result directory and mode).

    python tools/verify_summary.py                 # print a Markdown table
    python tools/verify_summary.py --update-readmes  # also refresh the block between
                                                     # <!-- VERIFY-BLOCK-START --> and <!-- VERIFY-BLOCK-END -->
                                                     # in each result directory's README.md
"""
import glob
import json
import os
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START, END = "<!-- VERIFY-BLOCK-START -->", "<!-- VERIFY-BLOCK-END -->"


def load(p):
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


def block(v):
    lines = ["| mode | date (UTC) | checks passed | skipped | timings (s) |", "|---|---|---|---|---|"]
    for m in ("quick", "reduced", "full"):
        if m not in v:
            continue
        r = v[m]
        t = r.get("timings_s", {})
        tt = "; ".join("%s %.0f" % (k, s) for k, s in t.items()) if t else ""
        lines.append("| %s | %s | %s/%s%s | %s | %s |" % (m, r.get("utc", ""), r.get("n_pass"), r.get("n_checks"),
                                                       "" if r.get("all_pass") else " (FAIL)", r.get("n_skipped", 0), tt))
    fails = [(m, c["id"]) for m in ("quick", "reduced", "full") if m in v for c in v[m]["checks"]
             if c.get("passed") is False]
    if fails:
        lines.append("")
        lines.append("Failed checks: " + ", ".join("%s:%s" % f for f in fails))
    lines.append("")
    lines.append("Machine of record: %s, %s CPUs, Python %s, numpy %s, scipy %s." % (
        v.get("quick", v.get("full", {})).get("processor", "?"), v.get("quick", v.get("full", {})).get("cpu_count", "?"),
        v.get("quick", v.get("full", {})).get("python", "?"), v.get("quick", v.get("full", {})).get("numpy", "?"),
        v.get("quick", v.get("full", {})).get("scipy", "?")))
    return "\n".join(lines)


MODES = ("quick", "reduced", "full")


def cell(v, m):
    """'n_pass/n_checks' of one mode ('-' if the mode was not run); skipped optional checks and failures are marked."""
    if m not in v:
        return "-"
    r = v[m]
    s = "%s/%s" % (r.get("n_pass"), r.get("n_checks"))
    if r.get("n_skipped"):
        s += " (%s skipped)" % r["n_skipped"]
    return s + ("" if r.get("all_pass") else " FAIL")


def main():
    # 2026-10-15 (code-finalize): one table format (quick | reduced | full) for all chains, so that
    # `--update-readmes` keeps every README marker block equal to its own VERIFY.json.
    upd = "--update-readmes" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None   # e.g. --only higgs
    rows = ["| result directory | quick | reduced | full |", "|---|---|---|---|"]
    for p in sorted(glob.glob(os.path.join(PKG, "*", "*", "VERIFY.json"))):
        d = os.path.dirname(p)
        v = load(p)
        rows.append("| %s | %s |" % (os.path.relpath(d, PKG).replace(os.sep, "/"), " | ".join(cell(v, m) for m in MODES)))
        rd = os.path.join(d, "README.md")
        if only and os.path.relpath(d, PKG).replace(os.sep, "/").split("/")[0] != only:
            continue
        if upd and os.path.exists(rd):
            txt = open(rd, encoding="utf-8").read()
            if START in txt and END in txt:
                a, b = txt.index(START) + len(START), txt.index(END)
                txt = txt[:a] + "\n" + block(v) + "\n" + txt[b:]
                with open(rd, "w", encoding="utf-8", newline="\n") as f:
                    f.write(txt)
    table = "\n".join(rows)
    if upd:   # chain READMEs (e.g. higgs/README.md) carry the summary table between the SUMMARY markers
        s0, s1 = "<!-- VERIFY-SUMMARY-START -->", "<!-- VERIFY-SUMMARY-END -->"
        for rd in glob.glob(os.path.join(PKG, "*", "README.md")):
            if only and os.path.basename(os.path.dirname(rd)) != only:
                continue
            txt = open(rd, encoding="utf-8").read()
            if s0 in txt and s1 in txt:
                chain = os.path.basename(os.path.dirname(rd)) + "/"
                sub = "\n".join([r for r in rows[:2]] + [r for r in rows[2:] if r.startswith("| " + chain)])
                a, b = txt.index(s0) + len(s0), txt.index(s1)
                txt = txt[:a] + "\n" + sub + "\n" + txt[b:]
                with open(rd, "w", encoding="utf-8", newline="\n") as f:
                    f.write(txt)
    print(table)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
