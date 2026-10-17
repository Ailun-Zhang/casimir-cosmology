# PROVENANCE: new tool written 2026-10-16 by code-finalize (v12 code/data package).
"""Generate numbers.tex -- the LaTeX macros of the claim-layer numbers shared by the PRL, the PRD and the companion
technical report -- from the single source numbers/numbers_v12.json.

    python numbers/make_numbers_tex.py                       # write numbers/numbers.tex
    python numbers/make_numbers_tex.py --out PATH [PATH ...]  # also write copies (e.g. into the PRL/PRD/report sources)
    python numbers/make_numbers_tex.py --check                # exit 1 if numbers/numbers.tex is not up to date
    python numbers/make_numbers_tex.py --compare PATH         # exit 1 if PATH differs from the generated text

Change a number only in numbers_v12.json: entry "numbers.<Macro>", field "values" (one value per number in the
macro body) and "precision" (decimals shown); the body is template.format(values rounded to precision).  The comment
of each macro (meaning, unrounded value, registry id) is kept in "comment"; "source" names the package file and JSON
pointer of every value (not written to numbers.tex); "check" holds the last check_numbers.py result.
Never edit numbers.tex by hand: regenerate it and copy it to the three documents.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "numbers_v12.json")
DST = os.path.join(HERE, "numbers.tex")


def body(entry):
    """Macro body: the template filled with the values, each rounded to its precision."""
    vals = entry["values"]
    prec = entry["precision"] if isinstance(entry["precision"], list) else [entry["precision"]] * len(vals)
    return entry["template"].format(*["%.*f" % (p, v) for v, p in zip(vals, prec)])


def render(doc):
    """Lines in layout order; each line ends with meta.newline unless its layout item gives its own "eol" (the
    imported file mixes LF and CRLF line ends; they are reproduced byte for byte)."""
    nl = doc["meta"].get("newline", "\n")
    out = []
    for item in doc["layout"]:
        if "text" in item:
            ln = item["text"]
        else:
            name = item["macro"]
            e = doc["numbers"][name]
            ln = "\\newcommand{\\%s}{%s}" % (name, body(e))
            if e.get("comment") is not None:
                ln += e.get("pad", " ") + "%" + e["comment"]
        out.append(ln + item.get("eol", nl))
    return "".join(out)


def load():
    with open(SRC, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", nargs="*", default=[])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--compare", nargs="*", default=[])
    a = ap.parse_args()
    data = render(load()).encode("utf-8")
    if a.check or a.compare:
        bad = 0
        for p in ([DST] if a.check else []) + a.compare:
            same = os.path.exists(p) and open(p, "rb").read() == data
            bad += not same
            print("%s %s" % ("identical" if same else "DIFFERS  ", p))
        return 1 if bad else 0
    for p in [DST] + a.out:
        with open(p, "wb") as f:
            f.write(data)
        print("wrote", p, len(data), "bytes")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
