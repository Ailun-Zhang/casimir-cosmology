# PROVENANCE: new tool written 2026-10-16 by code-finalize (v12 code/data package); revised 2026-10-16 by rev-code
#   (R18 MH-01/R2: an explicit file#key source for every macro, sign-aware and key-exact matching, registry key syntax).
"""Check every number of numbers/numbers_v12.json against the archived production outputs shipped in the package.

    python numbers/check_numbers.py              # report; exit 1 if a number has no source or disagrees with it
    python numbers/check_numbers.py --write      # also store the result in numbers_v12.json ("check" of each entry)
    python numbers/check_numbers.py --registry   # also trace every row of the run05 number registry (exit 1 if a
                                                 # row disagrees with its result file)

Source of a number: the field "source" of its entry in numbers_v12.json, a list of
    {"file": "<package-relative path>", "pointer": "<JSON pointer, RFC 6901>"}   (optional "sign": -1, "magnitude": true)
Each pointer must end at a single number (key-exact; list positions are integers).  The quantity is sign x number;
the macro shows the quantity, or its magnitude when "magnitude" is true (lower errors that the papers typeset as
"-\\Macro").  With one source per value of the macro, value i is compared with source i; otherwise every value with
every source until one agrees.  Numbers that are functions of archived outputs (\\xiHminEW, \\lamcErrUp/Dn) point
to numbers/derived_numbers.json, which numbers/derive_numbers.py writes from the files it names.
Status per number:
  verified    every shown value equals its source rounded to the shown precision, sign included; every unrounded
              value quoted in the comment agrees with the source to its last digit (a value written with an explicit
              sign is compared with the signed quantity, an unsigned one with what the macro shows; for a derived
              number every decimal of the comment must be its value or one of its inputs); every registry row named
              in the comment (R5-xxx) holds the number (or, for a derived number, one of its inputs) and agrees with
              its result file; derived_numbers.json is up to date;
  unresolved  no source given;
  mismatch    a source is given but one of these checks fails (missing file or key, key not a number, disagreement).
Registry rows (higgs/09_claims/expected/w/claims_run05/run01/numbers_run05.csv, the archived C161 registry) are traced
through their columns source_file and source_key_or_line: '|' separates keys; ';' starts a further key (a single key
is looked up next to the previous match, then further up); '*' and other shell patterns match any key; '{a,b}'
expands; 'a/b' lists alternatives ('x_P0.1/P0.05' = x_P0.1 and x_P0.05); in a list of records a name selects the
record with that "id"; a path that does not start at the top level is matched anywhere in the file; '(...)' remarks
are ignored; a file-name pattern ('C154_*summary.json') tries every match, and of several package copies of a file the
one whose directories match the registry path best is used.  Every decimal number of the row's value must be found,
sign included, among the numbers below the matched keys; a number found only next to them (the key column names part
of the row) is reported as such.  Where the result file stores the size of a lower error (under a key 'minus...', or
as the second entry of a 'levels' pair [up, down]), the registry writes it with a minus sign; such a negative
registry number is compared with the negated stored size.
"""
import csv
import fnmatch
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import derive_numbers  # noqa: E402

SRC = os.path.join(HERE, "numbers_v12.json")
REG = os.path.join(PKG, "higgs", "09_claims", "expected", "w", "claims_run05", "run01", "numbers_run05.csv")
DERIVED = "numbers/derived_numbers.json"
CLAIMS_DIR = "w/claims_run05/run01/"
NUMRE = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w])")
_CACHE = {}


def rel(p):
    return os.path.relpath(p, PKG).replace(os.sep, "/")


def load(path):
    if path not in _CACHE:
        with open(path, encoding="utf-8-sig") as f:
            _CACHE[path] = json.load(f)
    return _CACHE[path]


def decimals(s):
    return len(s.split(".")[1]) if "." in s else 0


def agrees(x, v, p):
    """x (shown with p decimals) is v rounded to p decimals; the sign counts."""
    return abs(x - v) <= 0.5 * 10.0 ** (-p) + 1e-12


def pointer(path):
    return "/" + "/".join(str(k).replace("~", "~0").replace("/", "~1") for k in path)


# ---------------------------------------------------------------- explicit sources (key-exact, sign-aware)
def resolve(src):
    """(label, quantity, shown_form) of one source entry; ValueError if the file, the key or the number is missing."""
    path = os.path.join(PKG, *src["file"].split("/"))
    if not os.path.exists(path):
        raise ValueError("file not in the package: %s" % src["file"])
    cur = load(path)
    for tok in src["pointer"].split("/")[1:]:
        tok = tok.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, dict) and tok in cur:
            cur = cur[tok]
        elif isinstance(cur, list) and tok.isdigit() and int(tok) < len(cur):
            cur = cur[int(tok)]
        else:
            raise ValueError("key %s not in %s" % (src["pointer"], src["file"]))
    if isinstance(cur, bool) or not isinstance(cur, (int, float)):
        raise ValueError("%s#%s is not a single number" % (src["file"], src["pointer"]))
    q = src.get("sign", 1) * float(cur)
    label = "%s#%s" % (src["file"], src["pointer"]) + (" (x -1)" if src.get("sign", 1) == -1 else "") + \
            (" (magnitude shown)" if src.get("magnitude") else "")
    return label, q, abs(q) if src.get("magnitude") else q


def comment_numbers(comment):
    """Decimal numbers of a comment outside key paths, registry ids, file names and macro names."""
    txt = re.sub(r"R5-\d+|\b[A-Za-z]+\d*\.json\b|\S*[A-Za-z_]\S*\.\S*|\b[A-Z]+\d+\b|\\\w+", " ", comment)
    return [m.group() for m in NUMRE.finditer(txt) if "." in m.group()]


def derived_inputs(srcs):
    """Values and input values (also of derived inputs) of the derived numbers named by the sources."""
    p = os.path.join(PKG, *DERIVED.split("/"))
    d = load(p) if os.path.exists(p) else {}
    vals, todo, seen = [], [s["pointer"].split("/")[1] for s in srcs if s["file"] == DERIVED], set()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        ent = d.get(name, {})
        vals += [ent.get("value")] + [i.get("value") for i in ent.get("inputs", {}).values()]
        todo += [i["pointer"].split("/")[1] for i in ent.get("inputs", {}).values() if i.get("file") == DERIVED]
    return [float(v) for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]


# ---------------------------------------------------------------- registry rows
def file_index():
    idx = {}
    for pat in ("*/*/expected/**/*.json", "*/data/**/*.json", "*/*/inputs/**/*.json"):
        for p in glob.glob(os.path.join(PKG, pat), recursive=True):
            idx.setdefault(os.path.basename(p), []).append(p)
    for k in idx:   # prefer the run05 claims copies, then the shortest path
        idx[k].sort(key=lambda p: (CLAIMS_DIR not in rel(p) or "regress_run02" in p or "t1new" in p, len(p)))
    return idx


def expand(comp):
    m = re.search(r"\{([^{}]*)\}", comp)
    if not m:
        return [comp]
    return [x for alt in m.group(1).split(",") for x in expand(comp[:m.start()] + alt + comp[m.end():])]


def children(node, comp):
    """(key, child) pairs of a dict/list node that match one key component of the registry syntax (in a list of
    records, a name selects the record with that "id")."""
    if isinstance(node, list):
        if comp == "*":
            return list(enumerate(node))
        if comp.isdigit():
            return [(int(comp), node[int(comp)])] if int(comp) < len(node) else []
        ids = expand(comp)
        return [(i, v) for i, v in enumerate(node) if isinstance(v, dict) and v.get("id") in ids]
    if not isinstance(node, dict):
        return []
    out = []
    for c in expand(comp):
        if c in node:
            out.append((c, node[c]))
        elif any(ch in c for ch in "*?["):
            out += [(k, v) for k, v in node.items() if fnmatch.fnmatchcase(k, c)]
        elif "/" in c:
            alts = c.split("/")
            for a in alts:
                cut = alts[0].rfind(a[:1]) if a not in node else -1
                k = a if a in node else (alts[0][:cut] + a if cut > 0 else None)
                if k in node:
                    out.append((k, node[k]))
    return out


def match(node, comps, path=()):
    if not comps:
        return [(path, node)]
    return [m for k, v in children(node, comps[0]) for m in match(v, comps[1:], path + (k,))]


def match_anywhere(node, comps, path=()):
    out = match(node, comps, path)
    items = node.items() if isinstance(node, dict) else enumerate(node) if isinstance(node, list) else []
    for k, v in items:
        out += match_anywhere(v, comps, path + (k,))
    return out


def at(node, path):
    for k in path:
        node = node[k]
    return node


def leaves(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from leaves(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from leaves(v, path + (i,))
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield path, float(node)


def lower_size(path):
    """A stored lower-error size: under a key 'minus...', or the second entry of a 'levels' pair [up, down]."""
    return any(str(k).startswith("minus") for k in path) or (len(path) >= 3 and path[-3] == "levels" and path[-1] == 1)


def hits_of(s, vals):
    x, p = float(s), decimals(s)
    return [(fp, q) for fp, q, v in vals if agrees(x, v, p) or (x < 0 and lower_size(q) and agrees(x, -v, p))]


def result_files(source_file, idx):
    """Package files for a registry source_file (name pattern allowed): per matching name, the copy whose directories
    share the longest tail with the registry path (ties: the preferred copy)."""
    parts = [x for x in source_file.replace("\\", "/").split("/") if x]
    if not parts:
        return []
    dirs, out = parts[:-1], []

    def score(p):
        pd, n = rel(p).split("/")[:-1], 0
        while n < min(len(pd), len(dirs)) and pd[-1 - n] == dirs[-1 - n]:
            n += 1
        return n
    for base in sorted(idx):
        if fnmatch.fnmatchcase(base, parts[-1]):
            out.append(max(idx[base], key=lambda p: (score(p), -idx[base].index(p))))
    return out


def trace_row(row, idx):
    """None if the row's result file is not in the package, else dict(ok, where, missing, near)."""
    files = result_files((row.get("source_file") or "").strip(), idx)
    if not files:
        return None
    spec = re.sub(r"\([^()]*\)", "", row.get("source_key_or_line") or "")
    found = []   # (file, path)
    for fp in files:
        root, prev = load(fp), []
        for part in [s.strip() for s in spec.split(";") if s.strip()]:
            comps = [c.strip() for c in part.split("|")]
            hits = []
            if len(comps) == 1:   # a bare key: next to the previous match, then further up
                for p, _ in prev:
                    for d in range(len(p) - 1, -1, -1):
                        h = match(at(root, p[:d]), comps, p[:d])
                        if h:
                            hits += h
                            break
            hits = hits or match(root, comps) or match_anywhere(root, comps)
            found += [(fp, p) for p, _ in hits]
            prev = hits or prev
    if not found:
        return dict(ok=False, where=", ".join(rel(f) for f in files), missing=["keys not found"], near=[])
    vals = [(fp, p, v) for fp, p0 in found for p, v in leaves(at(load(fp), p0), p0)]
    nvals = [(fp, p, v) for fp, pp in {(fp, p0[:-1]) for fp, p0 in found if p0} for p, v in leaves(at(load(fp), pp), pp)]
    missing, near = [], set()
    for m in NUMRE.finditer(row.get("value") or ""):
        s = m.group()
        if "." not in s or hits_of(s, vals):
            continue
        h = hits_of(s, nvals)   # not below the named keys: next to them (the registry key names only part)?
        if h:
            near.update("%s#%s" % (rel(fp), pointer(q)) for fp, q in h)
        else:
            missing.append(s)
    where = "; ".join(sorted({"%s#%s" % (rel(fp), pointer(p)) for fp, p in found}))
    return dict(ok=not missing, where=where, missing=missing, near=sorted(near))


# ---------------------------------------------------------------- one macro
def check_entry(e, reg, idx, derived_ok):
    comment = e.get("comment") or ""
    prec = e["precision"] if isinstance(e["precision"], list) else [e["precision"]] * len(e["values"])
    shown = ["%.*f" % (p, v) for v, p in zip(e["values"], prec)]
    srcs = e.get("source") or []
    if not srcs:
        return "unresolved", "no source given", [], shown
    labels, problems, res = [], [], []
    for s in srcs:
        try:
            res.append(resolve(s))
            labels.append(res[-1][0])
        except ValueError as exc:
            problems.append(str(exc))
    if problems:
        return "mismatch", "; ".join(problems), [], shown
    pairs = list(zip(shown, res)) if len(res) == len(shown) else None
    for i, s in enumerate(shown):
        cand = [pairs[i][1]] if pairs else res
        if not any(agrees(float(s), r[2], decimals(s)) for r in cand):
            problems.append("shown %s != %s" % (s, " / ".join("%.6g" % r[2] for r in cand)))
    if any(s["file"] == DERIVED for s in srcs):
        precise = comment_numbers(comment)
        cand = [r[1] for r in res] + [r[2] for r in res] + derived_inputs(srcs)
        for c in precise:
            if not any(agrees(float(c), v, decimals(c)) for v in cand):
                problems.append("comment value %s is neither the derived value nor one of its inputs" % c)
        if not derived_ok:
            problems.append("%s is not up to date (run python numbers/derive_numbers.py)" % DERIVED)
        labels.append("derived by numbers/derive_numbers.py (up to date)" if derived_ok else "")
    else:
        precise = [c for c in comment_numbers(comment)
                   if any(agrees(abs(float(c)), abs(float(s)), decimals(s)) for s in shown)]
        for c in precise:
            cand = [r[1] if c[0] in "+-" else r[2] for r in res]
            if not any(agrees(float(c), v, decimals(c)) for v in cand):
                problems.append("comment value %s != %s" % (c, " / ".join("%.6g" % v for v in cand)))
    mags = [abs(r[1]) for r in res] + [abs(v) for v in derived_inputs(srcs)]
    for rid in dict.fromkeys(re.findall(r"R5-\d+", comment)):
        if rid not in reg:
            problems.append("registry id %s not in numbers_run05.csv" % rid)
            continue
        rnum = [m.group() for m in NUMRE.finditer(reg[rid].get("value") or "") if "." in m.group()]
        if not any(agrees(abs(float(s)), c, decimals(s)) for s in rnum for c in mags):
            problems.append("registry %s (%s) holds neither this number nor its inputs" % (rid, reg[rid].get("value")))
        t = trace_row(reg[rid], idx)
        if t is None:
            labels.append("registry %s (result file not in the package)" % rid)
        elif t["ok"]:
            labels.append("registry %s = %s (agrees%s)" % (rid, t["where"], "; partly at the neighbouring key(s) "
                                                           + ", ".join(t["near"]) if t["near"] else ""))
        else:
            problems.append("registry %s disagrees with %s: %s" % (rid, t["where"], ", ".join(t["missing"])))
    ev = "; ".join(x for x in labels if x) + ("" if precise else " [shown value only]")
    if problems:
        return "mismatch", ev + " | PROBLEMS: " + "; ".join(problems), precise, shown
    return "verified", ev, precise, shown


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    write, do_reg = "--write" in sys.argv, "--registry" in sys.argv
    doc = json.load(open(SRC, encoding="utf-8"))
    idx = file_index()
    reg = {}
    if os.path.exists(REG):
        with open(REG, encoding="utf-8", newline="") as f:
            reg = {r["id"]: r for r in csv.DictReader(f)}
    derived_ok = derive_numbers.up_to_date()
    counts, rows = {}, []
    for name in [it["macro"] for it in doc["layout"] if "macro" in it]:
        e = doc["numbers"][name]
        status, ev, precise, shown = check_entry(e, reg, idx, derived_ok)
        counts[status] = counts.get(status, 0) + 1
        e["check"] = {"status": status, "evidence": ev, "unrounded_in_comment": precise}
        rows.append((name, ",".join(shown), status, ev))
    for name, val, status, ev in rows:
        print("%-22s %-12s %-10s %s" % (name, val, status, ev[:170]))
    print("summary:", counts, "of", len(rows))
    rc = 1 if (counts.get("mismatch") or counts.get("unresolved")) else 0
    if do_reg:
        rcount = {}
        print("\nregistry %s:" % rel(REG))
        for rid, r in reg.items():
            t = trace_row(r, idx)
            st = ("not in package" if t is None else "DISAGREES" if not t["ok"] else
                  "agrees (partly at a neighbouring key)" if t["near"] else "agrees")
            rcount[st] = rcount.get(st, 0) + 1
            print("  %-7s %-14s %s%s%s" % (rid, st, "" if t is None else t["where"][:150],
                                           "" if t is None or t["ok"] else "  missing: " + ", ".join(t["missing"]),
                                           "" if t is None or not t["near"] else "  also at: " + ", ".join(t["near"])[:200]))
        print("registry summary:", rcount, "of", len(reg))
        rc = rc or (1 if rcount.get("DISAGREES") else 0)
    if write:
        doc["meta"]["check_summary"] = counts
        with open(SRC, "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print("check results written to", rel(SRC))
    return rc


if __name__ == "__main__":
    sys.exit(main())
