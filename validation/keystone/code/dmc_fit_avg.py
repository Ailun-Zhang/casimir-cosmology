# keystone-blind (T3): nominal dE(r): tau-average for r>=1 (no resolved time-step trend; errors inflated by scatter),
# linear tau-extrapolation at r=0.5 (clear trend from S^3 step sqrt(tau)/r), exact r=0 from Luscher eps2.
import os, sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
d = json.load(open(os.path.join(RES, "dE_fit.json")))
r_out, dE, err = [0.0], [d["dE"][0]], [0.0]
for rs, e in sorted(d["fixr"].items(), key=lambda kv: float(kv[0])):
    r = float(rs); raw = np.array(e["raw"])      # tau, E, err
    if r < 0.75:
        E0, s = e["E0"], e["err"]
    else:
        w = 1/raw[:, 2]**2; E0 = float((w*raw[:, 1]).sum()/w.sum())
        chi2 = float((w*(raw[:, 1] - E0)**2).sum()); s = float(np.sqrt(1/w.sum())*np.sqrt(max(chi2/(len(raw) - 1), 1.0)))
    r_out.append(r); dE.append(E0 - e["EG"]); err.append(s)
    print("r=%.1f  dE=%.5f +- %.5f" % (r, E0 - e["EG"], s))
json.dump(dict(r=r_out, dE=dE, dE_err=err, tail_power=4.0, method="tau-average r>=1 (err scaled by sqrt(chi2/dof)), tau-extrapolation r=0.5, r=0 from eps2"),
          open(os.path.join(RES, "dE_fit_avg.json"), "w"), indent=1)
