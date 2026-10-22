"""Print markdown convergence tables from methodA_results.json / methodB_results.json / slope_results.json."""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
KEYS = ["e=1,lam=-1", "e=1,lam=-0.11174", "e=1,lam=-0.03", "e=1,lam=-0.01003",
        "e=0.873,lam=-0.02286387", "e=1,lam=-0.05", "e=1,lam=-0.02"]


def main():
    A = json.load(open(os.path.join(ROOT, "methodA_results.json")))["runs"]
    B = json.load(open(os.path.join(ROOT, "methodB_results.json")))
    lines = []
    for key in KEYS:
        lines.append(f"\n#### {key}\n")
        lines.append("Method B (HO basis, complex rotation; 40-digit Rayleigh quotient)\n")
        lines.append("| Nr x Ns | beta, gamma | theta | phi | eps | Gamma |")
        lines.append("|---|---|---|---|---|---|")
        for r in [r for r in B["runs"] if f"e={r['e']:.6g},lam={r['lam']:.10g}" == key]:
            lines.append(f"| {r['Nr']} x {r['Ns']} | {r['beta']}, {r['gamma']} | {r['theta']:.4f} | "
                         f"{r['phi']:.4f} | {float(r['eps']):.15f} | {float(r['Gamma']):.12e} |")
        lines.append("\nMethod A (smooth ECS grid; Gamma_eig = -2 Im E, Gamma_flux = J(R)/N(R) at the two radii)\n")
        lines.append("| variant | unknowns | eps | Gamma_eig | Gamma_flux(R1) | Gamma_flux(R2) |")
        lines.append("|---|---|---|---|---|---|")
        rr = [r for r in A if r["key"] == key]
        def order(r):
            return (0 if "," not in r["tag"] else 1, -r["h"], r["tag"])
        for r in sorted(rr, key=order):
            f1, f2 = r["flux"][0], r["flux"][-1]
            lines.append(f"| {r['tag']} (R0={r['R0']}, theta={r['theta']:.2f}, Rmax={r['Rmax']:.1f}) | {r['n']} | "
                         f"{r['eps']:.13f} | {r['Gamma_eig']:.10e} | {f1['Gamma_flux']:.10e} (R={f1['R']:.2f}) | "
                         f"{f2['Gamma_flux']:.10e} (R={f2['R']:.2f}) |")
    sp = os.path.join(ROOT, "slope_results.json")
    if os.path.exists(sp):
        S = json.load(open(sp))
        lines.append("\n#### Optional: local slope -d ln Gamma / d x, x = |lambda|^(-1/2), e = 1 (Method B)\n")
        lines.append("| lambda | x | slope (5-pt) | slope (3-pt) | BO-WKB 2 dS/dx (E = eps(lambda)) | "
                     "BO-WKB 2S | sqrt(2) pi x |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in S["slopes"]:
            lines.append(f"| {r['lam']} | {r['x']:.4f} | {r['slope']:.5f} | {r['slope_3pt']:.5f} | "
                         f"{r['wkb_2dSdx']:.4f} | {r['wkb_2S']:.3f} | {r['sqrt2pi_x']:.3f} |")
        lines.append("\nDense scan (Method B, Nr x Ns = 120 x 60, theta = pi/8):\n")
        lines.append("| x | lambda | eps | Gamma |")
        lines.append("|---|---|---|---|")
        for r in S["dense"]:
            lines.append(f"| {r['x']:.2f} | {r['lam']:.6f} | {r['eps']:.12f} | {r['Gamma']:.10e} |")
    txt = "\n".join(lines) + "\n"
    open(os.path.join(ROOT, "convergence_tables.md"), "w").write(txt)
    head = os.path.join(ROOT, "METHOD_head.md")
    if os.path.exists(head):
        open(os.path.join(ROOT, "METHOD.md"), "w", encoding="utf-8").write(
            open(head, encoding="utf-8").read() + txt)
    print(txt)


if __name__ == "__main__":
    main()
