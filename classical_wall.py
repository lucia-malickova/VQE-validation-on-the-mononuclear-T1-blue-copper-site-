#!/usr/bin/env python3
"""Classical FCI 'wall': exact cost of solving the active space classically
grows exponentially with its size, while VQE cost grows ~linearly in qubits.
This is machine-independent combinatorics -> a rigorous 'why we need QC' figure.
"""
import numpy as np
from math import comb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# active-space sizes: M spatial orbitals at half filling (the strongly-correlated
# regime that multi-copper / Cu-O2 clusters approach) -> N=M electrons, na=nb=M/2
Ms = [4, 6, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]
rows = []
for M in Ms:
    na = M // 2; nb = M - na           # half filling
    dim = comb(M, na) * comb(M, nb)    # FCI determinant dimension
    qubits = 2 * M
    mem_bytes = dim * 8                 # one double-precision CI vector (lower bound;
                                        # Davidson needs several such vectors)
    rows.append((M, qubits, na + nb, dim, mem_bytes))

print(f"{'orbitals':>8} {'qubits':>7} {'elec':>5} {'FCI dim':>18} {'CI-vector mem':>15}")
def human(b):
    for u in ["B","KB","MB","GB","TB","PB","EB","ZB"]:
        if b < 1024: return f"{b:.1f} {u}"
        b /= 1024
    return f"{b:.1f} YB"
for M, q, e, dim, mem in rows:
    print(f"{M:>8} {q:>7} {e:>5} {dim:>18.3e} {human(mem):>15}")

# ---- figure ----
q   = np.array([r[1] for r in rows])
mem = np.array([r[4] for r in rows], dtype=float) / 1024**4   # TB

NODE_TB   = 0.5     # ~ one high-memory HPC node
LEO_TB    = 100.0   # ~ generous distributed (Leonardo-scale) aggregate for one CI vector

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.semilogy(q, mem, "-o", color="#1b4f72", lw=2.2, ms=5,
            label="Classical exact FCI (memory for one CI vector)")
ax.axhline(NODE_TB, color="#c0392b", ls="--", lw=1.4)
ax.text(q[0], NODE_TB*1.4, "single HPC node (~0.5 TB)", color="#c0392b", fontsize=8)
ax.axhline(LEO_TB, color="#8e44ad", ls="--", lw=1.4)
ax.text(q[0], LEO_TB*1.4, "large distributed (~100 TB)", color="#8e44ad", fontsize=8)

# regions
ax.axvspan(16, 20, color="#2e86c1", alpha=0.12)
ax.text(18, mem.min()*3, "this work\n(18 q, exact\nclassical check)",
        ha="center", va="bottom", fontsize=8, color="#1b4f72")
ax.axvspan(50, 70, color="#27ae60", alpha=0.12)
ax.text(60, 1e6, "multi-copper\ntarget (50–70 q)\nclassically\nimpossible",
        ha="center", va="center", fontsize=8, color="#1e7d34")

ax.set_xlabel("qubits  (= 2 × active orbitals)")
ax.set_ylabel("classical FCI memory  (TB, log scale)")
ax.set_title("The classical wall: exact solution cost explodes with active-space size")
ax.set_xlim(0, 74)
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(loc="upper left", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig("fig_classical_wall.png", dpi=150)
print("\n[saved] fig_classical_wall.png")