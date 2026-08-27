# plot_report.py — Step 3: Porovnanie konvergencie reps=1 vs. reps=2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import config

def main():
    # Načítame dáta z oboch behov
    d1 = np.load("artifacts/prod/vqe_history.npz")
    d2 = np.load("artifacts/prod/vqe_history_reps2.npz")
    
    hist1 = d1["hist_ideal"]
    hist2 = d2["hist_ideal"]
    e0_exact = float(d1["e0_exact"])
    
    # Prevod na odchýlku v mEh od exaktnej hodnoty
    dev1 = (hist1 - e0_exact) * 1000
    dev2 = (hist2 - e0_exact) * 1000

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(dev1) + 1), dev1, "-", lw=2, label="VQE (HEA, reps=1, 53 params)")
    plt.plot(range(1, len(dev2) + 1), dev2, "-", lw=2, label="VQE (HEA, reps=2, 88 params)")
    plt.axhline(1.6, ls="--", color="r", alpha=0.7, label="Chemical accuracy (1.6 mEh)")
    
    plt.yscale("log")
    plt.xlabel("Energy evaluation", fontsize=12)
    plt.ylabel("Deviation from exact E₀ (mEh)", fontsize=12)
    plt.title("VQE Convergence Comparison: T1 Blue-Copper Active Site (18 Qubits)", fontsize=13, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Uloženie priamo do artifacts/prod/
    output_png = Path("artifacts/prod/vqe_convergence_comparison.png")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"\n[saved] Porovnávací graf úspešne uložený ako: {output_png}")
    plt.show()

if __name__ == "__main__":
    main()