import matplotlib.pyplot as plt

# Dáta priamo z tvojho úspešného behu s globálnou re-optimalizáciou
iters = [0, 5, 10, 20, 30, 40, 50, 51, 55, 60, 70, 76]
errors = [454.7, 143.1, 113.4, 110.0, 108.3, 106.0, 105.3, 12.3, 5.8, 4.25, 4.15, 4.11]

plt.figure(figsize=(8, 5))
plt.plot(iters, errors, marker='o', markersize=5, linestyle='-', color='#2ca02c', linewidth=2, label='Global ADAPT-VQE')

# Zvýraznenie kľúčových bodov
plt.scatter([0], [454.7], color='red', zorder=5, label='Hartree-Fock Start (454.7 mEh)')
plt.scatter([51], [12.3], color='orange', zorder=5, label='Correlation Barrier Broken (Iter 51)')
plt.scatter([76], [4.11], color='blue', zorder=5, label='Highly Compressed Ansatz (Iter 76, 4.1 mEh)')

plt.axhline(y=1.6, color='black', linestyle='--', label='Chemical Accuracy (1.6 mEh)')

plt.yscale('log')
plt.xlabel('Number of Operators in Ansatz')
plt.ylabel('Energy Error vs FCI (mEh) - Log Scale')
plt.title('Ansatz Compression: ADAPT-VQE Convergence Profile')
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend()
plt.tight_layout()

plt.savefig('fig_adapt_success.png', dpi=300)
print("Graf ulozeny ako 'fig_adapt_success.png'")
