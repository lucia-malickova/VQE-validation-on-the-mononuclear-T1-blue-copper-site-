import matplotlib.pyplot as plt

# Dáta priamo extrahované z tvojho logu (0 sekúnd výpočtového času)
iters = [0, 1, 2, 3, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 
         120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 362]
errors = [454.7, 381.9, 344.3, 313.1, 154.5, 139.3, 116.8, 113.9, 111.5, 110.5, 109.3, 109.1, 108.3, 107.7, 105.5, 
          94.2, 72.8, 56.7, 44.7, 37.5, 32.4, 28.4, 26.1, 24.6, 23.2, 22.3, 21.5, 20.8]

plt.figure(figsize=(8, 5))
plt.plot(iters, errors, marker='o', markersize=4, linestyle='-', color='#1f77b4', linewidth=2, label='ADAPT-VQE Convergence')

# Zvýraznenie štartu a konca
plt.scatter([0], [454.7], color='red', zorder=5, label='Hartree-Fock Start (454.7 mEh)')
plt.scatter([362], [20.8], color='green', zorder=5, label='Iter 362 (20.8 mEh, >95% error removed)')

plt.axhline(y=1.6, color='black', linestyle='--', label='Chemical Accuracy (1.6 mEh)')

plt.yscale('log')
plt.xlabel('ADAPT-VQE Iterations (Greedy Operator Addition)')
plt.ylabel('Energy Error vs FCI (mEh) - Log Scale')
plt.title('ADAPT-VQE Convergence Profile (Singles & Doubles Pool)')
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend()
plt.tight_layout()

# Uloží graf okamžite
plt.savefig('fig_adapt_convergence.png', dpi=300)
print("Graf bol úspešne uložený ako 'fig_adapt_convergence.png'!")