"""
Generovanie publikovateľného grafu energetického profilu PES skenu
pre Q1 časopis (uloženie do PDF a PNG s vysokým rozlíšením).
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_pes_profile():
    # Dáta získané z predchádzajúceho výpočtu pre 8-qubitový aktívny priestor
    steps = np.array([0, 1, 2, 3, 4])
    energies = np.array([99.364225, 102.473461, 105.258761, 105.871717, 107.419624])
    
    # Nastavenie štýlu pre vedecké publikácie (čistý a profesionálny vzhľad)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Vykreslenie krivky
    ax.plot(steps, energies, marker='o', linestyle='-', color='#1f77b4', linewidth=2.5, markersize=8, label='Active Space VQE (8 Qubits)')
    
    # Popisky a formátovanie
    ax.set_title('Potential Energy Surface (PES) Scan: Polymer Interaction', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Reaction Coordinate / Scan Step', fontsize=12, labelpad=10)
    ax.set_ylabel('Total Energy [Hartree]', fontsize=12, labelpad=10)
    
    ax.set_xticks(steps)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=11)
    
    # Uloženie do súborov pre článok
    plt.tight_layout()
    plt.savefig('pes_energy_profile.pdf', format='pdf', dpi=300)
    plt.savefig('pes_energy_profile.png', format='png', dpi=300)
    
    print("\nGraf úspešne vygenerovaný a uložený ako 'pes_energy_profile.pdf' a 'pes_energy_profile.png'.")
    plt.show()

if __name__ == "__main__":
    plot_pes_profile()