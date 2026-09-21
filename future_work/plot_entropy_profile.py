"""
Generovanie publikovateľného grafu Von Neumannovej entropie (Entanglement)
pre Q1 časopis (uloženie do PDF a PNG s vysokým rozlíšením).
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_entropy_profile():
    # Dáta získané z ADAPT-VQE simulácie
    steps = np.array([0, 1, 2, 3, 4])
    entropy = np.array([0.0000, 0.0006, 0.0122, 0.0030, 0.0000])
    
    # Nastavenie štýlu pre vedecké publikácie
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Vykreslenie krivky pre entropiu
    ax.plot(steps, entropy, marker='s', linestyle='-', color='#d62728', linewidth=2.5, markersize=8, label='ADAPT-VQE Entanglement (8 Qubits)')
    
    # Popisky a formátovanie
    ax.set_title('Quantum Entanglement Profile: Reaction Transition State', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Reaction Coordinate / Scan Step', fontsize=12, labelpad=10)
    ax.set_ylabel('Von Neumann Entropy [bits]', fontsize=12, labelpad=10)
    
    ax.set_xticks(steps)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=11)
    
    # Uloženie do súborov pre článok
    plt.tight_layout()
    plt.savefig('entropy_profile.pdf', format='pdf', dpi=300)
    plt.savefig('entropy_profile.png', format='png', dpi=300)
    
    print("\nGraf entropie úspešne vygenerovaný a uložený ako 'entropy_profile.pdf' a 'entropy_profile.png'.")
    plt.show()

if __name__ == "__main__":
    plot_entropy_profile()