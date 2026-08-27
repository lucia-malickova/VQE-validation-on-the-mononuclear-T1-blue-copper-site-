import numpy as np
from pyscf import gto, scf
from pyscf.tools import fcidump

def run_scan():
    # Plynulejší prechod v 5 krokoch bez obrovských skokov
    distances = np.linspace(1.4, 1.8, 5)
    energies = []
    dm = None  # prenos hustotnej matice (warm start)
    
    for i, d in enumerate(distances):
        print(f"\n--- Spúšťam krok {i+1}/5: Pozícia C = {d:.2f} Å ---")
        
        mol = gto.M(
            atom=f'''
                Cu  0.000000  0.000000  0.000000
                S   0.000000  2.152000  0.000000
                C   {d:.6f}  3.200000  0.000000
                O   2.600000  2.800000  0.000000
            ''',
            basis='sto-3g',
            charge=0,
            spin=1
        )
        
        mf = scf.ROHF(mol)
        mf.verbose = 0
        mf.level_shift = 0.2
        mf.max_cycle = 200
        
        # Spustenie s teplým štartom z predchádzajúceho kroku, ak existuje
        if dm is not None:
            e_rohf = mf.kernel(dm0=dm)
        else:
            e_rohf = mf.kernel()
            
        if mf.converged:
            print(f"Konvergovaná ROHF Energia: {e_rohf:.6f} Eh")
            energies.append((d, e_rohf))
            dm = mf.make_rdm1()  # uloženie hustotnej matice pre ďalší krok
            fcidump.from_scf(mf, filename=f"pes_smooth_step_{i}.FCIDUMP")
        else:
            print(f"Varovanie: Krok {i+1} nekonvergoval.")

    print("\n--- Kompletné Výsledky PES Scan ---")
    for d, e in energies:
        print(f"Pozícia C: {d:.2f} Å | Energia: {e:.6f} Eh")

if __name__ == "__main__":
    run_scan()