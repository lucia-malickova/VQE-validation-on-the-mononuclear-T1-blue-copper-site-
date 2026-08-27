# which_sulfur.py
import numpy as np
from pyscf import gto
from geometry import t1_model
import config

# Načítanie modelu
atoms, charge, spin = t1_model()
# Použijeme bázu z konfigurácie
mol = gto.M(atom=atoms, basis=config.profile()["basis"],
            charge=charge, spin=spin)

# Nájdenie indexov Cu a S
cu = [i for i in range(mol.natm) if mol.atom_symbol(i) == "Cu"]
s  = [i for i in range(mol.natm) if mol.atom_symbol(i) == "S"]
print("Cu atom idx:", cu, "   S atom idx:", s)

# Výpočet vzdialeností
coords = mol.atom_coords()          # Bohr
BOHR = 0.529177
for si in s:
    dist = np.linalg.norm(coords[cu[0]] - coords[si]) * BOHR
    print(f"  Cu - S(atom {si}) = {dist:.3f} A")