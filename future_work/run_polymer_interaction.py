import numpy as np
from pyscf import gto, scf, mcscf
from pyscf.lib import logger
from pyscf.tools import fcidump

def build_copper_polymer_system():
    # Definícia geometrie: T1 medové centrum + modelový fragment väzby substrátu
    mol = gto.M(
        atom='''
            Cu  0.000000  0.000000  0.000000
            S   0.000000  2.152000  0.000000
            C   1.500000  3.200000  0.000000
            O   2.600000  2.800000  0.000000
        ''',
        basis='sto-3g',
        charge=0,  # Zmenené na 0, aby mal systém nepárny počet elektrónov (59)
        spin=1     # Open-shell doublet (N_alpha - N_beta = 1)
    )
    return mol

def run_restricted_open_shell(mol):
    mf = scf.ROHF(mol)
    mf.verbose = 4
    energy_rohf = mf.kernel()
    print(f"ROHF Energy: {energy_rohf:.6f} Eh")
    return mf

def setup_active_space_and_fcidump(mf, mol):
    from pyscf.tools import fcidump
    # Priamy export integrálov z ROHF do FCIDUMP pre Qiskit Nature
    fcidump.from_scf(mf, filename="active_polymer.FCIDUMP")
    print("FCIDUMP successfully generated from ROHF: active_polymer.FCIDUMP")

if __name__ == "__main__":
    mol = build_copper_polymer_system()
    mf = run_restricted_open_shell(mol)
    setup_active_space_and_fcidump(mf, mol)