"""STEP 1 (expensive, run once): build the active-space Hamiltonian.

PySCF ROHF -> AVAS active-space selection (Cu 3d & S 3p) -> CASSCF reference
-> reduced 1e/2e integrals, persisted to disk.
"""
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo, tools

import config
from geometry import t1_model


def select_active_space(mf, p):
    """Return (ncas, nelecas, mo) for CASSCF, from config."""
    if p["active"] == "manual":
        return p["cas_orb"], p["cas_elec"], mf.mo_coeff
    from pyscf.mcscf import avas
    ncas, nelecas, mo = avas.avas(mf, p["avas_labels"],
                                  threshold=p["avas_threshold"])
    return ncas, nelecas, mo


def main():
    p = config.profile()
    atoms, charge, spin = t1_model()
    mol = gto.M(atom=atoms, basis=p["basis"], charge=charge, spin=spin,
                verbose=4)                       # verbose: watch SCF converge

    # open-shell Cu(II) doublet -> ROHF
    mf = scf.ROHF(mol)
    mf.level_shift = 0.2
    mf.max_cycle = 300
    mf.kernel()
    if not mf.converged:
        print("[warn] ROHF not converged; retrying with second-order solver")
        mf = scf.newton(mf); mf.kernel()

    ncas, nelecas, mo = select_active_space(mf, p)
    print(f"[avas] {ncas} orbitals, {nelecas} electrons -> {2*ncas} qubits",
          flush=True)

    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.verbose = 4                # HEARTBEAT: prints macro-iteration energies
    mc.max_cycle_macro = 50       # fail-fast cap: no infinite churn
    mc.max_cycle_micro = 20
    mc.conv_tol = 1e-6
    mc.conv_tol_grad = 1e-4
    mc.fcisolver.conv_tol = 1e-7
    mc.fix_spin_(ss=0.75)         # lock the doublet (S=1/2): kills the #1
                                  # open-shell CASSCF oscillation / hang
    e_casscf = mc.kernel(mo)[0]
    if not mc.converged:
        print("[warn] CASSCF hit the macro cap without full convergence; "
              "using best orbitals found (fine as a feasibility reference).")
    n_ab = mc.nelecas                       # (n_alpha, n_beta)
    print(f"[cas] {ncas} orbitals, particles={n_ab} -> {2*ncas} qubits (JW)")
    print(f"[cas] CASSCF energy = {e_casscf:.8f} Eh")

    # reduced active-space integrals (valid even if CASSCF stopped at the cap)
    h1, e_core = mc.get_h1eff()
    h2 = ao2mo.restore(1, mc.get_h2eff(), ncas)

    e_dft = np.nan
    if p["run_dft"]:
        from pyscf import dft
        ks = dft.ROKS(mol); ks.xc = "b3lyp"; ks.level_shift = 0.2
        e_dft = ks.kernel()

    np.savez(config.INTEGRALS_NPZ, h1=h1, h2=h2, e_core=e_core, ncas=ncas,
             n_alpha=n_ab[0], n_beta=n_ab[1], e_casscf=e_casscf, e_dft=e_dft,
             basis=p["basis"])
    tools.fcidump.from_integrals(str(config.FCIDUMP), h1, h2, ncas, n_ab,
                                 nuc=e_core)
    print(f"[saved] {config.INTEGRALS_NPZ}  and  {config.FCIDUMP}")


if __name__ == "__main__":
    main()