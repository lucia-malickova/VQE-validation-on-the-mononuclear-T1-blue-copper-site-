# probe_h_check.py — je Hamiltonian spravny? Over proti CASSCF/HF referencii.
import numpy as np
import scipy.sparse.linalg as sla
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
import config

d = np.load(config.INTEGRALS_NPZ)
ncas = int(d["ncas"]); n_particles = (int(d["n_alpha"]), int(d["n_beta"]))
e_casscf = float(d["e_casscf"])

mapper = JordanWignerMapper()

# --- H z FCIDUMP: jednoznacna konvencia, e_core uz zabudovane v konstante ---
problem = fcidump_to_problem(FCIDump.from_file(config.FCIDUMP))
qop = mapper.map(problem.hamiltonian.second_q_op())

# konstantny (identitny) clen - ci je e_core zabudovane
const = sum(c.real for pl, c in zip(qop.paulis, qop.coeffs) if not pl.x.any() and not pl.z.any())
print(f"identity coeff in qop = {const:.6f} Eh   (e_core z npz = {float(d['e_core']):.6f})")

H = qop.to_matrix(sparse=True).tocsr()

# HF stav v aktivnom priestore
psi_hf = Statevector(HartreeFock(ncas, n_particles, mapper)).data
E_hf = float((psi_hf.conj() @ (H @ psi_hf)).real)
print(f"<HF|H|HF>   = {E_hf:.6f} Eh   (ocakavam ~ -2518.7 az -2518.95)")

# presny zakladny stav aktivneho priestoru (najnizsie vlastne cislo)
print("... pocitam exact E0 (Lanczos, moze trvat par minut) ...", flush=True)
E0 = float(sla.eigsh(H, k=1, which="SA", return_eigenvectors=False)[0])
print(f"exact E0    = {E0:.6f} Eh")
print(f"E_casscf    = {e_casscf:.6f} Eh")
print(f"E0 - CASSCF = {(E0 - e_casscf)*1e3:+.3f} mEh   (ma byt ~ 0)")