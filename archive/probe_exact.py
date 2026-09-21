# probe_exact.py — exact ground state v N=15 sektore (potvrdi H = CASSCF)
import numpy as np
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
import config

d = np.load(config.INTEGRALS_NPZ)
e_core = float(d["e_core"]); e_casscf = float(d["e_casscf"])
N = int(d["n_alpha"]) + int(d["n_beta"])           # spolu 15 castic
mapper = JordanWignerMapper()
qop = mapper.map(fcidump_to_problem(FCIDump.from_file(config.FCIDUMP)).hamiltonian.second_q_op())
nq = qop.num_qubits

H = qop.to_matrix(sparse=True).tocsr()
idx = np.array([i for i in range(2**nq) if bin(i).count("1") == N], dtype=np.int64)
print(f"N-sektor rozmer = {len(idx)}  (z 2^{nq})", flush=True)

Hsub = H[idx][:, idx].toarray()
E0 = float(np.linalg.eigvalsh(Hsub)[0]) + e_core
print(f"exact E0 (N={N})  = {E0:.6f} Eh")
print(f"E_casscf          = {e_casscf:.6f} Eh")
print(f"E0 - CASSCF       = {(E0 - e_casscf)*1e3:+.3f} mEh   (ma byt ~0)")