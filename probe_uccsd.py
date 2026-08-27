# probe_uccsd.py — kolko stoji UCCSD? (params, hlbka, cas/eval)
import time, numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
import config

d = np.load(config.INTEGRALS_NPZ)
ncas, e_core = int(d["ncas"]), float(d["e_core"])
n_particles = (int(d["n_alpha"]), int(d["n_beta"]))
mapper = JordanWignerMapper()

qop = mapper.map(fcidump_to_problem(FCIDump.from_file(config.FCIDUMP)).hamiltonian.second_q_op())
H = qop.to_matrix(sparse=True).tocsr()

hf  = HartreeFock(ncas, n_particles, mapper)
ucc = UCCSD(ncas, n_particles, mapper, initial_state=hf)
ansatz = transpile(ucc, basis_gates=config.BASIS_GATES, optimization_level=1)
print(f"params = {ansatz.num_parameters}   depth = {ansatz.depth()}", flush=True)

x0 = np.zeros(ansatz.num_parameters)          # UCCSD(0) = HF presne
t = time.time()
psi = Statevector(ansatz.assign_parameters(x0)).data
E = float((psi.conj() @ (H @ psi)).real) + e_core
print(f"[1 eval] {time.time()-t:.2f} s   E(HF) = {E:.6f} Eh  (ma byt -2518.540296)", flush=True)