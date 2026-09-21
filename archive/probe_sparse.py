# probe_sparse.py — postav H raz, zmeraj, over spravnost. Izolovany test.
import time, numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper
import config

p = config.profile()
d = np.load(config.INTEGRALS_NPZ)
ncas, e_core = int(d["ncas"]), float(d["e_core"])

elec = ElectronicEnergy.from_raw_integrals(d["h1"], d["h2"])
qop = JordanWignerMapper().map(elec.second_q_op())
print(f"qubits={qop.num_qubits}  terms={len(qop)}", flush=True)

t = time.time()
H = qop.to_matrix(sparse=True).tocsr()
mem = (H.data.nbytes + H.indices.nbytes + H.indptr.nbytes) / 1e9
print(f"[build]  {time.time()-t:6.1f} s   nnz={H.nnz:,}   mem~{mem:.2f} GB", flush=True)

# statevector v x0 — presne ten stav, co ratal Aer (kvoli krizovej kontrole)
raw = EfficientSU2(2*ncas, su2_gates=["ry"], entanglement="linear", reps=p["hea_reps"])
qc = QuantumCircuit(raw.num_qubits); qc.compose(raw, inplace=True)
ansatz = transpile(qc, basis_gates=config.BASIS_GATES, optimization_level=1)
psi = Statevector(ansatz.assign_parameters(np.zeros(ansatz.num_parameters))).data

t = time.time()
E = complex(psi.conj() @ (H @ psi)).real + e_core
print(f"[matvec] {time.time()-t:6.3f} s   E(x0)={E:.6f} Eh", flush=True)