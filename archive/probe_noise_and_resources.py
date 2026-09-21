# probe_noise_and_resources.py — item 5 (sum) + presne zdroje reps=1 aj reps=2
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ExcitationPreserving
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.primitives import BackendEstimatorV2
import config

d = np.load(config.INTEGRALS_NPZ)
ncas, e_core = int(d["ncas"]), float(d["e_core"])
n_particles = (int(d["n_alpha"]), int(d["n_beta"]))
N = n_particles[0] + n_particles[1]
mapper = JordanWignerMapper()
qop = mapper.map(fcidump_to_problem(FCIDump.from_file(config.FCIDUMP)).hamiltonian.second_q_op())
nq = qop.num_qubits
hf = HartreeFock(ncas, n_particles, mapper)

def build(reps):
    anz = ExcitationPreserving(nq, entanglement="linear", reps=reps)
    raw = QuantumCircuit(nq); raw.compose(hf, inplace=True); raw.compose(anz, inplace=True)
    return transpile(raw, basis_gates=config.BASIS_GATES, optimization_level=1)

print("=== RESOURCES ===", flush=True)
for reps in (1, 2):
    a = build(reps); ops = a.count_ops(); twoq = ops.get("cz",0)+ops.get("cx",0)
    print(f"reps={reps}: params={a.num_parameters}  depth={a.depth()}  2q_gates={twoq}", flush=True)

print("=== NOISE (reps=1, reprezentativne prevadzkove parametre) ===", flush=True)
a1 = build(1)
theta = 0.2 * np.random.default_rng(0).standard_normal(a1.num_parameters)

H = qop.to_matrix(sparse=True).tocsr()
idx = np.array([i for i in range(2**nq) if bin(i).count("1")==N], dtype=np.int64)
Hsub = H[idx][:,idx].toarray()
ps = Statevector(a1.assign_parameters(theta)).data[idx]
E_ideal = float((ps.conj() @ (Hsub @ ps)).real) + e_core
print(f"E_ideal = {E_ideal:.6f} Eh", flush=True)

nm = NoiseModel()
nm.add_all_qubit_quantum_error(depolarizing_error(config.NOISE_1Q,1), ["rz","sx","x"])
nm.add_all_qubit_quantum_error(depolarizing_error(config.NOISE_2Q,2), ["cz","cx"])
sim = AerSimulator(noise_model=nm)
est = BackendEstimatorV2(backend=sim); est.options.default_shots = 2048
E_noisy = float(est.run([(a1, qop, theta)]).result()[0].data.evs) + e_core
print(f"E_noisy = {E_noisy:.6f} Eh   (shots=2048, p1={config.NOISE_1Q}, p2={config.NOISE_2Q})", flush=True)
print(f"noise shift = {(E_noisy-E_ideal)*1e3:+.1f} mEh", flush=True)