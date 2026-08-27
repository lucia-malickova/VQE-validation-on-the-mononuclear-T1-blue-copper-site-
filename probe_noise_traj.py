# probe_noise_traj.py — item 5 cez kvantove trajektorie (rychle)
import numpy as np, time
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ExcitationPreserving
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import config

NTRAJ = 40
d = np.load(config.INTEGRALS_NPZ)
ncas, e_core = int(d["ncas"]), float(d["e_core"])
n_particles = (int(d["n_alpha"]), int(d["n_beta"]))
mapper = JordanWignerMapper()
qop = mapper.map(fcidump_to_problem(FCIDump.from_file(config.FCIDUMP)).hamiltonian.second_q_op())
nq = qop.num_qubits
H = qop.to_matrix(sparse=True).tocsr()          # plna H (sum vystupuje z N-sektora)

hf  = HartreeFock(ncas, n_particles, mapper)
anz = ExcitationPreserving(nq, entanglement="linear", reps=1)
raw = QuantumCircuit(nq); raw.compose(hf, inplace=True); raw.compose(anz, inplace=True)
ansatz = transpile(raw, basis_gates=config.BASIS_GATES, optimization_level=1)
theta = 0.2 * np.random.default_rng(0).standard_normal(ansatz.num_parameters)
bound = ansatz.assign_parameters(theta)

# idealna energia (rovnaky theta)
psi = Statevector(bound).data
E_ideal = float((psi.conj() @ (H @ psi)).real) + e_core
print(f"E_ideal = {E_ideal:.6f} Eh", flush=True)

# sumove trajektorie
nm = NoiseModel()
nm.add_all_qubit_quantum_error(depolarizing_error(config.NOISE_1Q,1), ["rz","sx","x"])
nm.add_all_qubit_quantum_error(depolarizing_error(config.NOISE_2Q,2), ["cz","cx"])
sim = AerSimulator(method="statevector", noise_model=nm)
qc = bound.copy(); qc.save_statevector()
qc = transpile(qc, sim)

t = time.time()
result = sim.run([qc]*NTRAJ, shots=1).result()
Es = []
for i in range(NTRAJ):
    s = np.asarray(result.get_statevector(i).data)
    Es.append(float((s.conj() @ (H @ s)).real) + e_core)
Es = np.array(Es)
print(f"E_noisy = {Es.mean():.6f} +/- {Es.std()/np.sqrt(NTRAJ):.6f} Eh   "
      f"({NTRAJ} trajektorii, {time.time()-t:.0f}s)", flush=True)
print(f"noise shift = {(Es.mean()-E_ideal)*1e3:+.1f} mEh   (p1={config.NOISE_1Q}, p2={config.NOISE_2Q})", flush=True)