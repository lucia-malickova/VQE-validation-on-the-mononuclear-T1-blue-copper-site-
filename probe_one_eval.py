# probe_one_eval.py — JEDEN eval, ziadny VQE loop
import time, numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import EfficientSU2
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
import config

p = config.profile()
d = np.load(config.INTEGRALS_NPZ)
ncas, e_core = int(d["ncas"]), float(d["e_core"])

elec = ElectronicEnergy.from_raw_integrals(d["h1"], d["h2"])
qop = JordanWignerMapper().map(elec.second_q_op())
print(f"qubits = {qop.num_qubits}   pauli terms = {len(qop)}", flush=True)

raw = EfficientSU2(2 * ncas, su2_gates=["ry"], entanglement="linear",
                   reps=p["hea_reps"])
qc = QuantumCircuit(raw.num_qubits); qc.compose(raw, inplace=True)
ansatz = transpile(qc, basis_gates=config.BASIS_GATES, optimization_level=1)
x0 = np.zeros(ansatz.num_parameters)
print(f"parameters = {ansatz.num_parameters}   depth = {ansatz.depth()}", flush=True)

est = AerEstimator(options={
    "backend_options": {"method": "statevector", "device": "CPU"},
    "default_precision": 0.0,          # presna expectation, bez shot noise
})
t = time.time()
res = est.run([(ansatz, qop, x0)]).result()
E = float(res[0].data.evs) + e_core
print(f"[1 eval] {time.time() - t:.2f} s   E(x0) = {E:.6f} Eh", flush=True)