import time
import numpy as np
import config
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit.circuit.library import EfficientSU2

print("Načítavam integrály...")
d = np.load(config.INTEGRALS_NPZ)
elec = ElectronicEnergy.from_raw_integrals(d['h1'], d['h2'])

print("Robím Jordan-Wigner mapovanie...")
qop = JordanWignerMapper().map(elec.second_q_op())
print(f"   -> Qubity: {qop.num_qubits} | Pauli členy: {len(qop)}")

print("Pripravujem ansatz...")
ans_template = EfficientSU2(qop.num_qubits, su2_gates=["ry"], entanglement="linear", reps=1)

# KLÚČOVÁ ZMENA: pridali sme .decompose(), aby Aer videl základné brány
ans = ans_template.assign_parameters(np.zeros(ans_template.num_parameters)).decompose()

print(f"   -> Parametre na optimalizáciu: {ans_template.num_parameters}")

print("Testujem Aer Estimator (1 eval)...")
aer = AerEstimator(options={'backend_options': {'method': 'statevector', 'device': 'CPU'}, 'default_precision': 0.0})
t = time.time()
res = aer.run([(ans, qop)]).result()
print(f"   -> Hotovo! Čas 1 eval: {time.time()-t:.2f} s")
print(f"   -> Vypočítaná energia (1. iterácia): {res[0].data.evs[0]:.6f} Hartree")