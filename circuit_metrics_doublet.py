#!/usr/bin/env python3
"""Build + transpile the exact spin-pure ADAPT-VQE circuit (from
adapt_doublet_spin_pure.json, produced by run_adapt_doublet.py) and report
CNOT count + depth. Requires: pip install qiskit qiskit-nature

Reproduces the ADAPT-VQE row of Table 2 in arXiv:2609.20439v2.
"""
import json
import numpy as np
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate

from spin_fci import read_fcidump, excitation_specs_from_hf

with open("adapt_doublet_spin_pure.json") as f:
    res = json.load(f)

fcidump_path = res["fcidump"]
selected = res["selected_indices"]
theta = res["theta"]
print(f"Loaded {len(selected)} operators, final dev={res['history'][-1][2]:.4f} mEh")

data = read_fcidump(fcidump_path)
nalpha = (data.nelec + data.ms2) // 2
nbeta = data.nelec - nalpha
norb = data.norb
nso = 2 * norb

specs = excitation_specs_from_hf(norb, nalpha, nbeta, max_rank=3)
assert len(specs) == 323, f"pool mismatch: got {len(specs)}"

mapper = JordanWignerMapper()
hf_sv = Statevector(HartreeFock(norb, (nalpha, nbeta), mapper)).data
hf_i = int(np.argmax(np.abs(hf_sv)))
occ = [i for i in range(nso) if (hf_i >> i) & 1]

qc = QuantumCircuit(nso)
for o in occ:
    qc.x(o)

for k, th in zip(selected, theta):
    rem, add = specs[k]
    create_fwd = " ".join(f"+_{p}" for p in add)
    annih_fwd = " ".join(f"-_{q}" for q in reversed(rem))
    create_bwd = " ".join(f"+_{p}" for p in rem)
    annih_bwd = " ".join(f"-_{q}" for q in reversed(add))
    fdict = {f"{create_fwd} {annih_fwd}": 1.0, f"{create_bwd} {annih_bwd}": -1.0}
    fop = FermionicOp(fdict, num_spin_orbitals=nso)
    qubit_op = mapper.map(fop)
    H_herm = (-1j) * qubit_op
    qc.append(PauliEvolutionGate(H_herm, time=-th), range(nso))

print(f"Pre-transpile: size={qc.size()}, ops={dict(qc.count_ops())}")

transpiled = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=3)
gc = transpiled.count_ops()
cnot = gc.get("cx", 0) + gc.get("cz", 0)
depth = transpiled.depth()

print("\n=== RESULT ===")
print(f"Qubits: {nso}")
print(f"Operators: {len(selected)}")
print(f"Transpiled depth: {depth}")
print(f"CNOT count: {cnot}")
print(f"All gates: {dict(gc)}")

with open("circuit_metrics_doublet.json", "w") as f:
    json.dump({"n_qubits": nso, "n_operators": len(selected),
                "transpiled_depth": depth, "cnot_count": cnot,
                "gate_counts": dict(gc)}, f, indent=2)
print("saved circuit_metrics_doublet.json")
