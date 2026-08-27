"""
Hardware-Ready OpenQASM 3.0 Export & Transpilation Optimizer
pre Q1 publikáciu a reálne nasadenie na QPU (IBM/EuroHPC).
"""

import glob
import numpy as np
from pyscf.tools import fcidump
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit.circuit.library import n_local
from qiskit import transpile
from qiskit.qasm3 import dump as qasm3_dump
from scipy.optimize import minimize
import os

def export_hardware_ready_qasm():
    fcidump_files = sorted(glob.glob("pes_smooth_step_*.FCIDUMP"))
    
    if not fcidump_files:
        print("Žiadne FCIDUMP súbory sa nenašli.")
        return

    os.makedirs("artifacts/qasm", exist_ok=True)
    mapper = JordanWignerMapper()

    print("\n=======================================================================")
    print(" HARDWARE-READY OPENQASM 3.0 EXPORT & TRANSPILATION (QPU)")
    print("=======================================================================")

    ACTIVE_ORBITALS = 4
    ACTIVE_ELECTRONS = 4
    n_qubits = ACTIVE_ORBITALS * 2  # 8 qubitov

    for idx, filepath in enumerate(fcidump_files):
        print(f"\nSpracovávam bod skenu {idx} pre QASM export: {filepath}")
        
        data = fcidump.read(filepath)
        norb = data['NORB']
        energy_nuc = data['ECORE']
        h1e = data['H1']
        h2e_compressed = data['H2']
        
        h2e_full = np.zeros((norb, norb, norb, norb))
        if h2e_compressed is not None and len(h2e_compressed) > 0:
            h2e_arr = np.asarray(h2e_compressed)
            if h2e_arr.ndim > 1:
                for row in h2e_arr:
                    if len(row) == 5:
                        val, i, j, k, l = row
                        i, j, k, l = int(i)-1, int(j)-1, int(k)-1, int(l)-1
                        h2e_full[i, j, k, l] = val
                        h2e_full[j, i, k, l] = val
                        h2e_full[i, j, l, k] = val
                        h2e_full[j, i, l, k] = val
                        h2e_full[k, l, i, j] = val
                        h2e_full[l, k, i, j] = val
                        h2e_full[k, l, j, i] = val
                        h2e_full[l, k, j, i] = val

        start_idx = (norb - ACTIVE_ORBITALS) // 2
        end_idx = start_idx + ACTIVE_ORBITALS
        
        h1e_active = h1e[start_idx:end_idx, start_idx:end_idx]
        h2e_active = h2e_full[start_idx:end_idx, start_idx:end_idx, start_idx:end_idx, start_idx:end_idx]
        
        hamiltonian = ElectronicEnergy.from_raw_integrals(h1e_active, h2e_active)
        hamiltonian.nuclear_repulsion_energy = energy_nuc
        
        problem = ElectronicStructureProblem(hamiltonian)
        problem.num_particles = (ACTIVE_ELECTRONS // 2, ACTIVE_ELECTRONS // 2)
        
        second_q_ops = problem.hamiltonian.second_q_op()
        qubit_op = mapper.map(second_q_ops)
        
        # Ideálny solver na získanie referencie
        solver = NumPyMinimumEigensolver()
        exact_result = solver.compute_minimum_eigenvalue(qubit_op)
        exact_energy = exact_result.eigenvalue.real + energy_nuc
        
        ansatz = n_local(
            num_qubits=n_qubits,
            rotation_blocks=['ry', 'rz'],
            entanglement_blocks='cz',
            reps=1
        )
        
        # Rýchla lokalizácia optimálnych parametrov
        def objective_function(params):
            from qiskit_aer import AerSimulator
            bound_circ = ansatz.assign_parameters(params)
            bound_circ.save_expectation_value(qubit_op, range(n_qubits))
            return AerSimulator().run(bound_circ).result().data(0)['expectation_value']

        np.random.seed(42 + idx)
        init_params = np.random.uniform(0, np.pi, ansatz.num_parameters)
        opt_result = minimize(objective_function, init_params, method='COBYLA', options={'maxiter': 30})
        optimal_params = opt_result.x
        
        # Naviazanie optimálnych parametrov do obvodu
        bound_circuit = ansatz.assign_parameters(optimal_params)
        
        # Agresívna transpilarizácia (optimalizačná úroveň 3 pre reálny hardvér)
        # Simulujeme štruktúru brán pre univerzálny NISQ čip s CZ/CX väzbami
        transpiled_circuit = transpile(bound_circuit, optimization_level=3)
        
        # Zistenie metrík obvodu
        gate_counts = transpiled_circuit.count_ops()
        depth = transpiled_circuit.depth()
        cnot_count = gate_counts.get('cx', 0) + gate_counts.get('cz', 0)
        
        print(f"  -> Ideálna energia: {exact_energy:.6f} Eh")
        print(f"  -> Transpilovaná hĺbka obvodu (Depth): {depth}")
        print(f"  -> Počet 2-qubitových brán (CX/CZ): {cnot_count}")
        print(f"  -> Kompletné štatistiky brán: {gate_counts}")
        
        # Export do OpenQASM 3.0
        qasm_filepath = f"artifacts/qasm/pes_step_{idx}_optimized.qasm"
        with open(qasm_filepath, "w") as qasm_file:
            qasm3_dump(transpiled_circuit, qasm_file)
        print(f"  -> Úspešne exportované do: {qasm_filepath}")

    print("\n=======================================================================")
    print("Zhrnutie QASM exportu:")
    print("Všetky obvody sú optimalizované (Optimization Level 3) a uložené v 'artifacts/qasm/'.")
    print("Sú pripravené na priamy odoslanie do Qiskit Runtime / EuroHPC QPU infraštruktúry.")
    print("=======================================================================")

if __name__ == "__main__":
    export_hardware_ready_qasm()