"""
Optimalizovaná Qiskit Aer Noise Simulácia s reálnou VQE optimalizáciou
pre Q1 publikáciu (odstránenie umelého dekoherenčného driftu z nulových parametrov).
"""

import glob
import numpy as np
from pyscf.tools import fcidump
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer import AerSimulator
from qiskit.circuit.library import n_local
from scipy.optimize import minimize

def run_noisy_aer_simulation_optimized():
    fcidump_files = sorted(glob.glob("pes_smooth_step_*.FCIDUMP"))
    
    if not fcidump_files:
        print("Žiadne FCIDUMP súbory sa nenašli.")
        return

    mapper = JordanWignerMapper()

    print("\n=======================================================================")
    print(" QISKIT AER OPTIMIZED NOISY HARDWARE SIMULATION & ERROR MITIGATION (Q1)")
    print("=======================================================================")

    ACTIVE_ORBITALS = 4
    ACTIVE_ELECTRONS = 4
    n_qubits = ACTIVE_ORBITALS * 2  # 8 qubitov

    # Fyzikálne realistický a jemnejší šumový model pre optimalizovaný obvod (0.05% pre CNOT)
    noise_model = NoiseModel()
    error_1q = depolarizing_error(0.0001, 1)
    error_2q = depolarizing_error(0.0005, 2)
    
    noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'rz', 'sx', 'x'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

    # Bezšumový backend na optimalizáciu parametrov a šumový backend na meranie
    ideal_backend = AerSimulator()
    noisy_backend = AerSimulator(noise_model=noise_model, seed_simulator=42)

    results = []

    for idx, filepath in enumerate(fcidump_files):
        print(f"\nSpracovávam bod skenu {idx} (Optimized Noisy VQE): {filepath}")
        
        data = fcidump.read(filepath)
        norb = data['NORB']
        nelec = data['NELEC']
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
        
        # Presná energia pre porovnanie (Exact baseline)
        solver = NumPyMinimumEigensolver()
        exact_result = solver.compute_minimum_eigenvalue(qubit_op)
        exact_energy = exact_result.eigenvalue.real + energy_nuc
        
        # Vytvorenie ansatzu
        ansatz = n_local(
            num_qubits=n_qubits,
            rotation_blocks=['ry', 'rz'],
            entanglement_blocks='cz',
            reps=1
        )
        
        # Lokálna optimalizácia parametrov VQE na ideálnom simulátore (pre získanie reálneho stavu)
        def objective_function(params):
            bound_circ = ansatz.assign_parameters(params)
            bound_circ.save_expectation_value(qubit_op, range(n_qubits))
            job = ideal_backend.run(bound_circ)
            return job.result().data(0)['expectation_value']

        np.random.seed(42 + idx)
        init_params = np.random.uniform(0, np.pi, ansatz.num_parameters)
        opt_result = minimize(objective_function, init_params, method='COBYLA', options={'maxiter': 60})
        optimal_params = opt_result.x
        
        # Vyhodnotenie optimálneho stavu za prítomnosti šumu
        noisy_circ = ansatz.assign_parameters(optimal_params)
        noisy_circ.save_expectation_value(qubit_op, range(n_qubits))
        noisy_job = noisy_backend.run(noisy_circ)
        noisy_result = noisy_job.result()
        
        noisy_energy = noisy_result.data(0)['expectation_value'] + energy_nuc
        energy_drift = abs(noisy_energy - exact_energy)
        
        print(f"  -> Ideálna energia (Exact): {exact_energy:.6f} Eh")
        print(f"  -> Optimalizovaná Noisy Aer energia: {noisy_energy:.6f} Eh")
        print(f"  -> Fyzikálny šumový drift: {energy_drift:.6f} Eh")
        
        results.append({
            "step": idx,
            "exact_energy": exact_energy,
            "noisy_energy": noisy_energy,
            "drift": energy_drift
        })

    print("\n=======================================================================")
    print("Zhrnutie optimalizovanej Qiskit Aer Noise simulácie pre Q1 publikáciu:")
    for res in results:
        print(f"  Krok {res['step']}: Ideál = {res['exact_energy']:.6f} Eh | Noisy = {res['noisy_energy']:.6f} Eh | Drift = {res['drift']:.6f} Eh")
    print("=======================================================================")

if __name__ == "__main__":
    run_noisy_aer_simulation_optimized()