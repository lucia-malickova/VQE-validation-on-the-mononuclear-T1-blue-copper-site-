"""
Parameter Sensitivity & Loss Landscape Gradient Analysis
pre Q1 publikáciu a praktické mapovanie konvergencie VQE na emulátore.
"""

import glob
import numpy as np
from pyscf.tools import fcidump
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit.circuit.library import n_local
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

def run_landscape_sensitivity():
    fcidump_files = sorted(glob.glob("pes_smooth_step_*.FCIDUMP"))
    
    if not fcidump_files:
        print("Žiadne FCIDUMP súbory sa nenašli.")
        return

    mapper = JordanWignerMapper()
    backend = AerSimulator()

    print("\n=======================================================================")
    print(" LOSS LANDSCAPE & PARAMETER SENSITIVITY ANALYSIS (Q1 EMULATOR)")
    print("=======================================================================")

    ACTIVE_ORBITALS = 4
    ACTIVE_ELECTRONS = 4
    n_qubits = ACTIVE_ORBITALS * 2  # 8 qubitov

    for idx, filepath in enumerate(fcidump_files):
        print(f"\nSpracovávam bod skenu {idx} (Landscape Analysis): {filepath}")
        
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
        
        # Referencia
        solver = NumPyMinimumEigensolver()
        exact_result = solver.compute_minimum_eigenvalue(qubit_op)
        exact_energy = exact_result.eigenvalue.real + energy_nuc
        
        ansatz = n_local(
            num_qubits=n_qubits,
            rotation_blocks=['ry', 'rz'],
            entanglement_blocks='cz',
            reps=1
        )
        
        # Optimalizácia na zistenie optimálneho bodu v krajine
        def objective_function(params):
            bound_circ = ansatz.assign_parameters(params)
            bound_circ.save_expectation_value(qubit_op, range(n_qubits))
            return backend.run(bound_circ).result().data(0)['expectation_value']

        np.random.seed(42 + idx)
        init_params = np.random.uniform(0, np.pi, ansatz.num_parameters)
        opt_result = minimize(objective_function, init_params, method='COBYLA', options={'maxiter': 30})
        opt_params = opt_result.x
        opt_energy = opt_result.fun + energy_nuc
        
        # Parameter-Shift Rule / Finite Difference pre výpočet gradientov (citlivosti krajiny)
        eps = 1e-3
        gradients = []
        for p_idx in range(len(opt_params)):
            params_plus = opt_params.copy()
            params_minus = opt_params.copy()
            params_plus[p_idx] += eps
            params_minus[p_idx] -= eps
            
            e_plus = objective_function(params_plus)
            e_minus = objective_function(params_minus)
            
            grad = (e_plus - e_minus) / (2 * eps)
            gradients.append(grad)
            
        grad_norm = np.linalg.norm(gradients)
        max_grad = np.max(np.abs(gradients))
        
        print(f"  -> Ideálna energia: {exact_energy:.6f} Eh")
        print(f"  -> VQE dosiahnutá energia: {opt_energy:.6f} Eh")
        print(f"  -> Norma gradientu (Landscape Slope): {grad_norm:.6f}")
        print(f"  -> Maximálna citlivosť parametra: {max_grad:.6f}")
        
        if grad_norm < 1e-2:
            print(f"  -> [UPOZORNENIE] Indikovaná plochá oblasť / Barren Plateau riziko!")
        else:
            print(f"  -> [STABILNÉ] Krajina má dostatočný spád pre efektívnu optimalizáciu.")

    print("\n=======================================================================")
    print("Zhrnutie analýzy citlivosti krajiny:")
    print("Analýza úspešne preverila sklon energetickej krajiny a stabilitu gradientov.")
    print("=======================================================================")

if __name__ == "__main__":
    run_landscape_sensitivity()