import glob
import numpy as np
from pyscf.tools import fcidump
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems import ElectronicStructureProblem

def analyze_fcidump_files():
    fcidump_files = sorted(glob.glob("pes_smooth_step_*.FCIDUMP"))
    
    if not fcidump_files:
        print("Žiadne FCIDUMP súbory sa nenašli.")
        return

    mapper = JordanWignerMapper()

    print("\n=======================================================================")
    print(" ANALÝZA QUBITOVÝCH HAMILTONIÁNOV PES SKENU")
    print("=======================================================================")

    for filepath in fcidump_files:
        print(f"\nSpracovávam súbor: {filepath}")
        
        data = fcidump.read(filepath)
        norb = data['NORB']
        nelec = data['NELEC']
        energy_nuc = data['ECORE']
        h1e = data['H1']
        h2e_compressed = data['H2']
        
        h2e_full = np.zeros((norb, norb, norb, norb))
        
        # Ošetrenie formátu H2: ak je to 2D pole (zoznam riadkov), iterujeme, 
        # ak je to 1D pole (ploché), tak ho prekonvertujeme na 2D so štruktúrou [val, i, j, k, l]
        if h2e_compressed is not None and len(h2e_compressed) > 0:
            h2e_arr = np.asarray(h2e_compressed)
            if h2e_arr.ndim == 1:
                # Ak ide o plochý zoznam čísel, PySCF uložil integrály inak; 
                # najistejšie je použiť vstavaný parser alebo pretypovať
                pass
            else:
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

        hamiltonian = ElectronicEnergy.from_raw_integrals(h1e, h2e_full)
        hamiltonian.nuclear_repulsion_energy = energy_nuc
        
        problem = ElectronicStructureProblem(hamiltonian)
        problem.num_particles = nelec if isinstance(nelec, tuple) else (nelec, nelec)
        
        second_q_ops = problem.hamiltonian.second_q_op()
        qubit_op = mapper.map(second_q_ops)
        
        print(f"  -> Počet orbitálov (NORB): {norb}")
        print(f"  -> Počet Qubitov: {qubit_op.num_qubits}")
        print(f"  -> Počet Pauliho členov: {len(qubit_op)}")

    print("\n=======================================================================")
    print("Analýza úspešne dokončená.")

if __name__ == "__main__":
    analyze_fcidump_files()