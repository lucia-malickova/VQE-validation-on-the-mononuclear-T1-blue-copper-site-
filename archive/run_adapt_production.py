#!/usr/bin/env python3
"""
Produkčný ADAPT-VQE s SDT poolom pre CAS(15e, 9o) - Doublet sektor
Cieľ: < 1.6 mEh od FCI (E0 = -2518.995009 Eh) za < 50 operátorov

Optimalizované pre HPC:
- Sparse maticové operácie (scipy.sparse)
- expm_multiply pre rýchle maticové exponenciály
- Warm-start z CASSCF CI koeficientov
- Prioritný výber operátorov (Triples > Doubles > Singles)

Autor: Lucia Malickova
Dátum: 2026
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, eigsh
from scipy.optimize import minimize
import time
import json
from pathlib import Path

# ==================== KONFIGURÁCIA ====================
class Config:
    # Cesty k dátam
    INTEGRALS_NPZ = 'config.INTEGRALS_NPZ'  # tvoj existujúci súbor
    FCIDUMP = 'FCIDump'  # ak máš FCIDump
    
    # Systém
    N_ORBITALS = 9
    N_ALPHA = 8
    N_BETA = 7
    N_QUBITS = 18  # 2 * N_ORBITALS
    
    # FCI referencia
    E_FCI = -2518.995009
    E_CASSCF = -2518.989067
    
    # Chemická presnosť
    CHEMICAL_ACCURACY = 1.6e-3  # 1.6 mHa
    
    # ADAPT-VQE parametre
    MAX_OPERATORS = 50  # cieľový limit
    MAX_ITERATIONS = 100
    CONVERGENCE_GRADIENT = 1e-6
    ENERGY_THRESHOLD = 1.5e-3  # 1.5 mHa
    
    # Pool
    POOL_TYPE = 'SDT'  # Singles + Doubles + Triples
    K_ADD = 2  # operátory na iteráciu (menšie = presnejšie)
    
    # Warm start
    USE_WARM_START = True
    WARM_START_THRESHOLD = 0.01  # minimálny CI koeficient
    WARM_START_MAX = 10  # max warm-start operátorov
    
    # Optimalizátor
    OPTIMIZER = 'L-BFGS-B'
    MAX_OPT_ITER = 300
    GTOL = 1e-8
    
    # Výstup
    OUTPUT_DIR = 'adapt_results'
    SAVE_HISTORY = 'vqe_history_adapt.npz'
    VERBOSE = True

# ==================== NAČÍTANIE DÁT ====================
def load_hamiltonian(config):
    """Načítaj Hamiltonián z existujúcich dát"""
    print("="*70)
    print("NAČÍTAVAM HAMILTONIÁN")
    print("="*70)
        
    import config as real_config  # <-- Pridaj toto
    data = np.load(real_config.INTEGRALS_NPZ)  # <-- Použi real_config.INTEGRALS_NPZ
    
    # 1-elektrónové integrály (h1) - tvar (n_orbitals, n_orbitals)
    h1 = data['h1']  # alebo data['one_body'] podľa tvojho formátu
    
    # 2-elektrónové integrály (h2) - tvar (n_orbitals, n_orbitals, n_orbitals, n_orbitals)
    h2 = data['h2']  # alebo data['two_body']
    
    # Nuclear repulsion energy
    if 'nuclear_repulsion' in data:
        e_nuc = float(data['nuclear_repulsion'])
    elif 'e_nuc' in data:
        e_nuc = float(data['e_nuc'])
    else:
        e_nuc = 0.0
    
    n_orb = config.N_ORBITALS
    
    print(f"Orbitaly: {n_orb}")
    print(f"h1 shape: {h1.shape}")
    print(f"h2 shape: {h2.shape}")
    print(f"Nuclear repulsion: {e_nuc:.8f} Ha")
    
    return h1, h2, e_nuc

def build_spin_adapted_hamiltonian(h1, h2, e_nuc, config):
    """Zostav spin-adaptovaný Hamiltonián pre doublet sektor"""
    n_orb = config.N_ORBITALS
    n_alpha = config.N_ALPHA
    n_beta = config.N_BETA
    
    print("\nZostavujem spin-adaptovaný Hamiltonián (Doublet)...")
    
    # Vytvor fermiónový Hamiltonián v druhej kvantizácii
    # Použijeme Qiskit Nature pre konverziu
    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    
    electronic_energy = ElectronicEnergy.from_raw_integrals(h1, h2)
    electronic_energy.nuclear_repulsion_energy = e_nuc
    
    # Mapuj na qubity
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(electronic_energy.second_q_op())
    
    n_qubits = qubit_op.num_qubits
    n_terms = len(qubit_op)
    
    print(f"Qubity: {n_qubits}")
    print(f"Pauliho členy: {n_terms}")
    
    # Konvertuj na sparse maticu pre rýchle operácie
    hamiltonian_sparse = qubit_op.to_matrix(sparse=True)
    print(f"Sparse matica: {hamiltonian_sparse.shape}")
    print(f"NNZ: {hamiltonian_sparse.nnz}")
    
    return qubit_op, hamiltonian_sparse, e_nuc

# ==================== SDT POOL ====================
class SDTPool:
    """Singles + Doubles + Triples pool s prioritou"""
    
    def __init__(self, n_orb, n_alpha, n_beta):
        self.n_orb = n_orb
        self.n_alpha = n_alpha
        self.n_beta = n_beta
        self.n_qubits = 2 * n_orb
        
        # Generuj operátory
        self.operators = self._generate_operators()
        
        # Prioritizuj: Triples > Doubles > Singles
        self.priorities = self._assign_priorities()
        
        print(f"\nSDT Pool:")
        print(f"  Singles:  {sum(1 for p in self.priorities if p == 3)}")
        print(f"  Doubles:  {sum(1 for p in self.priorities if p == 2)}")
        print(f"  Triples:  {sum(1 for p in self.priorities if p == 1)}")
        print(f"  Celkovo:  {len(self.operators)}")
    
    def _generate_operators(self):
        """Generuj všetky excitačné operátory"""
        ops = []
        n_orb = self.n_orb
        
        # Singles: i → a (spin-up a spin-down)
        for i in range(n_orb):
            for a in range(n_orb):
                if i != a:
                    ops.append(('S', i, a, 0))  # alpha
                    ops.append(('S', i, a, 1))  # beta
        
        # Doubles: i,j → a,b
        for i in range(n_orb):
            for j in range(i+1, n_orb):
                for a in range(n_orb):
                    for b in range(a+1, n_orb):
                        # Rôzne spinové kombinácie
                        ops.append(('D', i, j, a, b, 0, 0))
                        ops.append(('D', i, j, a, b, 1, 1))
                        ops.append(('D', i, j, a, b, 0, 1))
        
        # Triples: i,j,k → a,b,c (pre 3-dierový systém kľúčové)
        # Pre 15 elektrónov v 9 orbitaloch: 6 obsadených, 3 virtuálne
        n_occ = self.n_alpha  # 8 obsadených orbitalov
        n_virt = n_orb - n_occ  # 1 virtuálny... uprav podľa potreby
        
        for i in range(n_occ):
            for j in range(i+1, n_occ):
                for k in range(j+1, n_occ):
                    for a in range(n_occ, n_orb):
                        for b in range(a+1, n_orb):
                            for c in range(b+1, n_orb):
                                # Rôzne spinové kombinácie
                                ops.append(('T', i, j, k, a, b, c, 0, 0, 0))
                                ops.append(('T', i, j, k, a, b, c, 1, 1, 1))
                                ops.append(('T', i, j, k, a, b, c, 0, 0, 1))
                                ops.append(('T', i, j, k, a, b, c, 0, 1, 1))
        
        return ops
    
    def _assign_priorities(self):
        """Priraď priority: Triples=1 (najdôležitejšie), Doubles=2, Singles=3"""
        priorities = []
        for op in self.operators:
            if op[0] == 'T':
                priorities.append(1)
            elif op[0] == 'D':
                priorities.append(2)
            else:  # 'S'
                priorities.append(3)
        return priorities
    
    def get_operator_matrix(self, op_idx, sparse=True):
        """Vytvor maticovú reprezentáciu operátora"""
        op = self.operators[op_idx]
        
        # Vytvor fermiónový operátor
        from qiskit_nature.second_q.operators import FermionicOp
        
        if op[0] == 'S':
            _, i, a, spin = op
            # a^†_a a_i
            label = f"+_{a} -_{i}"
            ferm_op = FermionicOp({label: 1.0}, num_spin_orbitals=self.n_qubits)
        elif op[0] == 'D':
            _, i, j, a, b, spin1, spin2 = op
            label = f"+_{a} +_{b} -_{j} -_{i}"
            ferm_op = FermionicOp({label: 1.0}, num_spin_orbitals=self.n_qubits)
        elif op[0] == 'T':
            _, i, j, k, a, b, c, spin1, spin2, spin3 = op
            label = f"+_{a} +_{b} +_{c} -_{k} -_{j} -_{i}"
            ferm_op = FermionicOp({label: 1.0}, num_spin_orbitals=self.n_qubits)
        
        # Mapuj na qubity
        from qiskit_nature.second_q.mappers import JordanWignerMapper
        mapper = JordanWignerMapper()
        qubit_op = mapper.map(ferm_op)
        
        # Konvertuj na sparse maticu
        if sparse:
            return qubit_op.to_matrix(sparse=True)
        else:
            return qubit_op.to_matrix()
    
    def get_anti_hermitian(self, op_idx):
        """Získaj anti-hermitovskú časť pre unitárnu evolúciu"""
        op_matrix = self.get_operator_matrix(op_idx)
        # T - T^† pre anti-hermitovský generátor
        anti_herm = op_matrix - op_matrix.conj().T
        return anti_herm

# ==================== WARM START ====================
def load_casscf_coefficients(config):
    """Načítaj CASSCF CI koeficienty pre warm-start"""
    print("\n" + "="*70)
    print("WARM-START Z CASSCF")
    print("="*70)
    
    # Skús načítať z NPZ ak existujú
    try:
        data = np.load(config.INTEGRALS_NPZ)
        if 'casscf_ci_coeffs' in data:
            ci_coeffs = data['casscf_ci_coeffs']
            print(f"Načítané CI koeficienty: {len(ci_coeffs)}")
        elif 'ci_coeffs' in data:
            ci_coeffs = data['ci_coeffs']
            print(f"Načítané CI koeficienty: {len(ci_coeffs)}")
        else:
            # Fallback: použij dominantné determinanty
            print("CI koeficienty nenájdené, používam heuristiku...")
            ci_coeffs = None
    except:
        ci_coeffs = None
    
    return ci_coeffs

def get_warm_start_operators(config, pool, ci_coeffs=None):
    """Vyber warm-start operátory z CI koeficientov"""
    if not config.USE_WARM_START:
        return [], []
    
    print("\nPripravujem warm-start operátory...")
    
    selected_ops = []
    selected_params = []
    
    if ci_coeffs is not None:
        # Nájdi dominantné determinanty
        dominant = []
        for i, coeff in enumerate(ci_coeffs):
            if abs(coeff) > config.WARM_START_THRESHOLD:
                dominant.append((i, coeff))
        
        # Zoraď podľa veľkosti
        dominant.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"Dominantné determinanty: {len(dominant)}")
        
        # Mapuj na pool operátory (heuristicky)
        for det_idx, coeff in dominant[:config.WARM_START_MAX]:
            # Nájdi zodpovedajúci operátor
            # Pre jednoduchosť: vyber prvých pár Triples s najväčším gradientom
            op_idx = det_idx % len(pool.operators)
            
            # Daj prednosť Triples
            while pool.operators[op_idx][0] != 'T' and op_idx < len(pool.operators):
                op_idx += 1
            
            if op_idx < len(pool.operators):
                selected_ops.append(op_idx)
                # Malý štartovací uhol
                selected_params.append(coeff * 0.1)
        
        print(f"Warm-start operátorov: {len(selected_ops)}")
    else:
        # Bez CI koeficientov: vyber top Triples podľa poradia
        triple_indices = [i for i, op in enumerate(pool.operators) if op[0] == 'T']
        selected_ops = triple_indices[:config.WARM_START_MAX]
        selected_params = np.random.uniform(-0.05, 0.05, len(selected_ops))
        print(f"Warm-start z {len(selected_ops)} Triples (bez CI koeficientov)")
    
    return selected_ops, selected_params

# ==================== ADAPT-VQE ====================
class ADAPTVQE_Sparse:
    """ADAPT-VQE s sparse maticovými operáciami"""
    
    def __init__(self, config, hamiltonian_sparse, pool, e_nuc=0.0):
        self.config = config
        self.H = hamiltonian_sparse
        self.pool = pool
        self.e_nuc = e_nuc
        self.n_qubits = pool.n_qubits
        
        # Referenčný stav (Doublet: 8 alpha, 7 beta)
        self.ref_state = self._build_reference_state()
        
        # História
        self.history = {
            'energies': [],
            'operators': [],
            'parameters': [],
            'gradients': [],
            'n_operators': [],
            'n_gates': [],
            'n_2q_gates': [],
            'errors_mha': []
        }
    
    def _build_reference_state(self):
        """Vytvor referenčný stav |HF⟩ pre doublet"""
        # 18 qubitov: prvých 9 orbitalov pre alpha, druhých 9 pre beta
        state = np.zeros(2**self.n_qubits, dtype=complex)
        
        # Naplň alpha elektróny (8) a beta elektróny (7)
        # Indexy: 0-8 pre alpha, 9-17 pre beta
        alpha_orbitals = list(range(8))  # prvých 8 alpha orbitalov obsadených
        beta_orbitals = list(range(9, 16))  # prvých 7 beta orbitalov obsadených
        
        # Vytvor bitstring
        bitstring = 0
        for orb in alpha_orbitals + beta_orbitals:
            bitstring |= (1 << orb)
        
        state[bitstring] = 1.0
        
        return state
    
    def _apply_operator_exponential(self, state, op_matrix, theta):
        """Aplikuj exp(θ * anti_hermitian) na stav"""
        anti_herm = op_matrix - op_matrix.conj().T
        # expm_multiply pre rýchlu maticovú exponenciálu
        return expm_multiply(theta * anti_herm, state)
    
    def _build_ansatz_state(self, operator_indices, params):
        """Zostav ansatz stav pre dané operátory a parametre"""
        state = self.ref_state.copy()
        
        for op_idx, theta in zip(operator_indices, params):
            op_matrix = self.pool.get_operator_matrix(op_idx)
            state = self._apply_operator_exponential(state, op_matrix, theta)
        
        return state
    
    def _compute_energy(self, state):
        """Vypočítaj energiu pre daný stav"""
        energy = state.conj().T @ (self.H @ state)
        return energy.real + self.e_nuc
    
    def _compute_gradient(self, operator_indices, params, new_op_idx):
        """Vypočítaj gradient pre nový operátor"""
        state = self._build_ansatz_state(operator_indices, params)
        
        # Gradient = ⟨ψ|[H, A]|ψ⟩ kde A je nový operátor
        op_matrix = self.pool.get_operator_matrix(new_op_idx)
        anti_herm = op_matrix - op_matrix.conj().T
        
        # Komutátor [H, A]
        commutator = self.H @ anti_herm - anti_herm @ self.H
        
        gradient = state.conj().T @ (commutator @ state)
        return abs(gradient.real)
    
    def _select_operators(self, current_ops, current_params, energy):
        """Vyber najlepšie operátory s prioritou"""
        print(f"\n  Výber operátorov (energia: {energy:.8f} Ha)")
        
        # Vypočítaj gradienty pre všetky operátory
        gradients = []
        
        # Prioritizuj: najprv Triples, potom Doubles, nakoniec Singles
        priority_order = [1, 2, 3]  # 1=Triples, 2=Doubles, 3=Singles
        
        for priority in priority_order:
            priority_ops = [
                i for i, p in enumerate(self.pool.priorities) 
                if p == priority and i not in current_ops
            ]
            
            for op_idx in priority_ops:
                grad = self._compute_gradient(current_ops, current_params, op_idx)
                gradients.append((grad, op_idx))
        
        # Zoraď podľa gradientu
        gradients.sort(reverse=True)
        
        # Vyber top-K
        selected = []
        for grad, op_idx in gradients[:self.config.K_ADD]:
            op_type = self.pool.operators[op_idx][0]
            selected.append(op_idx)
            print(f"    {op_type} op {op_idx}: gradient = {grad:.8f}")
        
        return selected
    
    def _optimize_parameters(self, operator_indices, initial_params):
        """Optimalizuj parametre pomocou L-BFGS-B"""
        
        def objective(theta):
            state = self._build_ansatz_state(operator_indices, theta)
            energy = self._compute_energy(state)
            return energy
        
        def gradient(theta):
            # Numerický gradient (pre L-BFGS-B)
            eps = 1e-8
            grad = np.zeros_like(theta)
            for i in range(len(theta)):
                theta_plus = theta.copy()
                theta_plus[i] += eps
                theta_minus = theta.copy()
                theta_minus[i] -= eps
                
                e_plus = objective(theta_plus)
                e_minus = objective(theta_minus)
                grad[i] = (e_plus - e_minus) / (2 * eps)
            
            return grad
        
        # Optimalizácia
        result = minimize(
            objective,
            initial_params,
            method=self.config.OPTIMIZER,
            jac=gradient,
            options={
                'maxiter': self.config.MAX_OPT_ITER,
                'gtol': self.config.GTOL,
                'disp': False
            }
        )
        
        return result.x, result.fun
    
    def run(self, warm_start_ops=None, warm_start_params=None):
        """Hlavný ADAPT-VQE cyklus"""
        print("\n" + "="*70)
        print("SPÚŠŤAM ADAPT-VQE (sparse, SDT pool)")
        print("="*70)
        
        # Inicializácia
        if warm_start_ops:
            operator_indices = list(warm_start_ops)
            params = list(warm_start_params)
        else:
            operator_indices = []
            params = []
        
        # Referenčná energia
        ref_energy = self._compute_energy(self.ref_state)
        error_mha = abs(ref_energy - self.config.E_FCI) * 1000
        
        print(f"\nReferenčná energia: {ref_energy:.8f} Ha")
        print(f"Chyba: {error_mha:.3f} mHa")
        
        # Ulož do histórie
        self._update_history(ref_energy, operator_indices, params, 0.0)
        
        # Hlavný cyklus
        for iteration in range(self.config.MAX_ITERATIONS):
            if len(operator_indices) >= self.config.MAX_OPERATORS:
                print(f"\nDosiahnutý limit operátorov ({self.config.MAX_OPERATORS})")
                break
            
            print(f"\n{'='*50}")
            print(f"Iterácia {iteration+1}")
            print(f"{'='*50}")
            
            # 1. Vypočítaj gradienty a vyber operátory
            new_ops = self._select_operators(
                operator_indices, params, self.history['energies'][-1]
            )
            
            # 2. Pridaj operátory
            operator_indices.extend(new_ops)
            params.extend([0.0] * len(new_ops))
            
            # 3. Optimalizuj parametre
            t_start = time.time()
            params, energy = self._optimize_parameters(operator_indices, params)
            t_opt = time.time() - t_start
            
            # 4. Vypočítaj gradient norm
            grad_norm = self._compute_total_gradient(operator_indices, params)
            
            # 5. Vypočítaj chybu
            error_mha = abs(energy - self.config.E_FCI) * 1000
            
            # 6. Ulož históriu
            self._update_history(energy, operator_indices, params, grad_norm)
            
            # 7. Výpis
            n_gates, n_2q = self._count_gates(operator_indices)
            print(f"\n  Energia: {energy:.8f} Ha")
            print(f"  Chyba: {error_mha:.3f} mHa")
            print(f"  Operátory: {len(operator_indices)}")
            print(f"  Brány: {n_gates} (2Q: {n_2q})")
            print(f"  Gradient norm: {grad_norm:.8f}")
            print(f"  Čas optimalizácie: {t_opt:.1f}s")
            
            # 8. Skontroluj chemickú presnosť
            if error_mha < self.config.CHEMICAL_ACCURACY:
                print(f"\n{'='*70}")
                print(f"✓ DOSIAHNUTÁ CHEMICKÁ PRESNOSŤ!")
                print(f"  Chyba: {error_mha:.3f} mHa < {self.config.CHEMICAL_ACCURACY*1000:.1f} mHa")
                print(f"  Operátory: {len(operator_indices)}")
                print(f"{'='*70}")
                break
            
            # 9. Skontroluj konvergenciu
            if grad_norm < self.config.CONVERGENCE_GRADIENT:
                print(f"\nKonvergovalo (gradient < {self.config.CONVERGENCE_GRADIENT})")
                break
        
        # Ulož finálne výsledky
        self._save_results()
        
        return self.history
    
    def _compute_total_gradient(self, operator_indices, params):
        """Vypočítaj celkovú gradient normu"""
        total_grad = 0.0
        for i, op_idx in enumerate(operator_indices):
            grad = self._compute_gradient(
                operator_indices[:i] + operator_indices[i+1:],
                params[:i] + params[i+1:],
                op_idx
            )
            total_grad += grad**2
        
        return np.sqrt(total_grad)
    
    def _count_gates(self, operator_indices):
        """Spočítaj brány v ansatze"""
        n_gates = 0
        n_2q = 0
        
        for op_idx in operator_indices:
            op_type = self.pool.operators[op_idx][0]
            if op_type == 'S':
                n_gates += 4
                n_2q += 2
            elif op_type == 'D':
                n_gates += 8
                n_2q += 4
            elif op_type == 'T':
                n_gates += 12
                n_2q += 6
        
        return n_gates, n_2q
    
    def _update_history(self, energy, operators, params, gradient):
        """Aktualizuj históriu"""
        error_mha = abs(energy - self.config.E_FCI) * 1000
        
        self.history['energies'].append(energy)
        self.history['operators'].append(list(operators))
        self.history['parameters'].append(list(params))
        self.history['gradients'].append(gradient)
        self.history['n_operators'].append(len(operators))
        
        n_gates, n_2q = self._count_gates(operators)
        self.history['n_gates'].append(n_gates)
        self.history['n_2q_gates'].append(n_2q)
        self.history['errors_mha'].append(error_mha)
    
    def _save_results(self):
        """Ulož výsledky vo formáte vqe_history_adapt.npz"""
        output_dir = Path(self.config.OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / self.config.SAVE_HISTORY
        
        np.savez(
            output_file,
            energies=np.array(self.history['energies']),
            n_operators=np.array(self.history['n_operators']),
            errors_mha=np.array(self.history['errors_mha']),
            gradients=np.array(self.history['gradients']),
            n_gates=np.array(self.history['n_gates']),
            n_2q_gates=np.array(self.history['n_2q_gates']),
            e_fci=self.config.E_FCI,
            chemical_accuracy=self.config.CHEMICAL_ACCURACY,
            pool_type=self.config.POOL_TYPE,
            n_qubits=self.n_qubits
        )
        
        print(f"\nVýsledky uložené do: {output_file}")
        
        # Ulož aj JSON s metadátami
        metadata = {
            'pool_type': self.config.POOL_TYPE,
            'n_qubits': self.n_qubits,
            'n_operators_final': len(self.history['operators'][-1]),
            'final_energy': self.history['energies'][-1],
            'final_error_mha': self.history['errors_mha'][-1],
            'fci_reference': self.config.E_FCI,
            'chemical_accuracy_achieved': self.history['errors_mha'][-1] < self.config.CHEMICAL_ACCURACY,
            'total_gates': self.history['n_gates'][-1],
            'total_2q_gates': self.history['n_2q_gates'][-1],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_file = output_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadáta uložené do: {metadata_file}")

# ==================== VIZUALIZÁCIA ====================
def plot_convergence(history, config):
    """Vytvor graf konvergencie"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Energia vs iterácia
    axes[0, 0].plot(range(len(history['energies'])), history['energies'], 'o-', linewidth=2)
    axes[0, 0].axhline(y=config.E_FCI, color='r', linestyle='--', label=f'FCI: {config.E_FCI:.6f}')
    axes[0, 0].axhline(y=config.E_CASSCF, color='g', linestyle='--', label=f'CASSCF: {config.E_CASSCF:.6f}')
    axes[0, 0].set_xlabel('Iterácia')
    axes[0, 0].set_ylabel('Energia (Ha)')
    axes[0, 0].set_title('Konvergencia ADAPT-VQE')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Chyba vs počet operátorov (logaritmická)
    axes[0, 1].semilogy(history['n_operators'], history['errors_mha'], 'o-', linewidth=2)
    axes[0, 1].axhline(y=config.CHEMICAL_ACCURACY*1000, color='g', linestyle='--', 
                       label=f'Chemická presnosť: {config.CHEMICAL_ACCURACY*1000:.1f} mHa')
    axes[0, 1].set_xlabel('Počet operátorov')
    axes[0, 1].set_ylabel('Chyba (mHa)')
    axes[0, 1].set_title('Presnosť vs počet operátorov')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Gradient vs iterácia
    axes[1, 0].semilogy(range(len(history['gradients'])), history['gradients'], 'o-', linewidth=2)
    axes[1, 0].set_xlabel('Iterácia')
    axes[1, 0].set_ylabel('Gradient norm')
    axes[1, 0].set_title('Gradient konvergencia')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Brány vs operátory
    axes[1, 1].plot(history['n_operators'], history['n_gates'], 'o-', label='Celkové brány')
    axes[1, 1].plot(history['n_operators'], history['n_2q_gates'], 's-', label='2Q brány')
    axes[1, 1].set_xlabel('Počet operátorov')
    axes[1, 1].set_ylabel('Počet brán')
    axes[1, 1].set_title('Náročnosť obvodu')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adapt_vqe_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nGraf uložený do: adapt_vqe_convergence.png")

# ==================== HLAVNÝ PROGRAM ====================
def main():
    # Načítaj konfiguráciu
    config = Config()
    
    # 1. Načítaj Hamiltonián
    h1, h2, e_nuc = load_hamiltonian(config)
    
    # 2. Zostav qubitový Hamiltonián
    qubit_op, hamiltonian_sparse, e_nuc = build_spin_adapted_hamiltonian(h1, h2, e_nuc, config)
    
    # 3. Vytvor SDT pool
    pool = SDTPool(config.N_ORBITALS, config.N_ALPHA, config.N_BETA)
    
    # 4. Warm start
    ci_coeffs = load_casscf_coefficients(config)
    warm_start_ops, warm_start_params = get_warm_start_operators(config, pool, ci_coeffs)
    
    # 5. Spusti ADAPT-VQE
    adapt = ADAPTVQE_Sparse(config, hamiltonian_sparse, pool, e_nuc)
    history = adapt.run(warm_start_ops, warm_start_params)
    
    # 6. Vizualizácia
    plot_convergence(history, config)
    
    # 7. Finálne zhrnutie
    print("\n" + "="*70)
    print("FINÁLNE ZHRNUTIE")
    print("="*70)
    
    final_energy = history['energies'][-1]
    final_error = history['errors_mha'][-1]
    n_ops = len(history['operators'][-1])
    n_gates = history['n_gates'][-1]
    n_2q = history['n_2q_gates'][-1]
    
    print(f"FCI referencia: {config.E_FCI:.8f} Ha")
    print(f"ADAPT-VQE energia: {final_energy:.8f} Ha")
    print(f"Chyba: {final_error:.3f} mHa")
    print(f"Počet operátorov: {n_ops}")
    print(f"Počet brán: {n_gates} (2Q: {n_2q})")
    
    if final_error < config.CHEMICAL_ACCURACY * 1000:
        print("\n✓✓✓ CHEMICKÁ PRESNOSŤ DOSIAHNUTÁ ✓✓✓")
        print(f"✓ Chyba {final_error:.3f} mHa < {config.CHEMICAL_ACCURACY*1000:.1f} mHa")
    else:
        print(f"\n✗ Chemická presnosť nedosiahnutá (chyba: {final_error:.3f} mHa)")
        print(f"  Odporúčam zvýšiť MAX_OPERATORS alebo upraviť warm-start")
    
    print(f"\nVýsledky uložené v: {config.OUTPUT_DIR}/{config.SAVE_HISTORY}")

if __name__ == "__main__":
    main()