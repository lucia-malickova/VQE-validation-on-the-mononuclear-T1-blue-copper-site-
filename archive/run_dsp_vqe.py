import numpy as np
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import minimize
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp

class DSP_VQE_Analytical:
    def __init__(self, H_sparse, ref_state, e_nuc, target_energy):
        self.H = H_sparse
        self.ref = ref_state
        self.e_nuc = e_nuc
        self.E_target = target_energy
        self.n_qubits = H_sparse.shape[0].bit_length() - 1
        
    def _exp_op(self, op_matrix, theta, state):
        """Aplikuj exp(θ·(T-T†)) na stav"""
        anti_herm = op_matrix - op_matrix.conj().T
        return expm_multiply(theta * anti_herm, state)
    
    def _build_state(self, op_matrices, theta):
        """Zostav stav sekvenčnou aplikáciou operátorov"""
        state = self.ref.copy()
        for i, (op, th) in enumerate(zip(op_matrices, theta)):
            state = self._exp_op(op, th, state)
        return state
    
    def _energy_and_grad(self, op_matrices, theta):
        """
        Vráť energiu a ANALYTICKÝ gradient.
        Parameter-shift rule pre unitárne operátory:
        ∂E/∂θᵢ = (E(θ + π/2·eᵢ) - E(θ - π/2·eᵢ)) / 2
        """
        state = self._build_state(op_matrices, theta)
        energy = (state.conj().T @ (self.H @ state)).real + self.e_nuc
        
        # Analytický gradient cez parameter-shift
        grad = np.zeros_like(theta)
        for i in range(len(theta)):
            # Shift dopredu
            theta_plus = theta.copy()
            theta_plus[i] += np.pi/2
            state_plus = self._build_state(op_matrices, theta_plus)
            e_plus = (state_plus.conj().T @ (self.H @ state_plus)).real + self.e_nuc
            
            # Shift dozadu
            theta_minus = theta.copy()
            theta_minus[i] -= np.pi/2
            state_minus = self._build_state(op_matrices, theta_minus)
            e_minus = (state_minus.conj().T @ (self.H @ state_minus)).real + self.e_nuc
            
            # Gradient
            grad[i] = (e_plus - e_minus) / 2
        
        return energy, grad
    
    def _initialize_from_ci(self, ci_coeffs, op_pool, max_ops=80):
        """
        Priama inicializácia z CI koeficientov.
        θᵢ = 2·arcsin(|cᵢ|/|c₀|) pre dominantné excitácie
        """
        # Nájdi dominantné determinanty
        c0 = ci_coeffs[0]  # referenčný determinant
        dominant = np.argsort(np.abs(ci_coeffs))[::-1][:max_ops]
        
        # Mapuj na operátory a inicializuj uhly
        selected_ops = []
        theta_init = []
        
        for idx in dominant[1:]:  # preskoč referenčný
            coeff = ci_coeffs[idx]
            if abs(coeff) < 0.01:  # prah
                continue
            
            # Nájdi zodpovedajúci operátor (zjednodušene)
            op_idx = min(idx, len(op_pool) - 1)
            selected_ops.append(op_pool[op_idx])
            
            # Uhol z pomeru koeficientov
            ratio = min(abs(coeff / c0), 0.99)
            theta = 2 * np.arcsin(ratio)
            theta_init.append(theta * np.sign(coeff.real))
        
        return selected_ops, np.array(theta_init)
    
    def run(self, ci_coeffs, op_pool, max_iter=200):
        """
        Hlavný beh s analytickým gradientom
        """
        # Inicializácia
        op_matrices, theta = self._initialize_from_ci(ci_coeffs, op_pool)
        
        print(f"Inicializovaných {len(theta)} operátorov")
        print(f"Počiatočná chyba: {abs(self._energy_and_grad(op_matrices, theta)[0] - self.E_target)*1000:.3f} mHa")
        
        # L-BFGS-B s analytickým gradientom
        def objective(theta_flat):
            e, _ = self._energy_and_grad(op_matrices, theta_flat)
            return e
        
        def gradient(theta_flat):
            _, g = self._energy_and_grad(op_matrices, theta_flat)
            return g
        
        result = minimize(
            objective,
            theta,
            method='L-BFGS-B',
            jac=gradient,  # ANALYTICKÝ gradient
            options={
                'maxiter': max_iter,
                'gtol': 1e-8,
                'ftol': 1e-12,
                'disp': True
            }
        )
        
        final_energy = result.fun
        final_error = abs(final_energy - self.E_target) * 1000
        
        print(f"\nFinálna energia: {final_energy:.8f} Ha")
        print(f"Finálna chyba: {final_error:.3f} mHa")
        
        if final_error < 1.6:
            print("✓ CHEMICKÁ PRESNOSŤ DOSIAHNUTÁ")
        else:
            print(f"✗ Chyba {final_error:.3f} mHa > 1.6 mHa")
        
        return final_energy, final_error, result.x

# POUŽITIE:
# 1. Načítaj H_sparse, ref_state, e_nuc z tvojho configu
# 2. Vytvor op_pool (spin-adaptované matice)
# 3. Načítaj ci_coeffs z CASSCF
# 
# vqe = DSP_VQE_Analytical(H_sparse, ref_state, e_nuc, E_FCI)
# energy, error, theta = vqe.run(ci_coeffs, op_pool)