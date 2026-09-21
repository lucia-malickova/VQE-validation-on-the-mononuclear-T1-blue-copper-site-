# dsp_analytical.py — DSP-VQE s analytickým parameter-shift gradientom
import numpy as np
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import minimize

class DSP_VQE_Analytical:
    def __init__(self, H_sparse, ref_state, e_nuc, target_energy):
        self.H = H_sparse
        self.ref = ref_state
        self.e_nuc = e_nuc
        self.E_target = target_energy
            
    def _exp_op(self, op_matrix, theta, state):
        anti_herm = op_matrix - op_matrix.conj().T
        return expm_multiply(theta * anti_herm, state)
        
    def _build_state(self, op_matrices, theta):
        state = self.ref.copy()
        for op, th in zip(op_matrices, theta):
            state = self._exp_op(op, th, state)
        return state
        
    def _energy_and_grad(self, op_matrices, theta):
        state = self._build_state(op_matrices, theta)
        energy = (state.conj().T @ (self.H @ state)).real + self.e_nuc
                
        grad = np.zeros_like(theta)
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += np.pi/2
            state_plus = self._build_state(op_matrices, theta_plus)
            e_plus = (state_plus.conj().T @ (self.H @ state_plus)).real + self.e_nuc
                        
            theta_minus = theta.copy()
            theta_minus[i] -= np.pi/2
            state_minus = self._build_state(op_matrices, theta_minus)
            e_minus = (state_minus.conj().T @ (self.H @ state_minus)).real + self.e_nuc
                        
            grad[i] = (e_plus - e_minus) / 2
                
        return energy, grad

    def run(self, ci_coeffs, op_pool, max_iter=100):
        c0 = ci_coeffs[0]
        dominant = np.argsort(np.abs(ci_coeffs))[::-1][:len(op_pool)]
        
        selected_ops = []
        theta_init = []
        for idx in dominant[1:]:
            if idx >= len(ci_coeffs): continue
            coeff = ci_coeffs[idx]
            if abs(coeff) < 0.01: continue
            op_idx = min(idx, len(op_pool) - 1)
            selected_ops.append(op_pool[op_idx])
            ratio = min(abs(coeff / c0), 0.99)
            theta_init.append(2 * np.arcsin(ratio) * np.sign(coeff.real))
            
        theta = np.array(theta_init)
        print(f"Spúšťam optimalizáciu s {len(theta)} operátormi (Analytický gradient)...", flush=True)

        def objective(th):
            e, _ = self._energy_and_grad(selected_ops, th)
            return e

        def gradient(th):
            _, g = self._energy_and_grad(selected_ops, th)
            return g

        result = minimize(objective, theta, method='L-BFGS-B', jac=gradient, options={'maxiter': max_iter, 'gtol': 1e-8, 'disp': True})
        final_error = abs(result.fun - self.E_target) * 1000
        print(f"Finálna chyba: {final_error:.3f} mHa", flush=True)
        return result.fun, final_error, result.x
