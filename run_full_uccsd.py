# run_full_uccsd.py — Plný UCCSD ansatz pre garantované prelomenie spinovej pasce
import numpy as np, time
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import minimize
import config

def main():
    d = np.load(config.INTEGRALS_NPZ)
    e_core = float(d["e_core"]); ncas = int(d["ncas"])
    n_particles = (int(d["n_alpha"]), int(d["n_beta"])); N = sum(n_particles)
    
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.second_q.formats.fcidump import FCIDump
    from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
    from qiskit_nature.second_q.operators import FermionicOp
    from qiskit_nature.second_q.circuit.library import HartreeFock
    from qiskit.quantum_info import Statevector
    
    mapper = JordanWignerMapper()
    problem = fcidump_to_problem(FCIDump.from_file(config.FCIDUMP))
    qop = mapper.map(problem.hamiltonian.second_q_op())
    
    nq = getattr(qop, 'register_length', getattr(qop, 'num_modes', 2 * problem.num_spatial_orbitals))
    
    Hfull = qop.to_matrix(sparse=True).tocsr()
    idx = np.array([i for i in range(2**nq) if bin(i).count("1") == N], dtype=np.int64)
    Hsub = np.asarray(Hfull[idx][:, idx].todense())
    
    E0 = float(np.linalg.eigvalsh(Hsub)[0]) + e_core
    print(f"Exact FCI E0 = {E0:.6f} Eh (Hilbert space dimension: {len(idx)})", flush=True)
    
    hf_sv = Statevector(HartreeFock(ncas, n_particles, mapper)).data
    hf_i = int(np.argmax(np.abs(hf_sv)))
    occ = [i for i in range(nq) if (hf_i >> i) & 1]
    vir = [i for i in range(nq) if not ((hf_i >> i) & 1)]
    pos = int(np.where(idx == hf_i)[0][0])
    
    psi0 = np.zeros(len(idx), complex); psi0[pos] = 1.0
    print(f"E(HF) = {float((psi0.conj()@(Hsub@psi0)).real)+e_core:.6f} Eh", flush=True)
    
    def gen(op_dict):
        f = FermionicOp(op_dict, num_spin_orbitals=nq)
        m = mapper.map(f).to_matrix(sparse=True)
        return sp.csr_matrix(m[idx][:, idx])
        
    pool = []
    
    # S+D Pool
    for a in occ:
        for r in vir:
            A = gen({f"+_{r} -_{a}": 1.0, f"+_{a} -_{r}": -1.0})
            if A.nnz: pool.append(A)
            
    for ii in range(len(occ)):
        for jj in range(ii+1, len(occ)):
            for kk in range(len(vir)):
                for ll in range(kk+1, len(vir)):
                    a, b, r, s = occ[ii], occ[jj], vir[kk], vir[ll]
                    A = gen({f"+_{r} +_{s} -_{b} -_{a}": 1.0, f"+_{a} +_{b} -_{s} -_{r}": -1.0})
                    if A.nnz: pool.append(A)
                    
    print(f"Executing Full UCCSD with {len(pool)} operators...", flush=True)
    
    # 360 parametrov inicializovanych jemnym šumom
    theta_init = np.random.uniform(-0.01, 0.01, len(pool))
    
    def obj_and_grad(th):
        v_list = [psi0]
        for idx_op, val in enumerate(th):
            v_list.append(expm_multiply(val * pool[idx_op], v_list[-1]))
            
        psi = v_list[-1]
        E = float((psi.conj() @ (Hsub @ psi)).real)
        
        w = Hsub @ psi
        grads = np.zeros_like(th)
        for idx_op in reversed(range(len(th))):
            A = pool[idx_op]
            grads[idx_op] = 2.0 * np.real(np.vdot(w, A @ v_list[idx_op+1]))
            w = expm_multiply(-th[idx_op] * A, w)
            
        return E + e_core, grads

    t0 = time.time()
    
    # Priebežný výpis (Callback)
    iter_count = [0]
    def callback(xk):
        iter_count[0] += 1
        E, grads = obj_and_grad(xk)
        dev = (E - E0) * 1e3
        print(f"L-BFGS-B Iter {iter_count[0]:3d} | E={E:.6f} | dev={dev:9.3f} mEh | gmax={np.max(np.abs(grads)):.2e}", flush=True)

    # Masívna globálna optimalizácia
    res = minimize(obj_and_grad, theta_init, method="L-BFGS-B", jac=True, callback=callback, options={"gtol": 1e-6, "ftol": 1e-9, "maxiter": 1000})
    
    final_dev = (res.fun - E0) * 1e3
    print(f"\nOptimization Finished in {time.time()-t0:.0f}s!")
    print(f"Final Deviation from FCI: {final_dev:.3f} mEh")
    
    if final_dev < 1.6:
        print("SUCCESS: Chemical Accuracy (< 1.6 mEh) reached via Full UCCSD!", flush=True)

if __name__ == "__main__":
    main()
