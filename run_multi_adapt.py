# run_multi_adapt.py — Multi-Operator ADAPT-VQE pre okamzite prelomenie symetrie
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
    print(f"Exact FCI E0 = {E0:.6f} Eh", flush=True)
    
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
        
    pool, labels = [], []
    for a in occ:
        for r in vir:
            A = gen({f"+_{r} -_{a}": 1.0, f"+_{a} -_{r}": -1.0})
            if A.nnz: pool.append(A); labels.append(("S", a, r))
            
    for ii in range(len(occ)):
        for jj in range(ii+1, len(occ)):
            for kk in range(len(vir)):
                for ll in range(kk+1, len(vir)):
                    a, b, r, s = occ[ii], occ[jj], vir[kk], vir[ll]
                    A = gen({f"+_{r} +_{s} -_{b} -_{a}": 1.0, f"+_{a} +_{b} -_{s} -_{r}": -1.0})
                    if A.nnz: pool.append(A); labels.append(("D", a, b, r, s))
                    
    print(f"Pool = {len(pool)} operatorov", flush=True)
    
    sel = []
    theta = np.zeros(0)
    
    def obj_and_grad(th):
        v_list = [psi0]
        for i, t in enumerate(th):
            v_list.append(expm_multiply(t * pool[sel[i]], v_list[-1]))
            
        psi = v_list[-1]
        E = float((psi.conj() @ (Hsub @ psi)).real)
        
        w = Hsub @ psi
        grads = np.zeros_like(th)
        for i in reversed(range(len(th))):
            A = pool[sel[i]]
            grads[i] = 2.0 * np.real(np.vdot(w, A @ v_list[i+1]))
            w = expm_multiply(-th[i] * A, w)
            
        return E + e_core, grads

    CHEM = 1.6; MAX_ITERS = 30; K_ADD = 4
    t0 = time.time()
    
    for it in range(1, MAX_ITERS+1):
        v = psi0.copy()
        for t, k in zip(theta, sel):
            v = expm_multiply(t * pool[k], v)
            
        u = Hsub @ v
        pool_grads = np.array([abs(2.0*np.real(np.vdot(u, pool[k] @ v))) for k in range(len(pool))])
        
        # MULTI-ADAPT: Vyberieme 4 najlepšie operátory naraz
        top_indices = np.argsort(pool_grads)[-K_ADD:][::-1]
        gmax = float(pool_grads[top_indices[0]])
        
        if gmax < 1e-5:
            print(f"[stop] max|grad| = {gmax:.2e}", flush=True); break
            
        for kbest in top_indices:
            sel.append(kbest)
            
        # Zlom symetrie: malé počiatočné hodnoty uhlov (namiesto presnej 0.0)
        theta = np.append(theta, np.random.uniform(-0.001, 0.001, K_ADD))
        
        res = minimize(obj_and_grad, theta, method="BFGS", jac=True, options={"gtol": 1e-5, "maxiter": 300})
        
        theta = res.x
        E = res.fun
        dev = (E - E0) * 1e3
        
        print(f"iter {it:2d}  ops={len(sel):3d}  E={E:.6f}  dev={dev:9.3f} mEh  gmax={gmax:.2e}", flush=True)
        
        if dev < CHEM:
            print(f"\nSUCCESS: Chemicka presnost ({dev:.3f} mEh) dosiahnuta s {len(sel)} operatormi za {time.time()-t0:.0f}s!", flush=True)
            break

if __name__ == "__main__":
    main()
