# run_ultimate_sdt.py — Plný S+D+T pool s priebojnou optimalizáciou (K_ADD=8)
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
    
    # 1. Singles
    for a in occ:
        for r in vir:
            A = gen({f"+_{r} -_{a}": 1.0, f"+_{a} -_{r}": -1.0})
            if A.nnz: pool.append(A); labels.append(("S", a, r))
            
    # 2. Doubles
    for ii in range(len(occ)):
        for jj in range(ii+1, len(occ)):
            for kk in range(len(vir)):
                for ll in range(kk+1, len(vir)):
                    a, b, r, s = occ[ii], occ[jj], vir[kk], vir[ll]
                    A = gen({f"+_{r} +_{s} -_{b} -_{a}": 1.0, f"+_{a} +_{b} -_{s} -_{r}": -1.0})
                    if A.nnz: pool.append(A); labels.append(("D", a, b, r, s))

    # 3. Triples (Záverečná zbraň pre pokorenie 4 mEh)
    for i in range(len(occ)):
        for j in range(i+1, len(occ)):
            for k in range(j+1, len(occ)):
                for l in range(len(vir)):
                    for m in range(l+1, len(vir)):
                        for n in range(m+1, len(vir)):
                            a, b, c = occ[i], occ[j], occ[k]
                            r, s, t = vir[l], vir[m], vir[n]
                            op_str1 = f"+_{r} +_{s} +_{t} -_{c} -_{b} -_{a}"
                            op_str2 = f"+_{a} +_{b} +_{c} -_{t} -_{s} -_{r}"
                            A = gen({op_str1: 1.0, op_str2: -1.0})
                            if A.nnz: pool.append(A); labels.append(("T", a, b, c, r, s, t))
                            
    print(f"Pool = {len(pool)} operatorov (Plná S+D+T expresivita)", flush=True)
    
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

    # AGRESÍVNE NASTAVENIA Z TVOJHO NAJLEPŠIEHO LOGU
    CHEM = 1.6; MAX_ITERS = 40; K_ADD = 8
    t0 = time.time()
    
    for it in range(1, MAX_ITERS+1):
        v = psi0.copy()
        for t, k in zip(theta, sel):
            v = expm_multiply(t * pool[k], v)
            
        u = Hsub @ v
        pool_grads = np.array([abs(2.0*np.real(np.vdot(u, pool[k] @ v))) for k in range(len(pool))])
        
        top_indices = np.argsort(pool_grads)[-K_ADD:][::-1]
        gmax = float(pool_grads[top_indices[0]])
        
        if gmax < 1e-6:
            print(f"[stop] max|grad| = {gmax:.2e}", flush=True); break
            
        for kbest in top_indices:
            sel.append(kbest)
            
        # Zlom symetrie
        theta = np.append(theta, np.random.uniform(-0.001, 0.001, K_ADD))
        
        # Ostrý L-BFGS-B, ktorý nespanikári
        res = minimize(obj_and_grad, theta, method="L-BFGS-B", jac=True, options={"gtol": 1e-7, "ftol": 1e-9, "maxiter": 800})
        
        theta = res.x
        E = res.fun
        dev = (E - E0) * 1e3
        
        print(f"iter {it:2d}  ops={len(sel):3d}  E={E:.6f}  dev={dev:9.3f} mEh  gmax={gmax:.2e}", flush=True)
        
        if dev < CHEM:
            print(f"\nSUCCESS: Chemicka presnost ({dev:.3f} mEh) dosiahnuta s {len(sel)} operatormi za {time.time()-t0:.0f}s!", flush=True)
            break

if __name__ == "__main__":
    main()
