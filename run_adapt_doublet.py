# run_adapt_doublet.py — ADAPT-VQE cielený na správny doubletový spinový stav (S=1/2)
import numpy as np, time
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import minimize
import config

def main():
    d = np.load(config.INTEGRALS_NPZ)
    e_core = float(d["e_core"]); ncas = int(d["ncas"])
    n_particles = (int(d["n_alpha"]), int(d["n_beta"])); N = sum(n_particles)
    
    print(f"Cielove el. / spinove obsadenie: alpha={n_particles[0]}, beta={n_particles[1]} (Spolu N={N})", flush=True)

    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.second_q.formats.fcidump import FCIDump
    from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
    from qiskit_nature.second_q.operators import FermionicOp
    from qiskit_nature.second_q.circuit.library import HartreeFock
    from qiskit.quantum_info import Statevector

    mapper = JordanWignerMapper()
    problem = fcidump_to_problem(FCIDump.from_file(config.FCIDUMP))
    qop = mapper.map(problem.hamiltonian.second_q_op())
    nq = qop.num_qubits
    
    Hfull = qop.to_matrix(sparse=True).tocsr()
    # Vybereme presne subspace pre N elektronov (Doublet sektor)
    idx = np.array([i for i in range(2**nq) if bin(i).count("1") == N], dtype=np.int64)
    Hsub = np.asarray(Hfull[idx][:, idx].todense())
    
    evals, evecs = np.linalg.eigh(Hsub)
    # Spravna referencia pre doublet (najnizsia energia v tomto spinovom sektore)
    E0 = float(evals[0]) + e_core
    print(f"Spravna Doublet FCI E0 = {E0:.6f} Eh", flush=True)

    hf_sv = Statevector(HartreeFock(ncas, n_particles, mapper)).data
    hf_i = int(np.argmax(np.abs(hf_sv)))
    occ = [i for i in range(nq) if (hf_i >> i) & 1]
    vir = [i for i in range(nq) if not ((hf_i >> i) & 1)]
    pos = int(np.where(idx == hf_i)[0][0])
    
    psi0 = np.zeros(len(idx), complex); psi0[pos] = 1.0
    print(f"E(HF) = {float((psi0.conj()@(Hsub@psi0)).real)+e_core:.6f} Eh", flush=True)
    
    # Skontrolujeme prekryv s novym spravnym doublet stavom
    ovl = float(np.abs(evecs[:, 0].conj() @ psi0)**2)
    print(f"Prekryv |<Doublet_FCI | HF>|^2 = {ovl:.4f} (Musi byt nenulovy!)", flush=True)

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
                    
    print(f"Pool = {len(pool)} operatorov (S+D)", flush=True)

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

    CHEM = 1.6; MAX_ITERS = 40; K_ADD = 4
    hist = []; t0 = time.time()
    
    for it in range(1, MAX_ITERS+1):
        v = psi0.copy()
        for t, k in zip(theta, sel):
            v = expm_multiply(t * pool[k], v)
            
        u = Hsub @ v
        pool_grads = np.array([abs(2.0*np.real(np.vdot(u, pool[k] @ v))) for k in range(len(pool))])
        
        top_indices = np.argsort(pool_grads)[-K_ADD:][::-1]
        gmax = float(pool_grads[top_indices[0]])
        
        if gmax < 1e-5:
            print(f"[stop] max|grad| = {gmax:.2e}", flush=True); break
            
        for kbest in top_indices:
            sel.append(kbest)
            
        theta = np.append(theta, np.random.uniform(-0.001, 0.001, K_ADD))
        
        res = minimize(obj_and_grad, theta, method="L-BFGS-B", jac=True, options={"gtol": 1e-6, "ftol": 1e-8, "maxiter": 600})
        
        theta = res.x; E = float(res.fun); dev = (E - E0) * 1e3
        hist.append((len(sel), E, dev, gmax))
        print(f"iter {it:2d}  ops={len(sel):3d}  E={E:.6f}  dev={dev:9.3f} mEh  gmax={gmax:.2e}", flush=True)
        
        if dev < CHEM:
            print(f"\nSUCCESS: Chemicka presnost dosiahnuta! ({dev:.3f} mEh)", flush=True)
            break

    nS = sum(1 for k in sel if labels[k][0] == "S"); nD = sum(1 for k in sel if labels[k][0] == "D")
    print(f"\n[circuit] aplikacii={len(sel)} (S={nS}, D={nD})  ~2q brany≈{2*nS+13*nD}", flush=True)
    np.savez(config.HISTORY_NPZ.replace(".npz", "_adapt.npz"),
             n_ops=np.array([h[0] for h in hist]), energy=np.array([h[1] for h in hist]),
             dev_meh=np.array([h[2] for h in hist]), e0_exact=E0, e_casscf=float(d["e_casscf"]))
    print("[saved] vqe_history_adapt.npz", flush=True)

if __name__ == "__main__":
    main()
