# run_bootstrapped_uccsdt.py — Dvojfázový UCCSD -> UCCSDT (Bootstrapping)
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
    
    def gen(op_dict):
        f = FermionicOp(op_dict, num_spin_orbitals=nq)
        m = mapper.map(f).to_matrix(sparse=True)
        return sp.csr_matrix(m[idx][:, idx])
        
    pool_SD = []
    # 1. Singles + Doubles
    for a in occ:
        for r in vir:
            A = gen({f"+_{r} -_{a}": 1.0, f"+_{a} -_{r}": -1.0})
            if A.nnz: pool_SD.append(A)
            
    for ii in range(len(occ)):
        for jj in range(ii+1, len(occ)):
            for kk in range(len(vir)):
                for ll in range(kk+1, len(vir)):
                    a, b, r, s = occ[ii], occ[jj], vir[kk], vir[ll]
                    A = gen({f"+_{r} +_{s} -_{b} -_{a}": 1.0, f"+_{a} +_{b} -_{s} -_{r}": -1.0})
                    if A.nnz: pool_SD.append(A)

    pool_T = []
    # 2. Triples
    for i in range(len(occ)):
        for j in range(i+1, len(occ)):
            for k in range(j+1, len(occ)):
                for l in range(len(vir)):
                    for m in range(l+1, len(vir)):
                        for n in range(m+1, len(vir)):
                            a, b, c = occ[i], occ[j], occ[k]
                            r, s, t = vir[l], vir[m], vir[n]
                            A = gen({f"+_{r} +_{s} +_{t} -_{c} -_{b} -_{a}": 1.0, f"+_{a} +_{b} +_{c} -_{t} -_{s} -_{r}": -1.0})
                            if A.nnz: pool_T.append(A)
                            
    def get_optimizer_funcs(active_pool):
        last_E = [0.0]; last_dev = [0.0]; last_gmax = [0.0]
        def obj_and_grad(th):
            v_list = [psi0]
            for idx_op, val in enumerate(th):
                v_list.append(expm_multiply(val * active_pool[idx_op], v_list[-1]))
            psi = v_list[-1]
            E = float((psi.conj() @ (Hsub @ psi)).real)
            w = Hsub @ psi
            grads = np.zeros_like(th)
            for idx_op in reversed(range(len(th))):
                A = active_pool[idx_op]
                grads[idx_op] = 2.0 * np.real(np.vdot(w, A @ v_list[idx_op+1]))
                w = expm_multiply(-th[idx_op] * A, w)
            last_E[0] = E + e_core
            last_dev[0] = (last_E[0] - E0) * 1e3
            last_gmax[0] = np.max(np.abs(grads))
            return last_E[0], grads
            
        iter_count = [0]
        def callback(xk):
            iter_count[0] += 1
            print(f"Iter {iter_count[0]:3d} | E={last_E[0]:.6f} | dev={last_dev[0]:9.3f} mEh | gmax={last_gmax[0]:.2e}", flush=True)
            
        return obj_and_grad, callback

    print(f"\n--- FÁZA 1: UCCSD (Len S+D, {len(pool_SD)} operátorov) ---", flush=True)
    obj_sd, cb_sd = get_optimizer_funcs(pool_SD)
    theta_sd_init = np.zeros(len(pool_SD))
    
    t0 = time.time()
    # Miernejšia tolerancia pre Fázu 1 (stačí nám dostať sa nahrubo na dno)
    res_sd = minimize(obj_sd, theta_sd_init, method="L-BFGS-B", jac=True, callback=cb_sd, options={"gtol": 1e-4, "ftol": 1e-6, "maxiter": 200})
    print(f"Fáza 1 hotová za {time.time()-t0:.0f}s. Odchýlka: {(res_sd.fun - E0)*1e3:.3f} mEh", flush=True)
    
    print(f"\n--- FÁZA 2: UCCSDT (S+D+T, {len(pool_SD) + len(pool_T)} operátorov) ---", flush=True)
    pool_full = pool_SD + pool_T
    obj_full, cb_full = get_optimizer_funcs(pool_full)
    
    # BOOTSTRAPPING: Použijeme zoptimalizované S+D uhly z Fázy 1, Triples sú na začiatku nula
    theta_full_init = np.concatenate([res_sd.x, np.zeros(len(pool_T))])
    
    t1 = time.time()
    # Prísna tolerancia pre presné dosiahnutie cieľa
    res_full = minimize(obj_full, theta_full_init, method="L-BFGS-B", jac=True, callback=cb_full, options={"gtol": 1e-6, "ftol": 1e-9, "maxiter": 300})
    
    final_dev = (res_full.fun - E0) * 1e3
    print(f"\nFáza 2 hotová za {time.time()-t1:.0f}s.", flush=True)
    print(f"FINÁLNA ODCHÝLKA: {final_dev:.3f} mEh")
    
    if final_dev < 1.6:
        print("\nSUCCESS: Chemická presnosť (< 1.6 mEh) úspešne dosiahnutá! 🎉", flush=True)

if __name__ == "__main__":
    main()
