# run_basin_uccsdt.py — Globálna Basin-Hopping optimalizácia pre prelomenie lokálneho minima
import numpy as np, time
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import minimize, basinhopping
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
        
    pool_SD, pool_T = [], []
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
                            
    pool_full = pool_SD + pool_T
    
    def get_optimizer_funcs(active_pool):
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
            return E + e_core, grads
        return obj_and_grad

    print(f"\n--- FÁZA 1: Rýchly S+D štart ---", flush=True)
    obj_sd = get_optimizer_funcs(pool_SD)
    res_sd = minimize(obj_sd, np.zeros(len(pool_SD)), method="L-BFGS-B", jac=True, options={"gtol": 1e-4, "ftol": 1e-6, "maxiter": 150})
    print(f"Fáza 1 hotová. Odchýlka: {(res_sd.fun - E0)*1e3:.3f} mEh", flush=True)
    
    print(f"\n--- FÁZA 2: Globálny Basin-Hopping pre S+D+T (815 parametrov) ---", flush=True)
    obj_full = get_optimizer_funcs(pool_full)
    theta_init = np.concatenate([res_sd.x, np.zeros(len(pool_T))])
    
    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "jac": True,
        "options": {"gtol": 1e-6, "ftol": 1e-9, "maxiter": 200}
    }
    
    step_count = [0]
    def print_progress(x, f, accept):
        step_count[0] += 1
        dev = (f - E0) * 1e3
        print(f"Basin Hop Step {step_count[0]:2d} | Energia E={f:.6f} | Odchylka dev={dev:9.3f} mEh | Akceptovane={accept}", flush=True)

    t0 = time.time()
    # Spustíme globálne skoky, ktoré vytrasú systém z lokálnych miním
    res_global = basinhopping(
        obj_full, 
        theta_init, 
        niter=15,               # Počet globálnych skokov von z minima
        T=0.001,                # Teplota pre prienik cez bariéry
        stepsize=0.02,          # Veľkosť skoku
        minimizer_kwargs=minimizer_kwargs,
        callback=print_progress
    )
    
    final_dev = (res_global.fun - E0) * 1e3
    print(f"\nVýpočet hotový za {time.time()-t0:.0f}s.", flush=True)
    print(f"FINÁLNA ODCHÝLKA: {final_dev:.3f} mEh")
    
    if final_dev < 1.6:
        print("\nSUCCESS: Chemická presnosť (< 1.6 mEh) úspešne pokorená! 🎉", flush=True)

if __name__ == "__main__":
    main()
