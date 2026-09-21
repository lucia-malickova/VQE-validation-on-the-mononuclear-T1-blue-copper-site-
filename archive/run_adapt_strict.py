# run_adapt_strict.py — ADAPT-VQE: 1 operator/krok, perturbovany start, prisna re-optimalizacia
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
    nq = qop.num_qubits
    Hfull = qop.to_matrix(sparse=True).tocsr()
    idx = np.array([i for i in range(2**nq) if bin(i).count("1") == N], dtype=np.int64)
    Hsub = np.asarray(Hfull[idx][:, idx].todense())

    evals, evecs = np.linalg.eigh(Hsub)
    E0 = float(evals[0]) + e_core
    print(f"exact FCI E0 = {E0:.6f} Eh", flush=True)

    hf_sv = Statevector(HartreeFock(ncas, n_particles, mapper)).data
    hf_i = int(np.argmax(np.abs(hf_sv)))
    occ = [i for i in range(nq) if (hf_i >> i) & 1]
    vir = [i for i in range(nq) if not ((hf_i >> i) & 1)]
    pos = int(np.where(idx == hf_i)[0][0])
    psi0 = np.zeros(len(idx), complex); psi0[pos] = 1.0
    print(f"E(HF) = {float((psi0.conj()@(Hsub@psi0)).real)+e_core:.6f} Eh", flush=True)
    # kontrola: prekryv exact zakladneho stavu s HF (ak je maly, HF je zly start)
    print(f"|<FCI|HF>|^2 = {abs(evecs[:,0]@psi0)**2:.4f}", flush=True)

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
    print(f"pool = {len(pool)} operatorov (S+D)", flush=True)

    sel = []; theta = np.zeros(0)
    rng = np.random.default_rng(0)

    def build(th):
        v = psi0.copy()
        for t, k in zip(th, sel):
            v = expm_multiply(t * pool[k], v)
        return v
    def energy(th):
        v = build(th)
        return float((v.conj() @ (Hsub @ v)).real) + e_core

    CHEM = 1.6; MAXOPS = 60; GTOL = 1e-5
    hist = []; t0 = time.time()
    for it in range(1, MAXOPS + 1):
        psi = build(theta)
        u = Hsub @ psi
        grads = np.array([abs(2.0*np.real(np.vdot(u, pool[k] @ psi))) for k in range(len(pool))])
        kbest = int(np.argmax(grads)); gmax = float(grads[kbest])
        if gmax < GTOL:
            print(f"[stop] max|grad| = {gmax:.2e}", flush=True); break

        # >>> 1 operator za krok + perturbovany start noveho uhla <
        sel.append(kbest)
        theta = np.append(theta, rng.normal(0, 0.05))
        # prisna globalna re-optimalizacia vsetkych uhlov, viac restartov na istotu
        best = None
        for attempt in range(2):
            th0 = theta if attempt == 0 else theta + rng.normal(0, 0.02, size=theta.shape)
            r = minimize(energy, th0, method="BFGS", options={"gtol": 1e-8, "maxiter": 2000})
            if best is None or r.fun < best.fun: best = r
        theta = best.x; E = float(best.fun); dev = (E - E0) * 1e3
        hist.append((len(sel), E, dev, gmax))
        print(f"iter {it:3d}  ops={len(sel):3d}  add={labels[kbest]}  E={E:.6f}  dev={dev:9.4f} mEh  gmax={gmax:.2e}", flush=True)

        np.savez(config.HISTORY_NPZ.replace(".npz", "_adapt.npz"),
                 n_ops=np.array([h[0] for h in hist]), energy=np.array([h[1] for h in hist]),
                 dev_meh=np.array([h[2] for h in hist]), e0_exact=E0, e_casscf=float(d["e_casscf"]))

        if dev < CHEM:
            nS = sum(1 for k in sel if labels[k][0]=="S"); nD = sum(1 for k in sel if labels[k][0]=="D")
            print(f"\nSUCCESS: {dev:.4f} mEh za {len(sel)} operatorov ({time.time()-t0:.0f}s)", flush=True)
            print(f"[circuit] S={nS} D={nD}  ~2q brany≈{2*nS+13*nD}", flush=True)
            break

if __name__ == "__main__":
    main()