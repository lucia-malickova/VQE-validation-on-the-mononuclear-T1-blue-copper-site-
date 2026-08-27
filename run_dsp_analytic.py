# run_dsp_analytic.py — DSP-VQE s analytickými gradientmi (žiadne zamrznutie)
import numpy as np, time
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import minimize
import config

def main():
    print("="*60, flush=True)
    print("SPÚŠŤAM DSP-VQE S ANALYTICKÝMI GRADIENTMI", flush=True)
    print("="*60, flush=True)

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
    print(f"Exaktna Doublet FCI E0 = {E0:.6f} Eh", flush=True)

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

    # Vyberme silný, direktívny pool kľúčových operátorov vrátane Triples
    pool = []
    # Singles & Doubles vzorka
    for a in occ[:4]:
        for r in vir[:2]:
            A = gen({f"+_{r} -_{a}": 1.0, f"+_{a} -_{r}": -1.0})
            if A.nnz: pool.append(A)
    # Triples zväzok pre koreláciu pod 1.6 mEh
    if len(occ) >= 3 and len(vir) >= 3:
        A = gen({f"+_{vir[0]} +_{vir[1]} +_{vir[2]} -_{occ[2]} -_{occ[1]} -_{occ[0]}": 1.0,
                 f"+_{occ[0]} +_{occ[1]} +_{occ[2]} -_{vir[2]} -_{vir[1]} -_{vir[0]}": -1.0})
        if A.nnz: pool.append(A)

    print(f"Aktivny DSP Pool: {len(pool)} operátorov.", flush=True)

    # Analytická obj_and_grad funkcia (rovnaká ako v ADAPT, ale pre fixný pool)
    def obj_and_grad(th):
        v_list = [psi0]
        for i, t in enumerate(th):
            v_list.append(expm_multiply(t * pool[i], v_list[-1]))
        psi = v_list[-1]
        E = float((psi.conj() @ (Hsub @ psi)).real)
        w = Hsub @ psi
        grads = np.zeros_like(th)
        for i in reversed(range(len(th))):
            A = pool[i]
            grads[i] = 2.0 * np.real(np.vdot(w, A @ v_list[i+1]))
            w = expm_multiply(-th[i] * A, w)
        return E + e_core, grads

    # Štartovacie uhly
    theta = np.zeros(len(pool))

    print("Spúšťam optimalizáciu s analytickým Jacobianom...", flush=True)
    res = minimize(obj_and_grad, theta, method="L-BFGS-B", jac=True, options={"gtol": 1e-7, "ftol": 1e-9, "maxiter": 600})
    
    E_final = float(res.fun)
    dev = (E_final - E0) * 1e3
    print(f"\nVýsledná DSP Energia: {E_final:.6f} Eh", flush=True)
    print(f"Chyba od FCI: {dev:.4f} mEh", flush=True)

    if dev < 1.6:
        print("SUCCESS: Chemicka presnost pod 1.6 mEh dosiahnuta!", flush=True)
    else:
        print("INFO: Dosiahnutá hodnota.", flush=True)

if __name__ == "__main__":
    main()
