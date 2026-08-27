# run_vqe_fast.py — IDEAL VQE: FCIDUMP H, N-sektor + L-BFGS-B + dynamicke reps
import sys, time, numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ExcitationPreserving
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.circuit.library import HartreeFock
import config

p = config.profile()
d = np.load(config.INTEGRALS_NPZ)
ncas, e_core = int(d["ncas"]), float(d["e_core"])
e_casscf = float(d["e_casscf"])
n_particles = (int(d["n_alpha"]), int(d["n_beta"]))
N = n_particles[0] + n_particles[1]

maxiter = int(sys.argv[1]) if len(sys.argv) > 1 else p["ideal_maxiter"]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else p["hea_reps"]

mapper = JordanWignerMapper()
qop = mapper.map(fcidump_to_problem(FCIDump.from_file(config.FCIDUMP)).hamiltonian.second_q_op())
nq = qop.num_qubits

t = time.time()
H = qop.to_matrix(sparse=True).tocsr()
idx = np.array([i for i in range(2**nq) if bin(i).count("1") == N], dtype=np.int64)
Hsub = H[idx][:, idx].toarray()
E0_exact = float(np.linalg.eigvalsh(Hsub)[0]) + e_core
print(f"[H] {time.time()-t:.1f}s  N-sektor={len(idx)}  exact E0={E0_exact:.6f}  CASSCF={e_casscf:.6f}", flush=True)

hf  = HartreeFock(ncas, n_particles, mapper)
anz = ExcitationPreserving(nq, entanglement="linear", reps=reps)
raw = QuantumCircuit(nq); raw.compose(hf, inplace=True); raw.compose(anz, inplace=True)
ansatz = transpile(raw, basis_gates=config.BASIS_GATES, optimization_level=1)
nparams = ansatz.num_parameters
print(f"[ansatz] qubits={nq} params={nparams} depth={ansatz.depth()} reps={reps}", flush=True)

history = []
def energy(theta):
    ps = Statevector(ansatz.assign_parameters(theta)).data[idx]
    E = float((ps.conj() @ (Hsub @ ps)).real) + e_core
    history.append(E)
    n = len(history)
    if n == 1 or n % 25 == 0:
        leak = 1.0 - float(np.vdot(ps, ps).real)
        print(f"  eval {n:5d}: E={E:.6f}  dev(exact)={(E-E0_exact)*1e3:+.3f} mEh  leak={leak:.1e}", flush=True)
    return E

rng = np.random.default_rng(0)
x0 = 0.2 * rng.standard_normal(nparams)
print(f"[start] maxiter={maxiter}  E(x0)={energy(x0):.6f}", flush=True)

t = time.time()
res = minimize(energy, x0, method="L-BFGS-B",
               options={"maxiter": maxiter, "eps": 1e-4, "ftol": 1e-12, "gtol": 1e-8})
print(f"[done] {time.time()-t:.1f}s  evals={len(history)}  success={res.success}", flush=True)
print(f"[result] E_vqe={res.fun:.6f}", flush=True)
print(f"[result] dev vs exact = {(res.fun-E0_exact)*1e3:+.3f} mEh   (chem acc 1.6 mEh)", flush=True)
print(f"[result] dev vs CASSCF= {(res.fun-e_casscf)*1e3:+.3f} mEh", flush=True)

out_npz = config.HISTORY_NPZ.replace(".npz", f"_reps{reps}.npz")
np.savez(out_npz, hist_ideal=np.array(history), e0_exact=E0_exact,
         e_casscf=e_casscf, e_dft=float(d["e_dft"]), qubits=nq, params=nparams,
         depth=ansatz.depth(), reps=reps)
print(f"[saved] {out_npz}", flush=True)