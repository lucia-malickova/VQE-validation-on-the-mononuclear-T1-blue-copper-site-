"""STEP 2 (cheap, re-runnable): VQE on the emulator from saved integrals."""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import StatevectorEstimator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP, SPSA
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

import config


def build_ansatz(p, n_orb, n_particles, mapper):
    if p["ansatz"] == "uccsd":
        init = HartreeFock(n_orb, n_particles, mapper)
        return UCCSD(n_orb, n_particles, mapper, initial_state=init)
    
    hf_state = HartreeFock(n_orb, n_particles, mapper)
    ansatz = EfficientSU2(2 * n_orb, su2_gates=["ry"],
                          entanglement="linear", reps=p["hea_reps"])
    ansatz.compose(hf_state, front=True, inplace=True)
    return ansatz


def run(estimator, ansatz, optimizer, qubit_op, e_core, label=""):
    history = []

    def cb(c, prm, val, md=None):
        history.append(val + e_core)
        if c == 1 or c % 5 == 0 or c == optimizer.maxiter:
            print(f"  [{label}] eval {c:4d}: E = {val + e_core:.6f} Eh", flush=True)

    VQE(estimator, ansatz, optimizer, callback=cb
        ).compute_minimum_eigenvalue(qubit_op)
    return history


def main():
    p = config.profile()
    d = np.load(config.INTEGRALS_NPZ)
    h1, h2, e_core = d["h1"], d["h2"], float(d["e_core"])
    ncas = int(d["ncas"]); e_casscf = float(d["e_casscf"])
    n_particles = (int(d["n_alpha"]), int(d["n_beta"]))

    mapper = JordanWignerMapper()
    elec = ElectronicEnergy.from_raw_integrals(h1, h2)
    qubit_op = mapper.map(elec.second_q_op())
    print(f"[map] qubits = {qubit_op.num_qubits}")

    raw_ansatz = build_ansatz(p, ncas, n_particles, mapper)
    
    qc = QuantumCircuit(raw_ansatz.num_qubits)
    qc.compose(raw_ansatz, inplace=True)
    ansatz = transpile(qc, basis_gates=config.BASIS_GATES, optimization_level=1)
    
    tqc = ansatz
    two_q = tqc.count_ops().get("cz", 0) + tqc.count_ops().get("cx", 0)
    print(f"[resources] ansatz={p['ansatz']} depth={tqc.depth()} 2q={two_q}")

    print("[ideal] running state-vector VQE ...", flush=True)
    hist_ideal = run(StatevectorEstimator(), ansatz,
                     SLSQP(maxiter=p["ideal_maxiter"]), qubit_op, e_core,
                     label="ideal")

    print("[noisy] running Aer-noise VQE (method='automatic') ...", flush=True)
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(config.NOISE_1Q, 1),
                                   ["rz", "sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(config.NOISE_2Q, 2),
                                   ["cz", "cx"])
    
    sim = AerSimulator(noise_model=nm, method='automatic', device='CPU')
    sim.set_options(max_parallel_experiments=1, max_parallel_threads=1)
    
    from qiskit.primitives import BackendEstimatorV2
    est = BackendEstimatorV2(backend=sim)
    est.options.default_shots = p["shots"]
    
    hist_noisy = run(est, ansatz, SPSA(maxiter=p["noisy_maxiter"]),
                     qubit_op, e_core, label="noisy")

    np.savez(config.HISTORY_NPZ,
             hist_ideal=np.array(hist_ideal), hist_noisy=np.array(hist_noisy),
             e_casscf=e_casscf, e_dft=float(d["e_dft"]), depth=tqc.depth(),
             two_q=two_q, qubits=qubit_op.num_qubits, ansatz=p["ansatz"])
    print(f"[saved] {config.HISTORY_NPZ}")


if __name__ == "__main__":
    main()