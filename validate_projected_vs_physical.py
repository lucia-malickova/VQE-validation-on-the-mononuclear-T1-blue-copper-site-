#!/usr/bin/env python3
"""
Decisive equivalence test for the spin-projected ADAPT-VQE result.

Run this from the ROOT of the current GitHub repository after
run_adapt_doublet.py has produced adapt_doublet_spin_pure.json.

It compares:

  (1) the state actually optimized in run_adapt_doublet.py
          prod_k exp(theta_k U^H A_k U) |HF>_doublet

  (2) the unprojected fermionic sequence that circuit_metrics_doublet.py
      implicitly turns into an 18-qubit circuit
          prod_k exp(theta_k A_k) |HF>

Both are evaluated exactly in the (N_alpha,N_beta)=(8,7) determinant sector.

Optional --qiskit additionally reconstructs the Qiskit circuit used by
circuit_metrics_doublet.py, transpiles it, simulates its statevector, and
compares that state with the projected ADAPT state.  This second test requires
qiskit, qiskit-nature and qiskit-aer.

Outputs:
  projected_vs_physical_validation.json
  projected_vs_physical_validation.txt

The script DOES NOT modify the repository or submit anything to hardware.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import expm_multiply

from spin_fci import (
    read_fcidump,
    pure_spin_hamiltonian,
    hf_determinant,
    excitation_specs_from_hf,
    excitation_generator,
    spin_square_expectation,
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scalar(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fcidump",
        default="artifacts/prod/active.FCIDUMP",
        help="FCIDUMP used for the published/revised calculation",
    )
    ap.add_argument(
        "--checkpoint",
        default="adapt_doublet_spin_pure.json",
        help="JSON produced by run_adapt_doublet.py",
    )
    ap.add_argument(
        "--output-json",
        default="projected_vs_physical_validation.json",
    )
    ap.add_argument(
        "--output-txt",
        default="projected_vs_physical_validation.txt",
    )
    ap.add_argument(
        "--qiskit",
        action="store_true",
        help="Also build and simulate the actual Qiskit circuit from circuit_metrics_doublet.py",
    )
    ap.add_argument(
        "--optimization-level",
        type=int,
        default=3,
        choices=(0, 1, 2, 3),
        help="Qiskit transpiler level for the optional circuit test",
    )
    args = ap.parse_args()

    if not Path(args.fcidump).is_file():
        raise SystemExit(f"Missing FCIDUMP: {args.fcidump}")
    if not Path(args.checkpoint).is_file():
        raise SystemExit(
            f"Missing checkpoint: {args.checkpoint}\n"
            "First run:\n"
            "  python run_adapt_doublet.py artifacts/prod/active.FCIDUMP"
        )

    with open(args.checkpoint, "r", encoding="utf-8") as fh:
        ck = json.load(fh)

    selected = [int(x) for x in ck["selected_indices"]]
    theta = np.asarray(ck["theta"], dtype=float)
    if len(selected) != len(theta):
        raise RuntimeError(
            f"selected/theta length mismatch: {len(selected)} vs {len(theta)}"
        )
    if not np.all(np.isfinite(theta)):
        raise RuntimeError("theta contains non-finite values")

    data = read_fcidump(args.fcidump)
    nalpha = (data.nelec + data.ms2) // 2
    nbeta = data.nelec - nalpha
    ms = 0.5 * (nalpha - nbeta)

    if abs(ms - 0.5) > 1e-12:
        raise RuntimeError(f"This test expects M_S=1/2, got {ms}")

    basis, Hms, U, Hd, Splus = pure_spin_hamiltonian(
        data, nalpha, nbeta, target_s=0.5
    )
    E0 = float(np.linalg.eigvalsh(Hd)[0])

    hfd = hf_determinant(data.norb, nalpha, nbeta)
    pos = {int(d): i for i, d in enumerate(basis)}[hfd]
    psi0_ms = np.zeros(len(basis), dtype=complex)
    psi0_ms[pos] = 1.0

    phi0 = U.conj().T @ psi0_ms
    phi0_norm2 = float(np.vdot(phi0, phi0).real)
    if abs(phi0_norm2 - 1.0) > 1e-10:
        raise RuntimeError(
            f"HF determinant is not pure doublet: projection norm^2={phi0_norm2:.12g}"
        )
    phi0 /= np.sqrt(phi0_norm2)

    specs = excitation_specs_from_hf(
        data.norb, nalpha, nbeta, max_rank=3
    )
    if max(selected, default=-1) >= len(specs):
        raise RuntimeError(
            f"Selected pool index exceeds pool length {len(specs)}"
        )

    A_cache = {}

    def Ams(k):
        if k not in A_cache:
            A_cache[k] = excitation_generator(basis, *specs[k])
        return A_cache[k]

    def Ad(k):
        M = U.conj().T @ (Ams(k) @ U)
        return 0.5 * (M - M.conj().T)

    # ------------------------------------------------------------------
    # 1) Reconstruct exactly the state that run_adapt_doublet.py optimized
    # ------------------------------------------------------------------
    phi_proj = phi0.copy()
    for t, k in zip(theta, selected):
        phi_proj = expm_multiply(t * Ad(k), phi_proj)
    phi_proj /= np.linalg.norm(phi_proj)
    psi_proj = U @ phi_proj
    psi_proj /= np.linalg.norm(psi_proj)

    E_proj = float(np.vdot(psi_proj, Hms @ psi_proj).real)
    S2_proj = spin_square_expectation(psi_proj, Splus, ms=ms)
    dev_proj_mEh = (E_proj - E0) * 1e3

    # ------------------------------------------------------------------
    # 2) Apply the same selected *unprojected* fermionic generators and
    #    parameters.  This is the mathematical target represented by the
    #    operator sequence in circuit_metrics_doublet.py before any extra
    #    Qiskit synthesis/Trotterisation effects.
    # ------------------------------------------------------------------
    psi_bare = psi0_ms.copy()
    for t, k in zip(theta, selected):
        psi_bare = expm_multiply(t * Ams(k), psi_bare)
    psi_bare /= np.linalg.norm(psi_bare)

    E_bare = float(np.vdot(psi_bare, Hms @ psi_bare).real)
    S2_bare = spin_square_expectation(psi_bare, Splus, ms=ms)
    F_proj_bare = float(abs(np.vdot(psi_proj, psi_bare)) ** 2)

    bare_doublet_coeff = U.conj().T @ psi_bare
    p_doublet_bare = float(np.vdot(bare_doublet_coeff, bare_doublet_coeff).real)
    p_doublet_bare = min(1.0, max(0.0, p_doublet_bare))
    p_nondoublet_bare = 1.0 - p_doublet_bare

    E_bare_projected = None
    F_proj_vs_bare_doublet_component = None
    if p_doublet_bare > 1e-14:
        psi_bare_D = U @ (bare_doublet_coeff / np.sqrt(p_doublet_bare))
        psi_bare_D /= np.linalg.norm(psi_bare_D)
        E_bare_projected = float(np.vdot(psi_bare_D, Hms @ psi_bare_D).real)
        F_proj_vs_bare_doublet_component = float(
            abs(np.vdot(psi_proj, psi_bare_D)) ** 2
        )

    # In CAS(15e,9o), N=15 permits only S=1/2 and S=3/2.
    max_spin = 0.5 * min(data.nelec, 2 * data.norb - data.nelec)
    quartet_weight_from_S2 = None
    if abs(max_spin - 1.5) < 1e-12:
        quartet_weight_from_S2 = float((S2_bare - 0.75) / (3.75 - 0.75))

    # ------------------------------------------------------------------
    # 3) Structural diagnostic: does each selected A_k preserve the
    #    doublet subspace?  If not, PAP and A are not interchangeable.
    # ------------------------------------------------------------------
    leakage_ratios = []
    for k in selected:
        AU = Ams(k) @ U
        inside = U @ (U.conj().T @ AU)
        denom = float(np.linalg.norm(AU))
        leak = float(np.linalg.norm(AU - inside))
        leakage_ratios.append(0.0 if denom == 0.0 else leak / denom)

    leak_arr = np.asarray(leakage_ratios, dtype=float)
    leak_summary = {
        "count_selected": len(selected),
        "count_non_spin_preserving_gt_1e-10": int(np.sum(leak_arr > 1e-10)),
        "max_ratio": float(np.max(leak_arr)) if len(leak_arr) else 0.0,
        "median_ratio": float(np.median(leak_arr)) if len(leak_arr) else 0.0,
        "mean_ratio": float(np.mean(leak_arr)) if len(leak_arr) else 0.0,
    }

    saved_final_energy = None
    saved_final_dev = None
    if ck.get("history"):
        saved_final_energy = float(ck["history"][-1][1])
        saved_final_dev = float(ck["history"][-1][2])

    # Tight equivalence criterion.  This is deliberately much tighter than
    # chemical accuracy: here we are testing whether two claimed
    # representations describe the SAME state/algorithm.
    equivalence_pass = bool(
        F_proj_bare > 1.0 - 1e-8
        and abs(E_bare - E_proj) < 1e-8
        and abs(S2_bare - 0.75) < 1e-8
    )

    result = {
        "input": {
            "fcidump": args.fcidump,
            "fcidump_sha256": sha256_file(args.fcidump),
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "norb": data.norb,
            "nelec": data.nelec,
            "nalpha": nalpha,
            "nbeta": nbeta,
            "Ms": ms,
            "Ms_sector_dim": int(len(basis)),
            "pure_doublet_dim": int(U.shape[1]),
            "pool_size": int(len(specs)),
            "n_selected": int(len(selected)),
        },
        "saved_checkpoint": {
            "final_energy_Eh": saved_final_energy,
            "final_dev_mEh": saved_final_dev,
        },
        "projected_ADAPT_state": {
            "energy_Eh": E_proj,
            "exact_doublet_reference_Eh": E0,
            "dev_mEh": dev_proj_mEh,
            "S2": S2_proj,
        },
        "same_parameters_unprojected_generators": {
            "energy_Eh": E_bare,
            "dev_vs_exact_doublet_mEh": (E_bare - E0) * 1e3,
            "S2": S2_bare,
            "fidelity_with_projected_state": F_proj_bare,
            "doublet_weight": p_doublet_bare,
            "non_doublet_weight": p_nondoublet_bare,
            "quartet_weight_from_S2_if_applicable": quartet_weight_from_S2,
            "energy_of_normalized_doublet_component_Eh": E_bare_projected,
            "fidelity_projected_vs_normalized_doublet_component":
                F_proj_vs_bare_doublet_component,
        },
        "selected_generator_doublet_leakage": {
            **leak_summary,
            "ratios_in_selected_order": leakage_ratios,
        },
        "equivalence_test": {
            "PASS_same_algorithmic_state": equivalence_pass,
            "criteria": {
                "fidelity_min": 1.0 - 1e-8,
                "abs_energy_difference_Eh_max": 1e-8,
                "abs_S2_minus_0p75_max": 1e-8,
            },
            "energy_difference_bare_minus_projected_Eh": E_bare - E_proj,
        },
    }

    if args.qiskit:
        try:
            from qiskit import QuantumCircuit, transpile
            from qiskit.circuit.library import PauliEvolutionGate
            from qiskit_nature.second_q.mappers import JordanWignerMapper
            from qiskit_nature.second_q.operators import FermionicOp
            from qiskit_aer import AerSimulator
        except Exception as exc:
            raise RuntimeError(
                "--qiskit requested but Qiskit/Qiskit Nature/Aer imports failed"
            ) from exc

        mapper = JordanWignerMapper()
        nso = 2 * data.norb
        occ = [p for p in range(nso) if (hfd >> p) & 1]

        qc = QuantumCircuit(nso)
        for o in occ:
            qc.x(o)

        for k, th in zip(selected, theta):
            rem, add = specs[k]
            create_fwd = " ".join(f"+_{p}" for p in add)
            annih_fwd = " ".join(f"-_{q}" for q in reversed(rem))
            create_bwd = " ".join(f"+_{p}" for p in rem)
            annih_bwd = " ".join(f"-_{q}" for q in reversed(add))
            fdict = {
                f"{create_fwd} {annih_fwd}": 1.0,
                f"{create_bwd} {annih_bwd}": -1.0,
            }
            fop = FermionicOp(fdict, num_spin_orbitals=nso)
            qubit_op = mapper.map(fop)
            H_herm = (-1j) * qubit_op
            qc.append(PauliEvolutionGate(H_herm, time=-float(th)), range(nso))

        tc = transpile(
            qc,
            basis_gates=["rz", "sx", "x", "cx"],
            optimization_level=args.optimization_level,
        )
        gate_counts = {str(k): int(v) for k, v in tc.count_ops().items()}

        tc.save_statevector()
        sim = AerSimulator(method="statevector")
        sv = np.asarray(sim.run(tc).result().data(0)["statevector"], dtype=complex)

        psi_q = np.array([sv[int(det)] for det in basis], dtype=complex)
        p_ms = float(np.vdot(psi_q, psi_q).real)
        if p_ms <= 1e-14:
            raise RuntimeError("Qiskit circuit has negligible weight in target Ms sector")
        psi_q_norm = psi_q / np.sqrt(p_ms)

        E_q = float(np.vdot(psi_q_norm, Hms @ psi_q_norm).real)
        S2_q = spin_square_expectation(psi_q_norm, Splus, ms=ms)
        F_proj_q = float(abs(np.vdot(psi_proj, psi_q_norm)) ** 2)
        F_bare_q = float(abs(np.vdot(psi_bare, psi_q_norm)) ** 2)

        qD = U.conj().T @ psi_q_norm
        pD_q = float(np.vdot(qD, qD).real)
        pD_q = min(1.0, max(0.0, pD_q))

        q_pass = bool(
            F_proj_q > 1.0 - 1e-8
            and abs(E_q - E_proj) < 1e-8
            and abs(S2_q - 0.75) < 1e-8
        )

        result["qiskit_transpiled_circuit"] = {
            "optimization_level": args.optimization_level,
            "depth": int(tc.depth()),
            "gate_counts": gate_counts,
            "Ms_sector_weight": p_ms,
            "energy_Eh_conditioned_on_Ms_sector": E_q,
            "dev_vs_exact_doublet_mEh": (E_q - E0) * 1e3,
            "S2_conditioned_on_Ms_sector": S2_q,
            "doublet_weight_within_Ms_sector": pD_q,
            "fidelity_with_projected_ADAPT_state": F_proj_q,
            "fidelity_with_exact_unprojected_generator_sequence": F_bare_q,
            "PASS_same_as_projected_ADAPT_state": q_pass,
        }

    with open(args.output_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=scalar)

    lines = []
    lines.append("PROJECTED-vs-PHYSICAL ADAPT EQUIVALENCE TEST")
    lines.append("=" * 60)
    lines.append(f"FCIDUMP SHA256 : {result['input']['fcidump_sha256']}")
    lines.append(f"checkpoint SHA : {result['input']['checkpoint_sha256']}")
    lines.append(f"selected ops   : {len(selected)}")
    lines.append("")
    lines.append("Projected ADAPT (the state actually optimized):")
    lines.append(f"  E       = {E_proj:.12f} Eh")
    lines.append(f"  dev     = {dev_proj_mEh:.6f} mEh")
    lines.append(f"  <S^2>   = {S2_proj:.12f}")
    lines.append("")
    lines.append("Same theta + same selected labels, but UNPROJECTED generators:")
    lines.append(f"  E       = {E_bare:.12f} Eh")
    lines.append(f"  dev     = {(E_bare-E0)*1e3:.6f} mEh")
    lines.append(f"  <S^2>   = {S2_bare:.12f}")
    lines.append(f"  fidelity(projected,bare) = {F_proj_bare:.12g}")
    lines.append(f"  doublet weight            = {p_doublet_bare:.12g}")
    lines.append(f"  non-doublet weight        = {p_nondoublet_bare:.12g}")
    if quartet_weight_from_S2 is not None:
        lines.append(f"  quartet weight from S^2   = {quartet_weight_from_S2:.12g}")
    lines.append("")
    lines.append("Selected-generator leakage out of the exact doublet subspace:")
    lines.append(
        f"  non-spin-preserving generators (>1e-10): "
        f"{leak_summary['count_non_spin_preserving_gt_1e-10']} / {len(selected)}"
    )
    lines.append(f"  max leakage ratio    = {leak_summary['max_ratio']:.12g}")
    lines.append(f"  median leakage ratio = {leak_summary['median_ratio']:.12g}")
    lines.append("")
    lines.append(
        "EQUIVALENCE VERDICT: "
        + ("PASS" if equivalence_pass else "FAIL")
    )

    if "qiskit_transpiled_circuit" in result:
        q = result["qiskit_transpiled_circuit"]
        lines.append("")
        lines.append("Actual Qiskit synthesized/transpiled circuit:")
        lines.append(f"  depth   = {q['depth']}")
        lines.append(f"  E       = {q['energy_Eh_conditioned_on_Ms_sector']:.12f} Eh")
        lines.append(f"  <S^2>   = {q['S2_conditioned_on_Ms_sector']:.12f}")
        lines.append(f"  doublet weight = {q['doublet_weight_within_Ms_sector']:.12g}")
        lines.append(f"  fidelity(projected,Qiskit) = {q['fidelity_with_projected_ADAPT_state']:.12g}")
        lines.append(
            "  QISKIT EQUIVALENCE VERDICT: "
            + ("PASS" if q["PASS_same_as_projected_ADAPT_state"] else "FAIL")
        )

    Path(args.output_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved: {args.output_json}")
    print(f"Saved: {args.output_txt}")


if __name__ == "__main__":
    main()
