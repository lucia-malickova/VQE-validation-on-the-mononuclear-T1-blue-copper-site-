#!/usr/bin/env python3
"""Spin-pure S=1/2 ADAPT-VQE emulator benchmark for the T1 FCIDUMP.

Key correction relative to the original (pre-v2) version of this script:
  * the working state lives in the exact S=1/2, M_S=1/2 subspace ker(S_+),
    not merely the N=15 or M_S=1/2 determinant sector;
  * the S/D/T candidate pool preserves N_alpha,N_beta and every generator is
    projected into the pure-doublet subspace;
  * operator selection is deterministic (one operator per iteration by default);
  * the exact doublet energy, selected operators, parameters, and residual are saved.

This is an emulator/reference implementation.  A hardware circuit should use
spin-adapted generators or an explicitly validated spin-symmetry strategy; the
projected matrices used here are not themselves a hardware decomposition.

See paper Section 2.4 / arXiv:2609.20439v2 for the full methodology.
"""
from __future__ import annotations
import argparse
import json
import time

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply

from spin_fci import (
    read_fcidump, pure_spin_hamiltonian, sector_hamiltonian,
    hf_determinant, excitation_specs_from_hf, excitation_generator,
    spin_square_expectation,
)


def label_spec(spec, norb):
    rem, add = spec
    def lab(p):
        return ("a" if p < norb else "b") + str((p % norb) + 1)
    return {"remove": [lab(x) for x in rem], "add": [lab(x) for x in add]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fcidump")
    ap.add_argument("--max-iters", type=int, default=160)
    ap.add_argument("--grad-tol", type=float, default=1e-7)
    ap.add_argument("--energy-tol-meh", type=float, default=1.6)
    ap.add_argument("--max-rank", type=int, default=3)
    ap.add_argument("--output", default="adapt_doublet_spin_pure.npz")
    ap.add_argument("--json", default="adapt_doublet_spin_pure.json")
    args = ap.parse_args()

    data = read_fcidump(args.fcidump)
    nalpha = (data.nelec + data.ms2) // 2
    nbeta = data.nelec - nalpha
    ms = 0.5 * (nalpha - nbeta)
    if abs(ms - 0.5) > 1e-12:
        raise SystemExit("This reference implementation expects the paper's M_S=1/2 doublet target")

    basis, Hms, U, Hd, Splus = pure_spin_hamiltonian(data, nalpha, nbeta, target_s=0.5)
    ed = np.linalg.eigvalsh(Hd)
    E0 = float(ed[0])

    # Diagnostic quartet energy where available.
    Eq = np.nan
    if nalpha < data.norb and nbeta > 0:
        _, Hq = sector_hamiltonian(data, nalpha + 1, nbeta - 1)
        Eq = float(np.linalg.eigvalsh(Hq)[0])

    hfd = hf_determinant(data.norb, nalpha, nbeta)
    pos = {int(x): i for i, x in enumerate(basis)}[hfd]
    psi0_ms = np.zeros(len(basis), dtype=complex); psi0_ms[pos] = 1.0
    phi0 = U.conj().T @ psi0_ms
    proj_norm = float(np.vdot(phi0, phi0).real)
    if proj_norm < 1.0 - 1e-10:
        raise RuntimeError("HF determinant is not a pure doublet: projection norm %.12f" % proj_norm)
    phi0 /= np.sqrt(proj_norm)

    specs = excitation_specs_from_hf(data.norb, nalpha, nbeta, max_rank=args.max_rank)
    # For CAS(15e,9o), max_rank=3 must give 323 M_S-preserving non-reference excitations.
    print("doublet_dim=%d Ms_dim=%d pool_specs=%d E0_doublet=%.12f Eq=%.12f" %
          (Hd.shape[0], Hms.shape[0], len(specs), E0, Eq), flush=True)

    Ams_cache = {}
    Ad_cache = {}

    def Ams(k):
        if k not in Ams_cache:
            Ams_cache[k] = excitation_generator(basis, *specs[k])
        return Ams_cache[k]

    def Ad(k):
        if k not in Ad_cache:
            M = U.conj().T @ (Ams(k) @ U)
            M = 0.5 * (M - M.conj().T)
            if not np.all(np.isfinite(M)):
                raise RuntimeError(f"Ad({k}) obsahuje non-finite hodnoty!")
            Ad_cache[k] = M
        return Ad_cache[k]

    selected = []
    theta = np.zeros(0, dtype=float)
    history = []

    def build(th):
        v = phi0.copy()
        for t, k in zip(th, selected):
            v = expm_multiply(t * Ad(k), v)
        return v

    def obj_grad(th):
        states = [phi0]
        for t, k in zip(th, selected):
            states.append(expm_multiply(t * Ad(k), states[-1]))
        psi = states[-1]
        w = Hd @ psi
        E = float(np.vdot(psi, w).real)
        g = np.zeros(len(th), dtype=float)
        for i in range(len(th)-1, -1, -1):
            Ai = Ad(selected[i])
            g[i] = 2.0 * np.real(np.vdot(w, Ai @ states[i+1]))
            w = expm_multiply(-th[i] * Ai, w)
        return E, g

    t0 = time.time()
    for it in range(1, args.max_iters + 1):
        phi = build(theta)
        psi_ms = U @ phi
        u_ms = Hms @ psi_ms
        grads = np.full(len(specs), -np.inf, dtype=float)
        used = set(selected)
        for k in range(len(specs)):
            if k in used:
                continue
            grads[k] = abs(2.0 * np.real(np.vdot(u_ms, Ams(k) @ psi_ms)))
        kbest = int(np.argmax(grads))
        gmax = float(grads[kbest])
        if not np.isfinite(gmax) or gmax < args.grad_tol:
            print("[stop] max unused |grad| = %.3e" % gmax, flush=True)
            break

        selected.append(kbest)
        theta = np.append(theta, 0.0)  # deterministic symmetry-preserving start
        res = minimize(obj_grad, theta, method="L-BFGS-B", jac=True,
                       options={"gtol": 1e-8, "ftol": 1e-12, "maxiter": 1000, "maxls": 50})
        theta = np.asarray(res.x, dtype=float)
        E = float(res.fun)
        dev = (E - E0) * 1e3
        phi = build(theta); psi_ms = U @ phi
        s2 = spin_square_expectation(psi_ms, Splus, ms=0.5)
        resid = float(np.linalg.norm(Hd @ phi - E * phi))
        if not all(np.isfinite(x) for x in (E, dev, gmax, s2, resid)) or not np.all(np.isfinite(theta)):
            raise RuntimeError(f"NON-FINITE na iter={it}: E={E} dev={dev} gmax={gmax} s2={s2} resid={resid}")
        history.append((len(selected), E, dev, gmax, s2, resid))
        print("iter=%3d ops=%3d E=%.12f dev=%9.5f mEh gmax=%.3e <S2>=%.12f" %
              (it, len(selected), E, dev, gmax, s2), flush=True)
        if not np.isfinite(s2) or abs(s2 - 0.75) > 1e-9:
            raise RuntimeError("spin-purity regression: <S^2>=%s" % s2)
        if dev <= args.energy_tol_meh:
            print("SUCCESS: target energy tolerance reached in pure doublet sector", flush=True)
            break

    labels = [label_spec(specs[k], data.norb) for k in selected]
    hist = np.asarray(history, dtype=float) if history else np.empty((0,6), dtype=float)
    np.savez(args.output, theta=theta, selected=np.asarray(selected, dtype=int), history=hist,
             e0_doublet=E0, e0_quartet=Eq, doublet_basis=U)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump({
            "fcidump": args.fcidump,
            "e0_doublet_Eh": E0,
            "e0_quartet_Eh": Eq,
            "selected_labels": labels,
            "selected_indices": selected,
            "theta": theta.tolist(),
            "history_columns": ["n_ops","energy_Eh","dev_mEh","gmax","S2","residual_norm"],
            "history": hist.tolist(),
            "elapsed_s": time.time()-t0,
        }, fh, indent=2)
    print("saved %s and %s" % (args.output, args.json), flush=True)


if __name__ == "__main__":
    main()
