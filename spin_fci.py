#!/usr/bin/env python3
"""Spin-resolved exact diagonalisation utilities for the T1 blue-copper benchmark.

This module is intentionally independent of Qiskit.  It parses a standard real
FCIDUMP, builds the determinant Hamiltonian in a fixed (N_alpha,N_beta) sector,
and constructs the exact highest-weight S=1/2 subspace as ker(S_+).

Spin-orbital convention: alpha block first, beta block second.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence, Tuple
import re

import numpy as np
import scipy.linalg
import scipy.sparse as sp


@dataclass(frozen=True)
class FCIDumpData:
    norb: int
    nelec: int
    ms2: int
    h1: np.ndarray
    eri: np.ndarray
    ecore: float


def read_fcidump(path: str) -> FCIDumpData:
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    header = "".join(lines[: min(20, len(lines))])
    def get_int(key: str) -> int:
        m = re.search(r"\b%s\s*=\s*([+-]?\d+)" % re.escape(key), header, flags=re.I)
        if not m:
            raise ValueError("FCIDUMP header missing %s" % key)
        return int(m.group(1))
    norb, nelec, ms2 = get_int("NORB"), get_int("NELEC"), get_int("MS2")
    h1 = np.zeros((norb, norb), dtype=float)
    eri = np.zeros((norb, norb, norb, norb), dtype=float)
    ecore = 0.0
    in_body = False
    for raw in lines:
        if "&END" in raw.upper() or raw.strip() == "/":
            in_body = True
            continue
        if not in_body:
            continue
        f = raw.split()
        if len(f) != 5:
            continue
        try:
            v = float(f[0].replace("D", "E").replace("d", "e"))
            i, j, k, l = map(int, f[1:])
        except ValueError:
            continue
        if i == j == k == l == 0:
            ecore = v
        elif k == 0 and l == 0 and i > 0 and j > 0:
            i -= 1; j -= 1
            h1[i, j] = h1[j, i] = v
        elif min(i, j, k, l) > 0:
            i -= 1; j -= 1; k -= 1; l -= 1
            perms = {
                (i,j,k,l),(j,i,k,l),(i,j,l,k),(j,i,l,k),
                (k,l,i,j),(l,k,i,j),(k,l,j,i),(l,k,j,i),
            }
            for p in perms:
                eri[p] = v
    return FCIDumpData(norb, nelec, ms2, h1, eri, ecore)


def sector_determinants(norb: int, nalpha: int, nbeta: int) -> np.ndarray:
    dets = []
    for aa in combinations(range(norb), nalpha):
        amask = sum(1 << p for p in aa)
        for bb in combinations(range(norb), nbeta):
            bmask = sum(1 << (norb + p) for p in bb)
            dets.append(amask | bmask)
    return np.asarray(dets, dtype=np.int64)


def _pop_below(det: int, orb: int) -> int:
        return bin(det & ((1 << orb) - 1)).count("1")


def annihilate(det: int, orb: int) -> Tuple[int, float]:
    if not (det >> orb) & 1:
        return det, 0.0
    sign = -1.0 if (_pop_below(det, orb) & 1) else 1.0
    return det ^ (1 << orb), sign


def create(det: int, orb: int) -> Tuple[int, float]:
    if (det >> orb) & 1:
        return det, 0.0
    sign = -1.0 if (_pop_below(det, orb) & 1) else 1.0
    return det ^ (1 << orb), sign


def apply_excitation(det: int, remove: Sequence[int], add: Sequence[int]) -> Tuple[int, float]:
    """Apply T = a^+_{r1}...a^+_{rk} a_{ak}...a_{a1}.

    `remove` and `add` are supplied in ascending canonical order.  The rightmost
    annihilator acts first, hence annihilations are applied ascending, while
    creations are applied in reverse order.
    """
    x, sgn = int(det), 1.0
    for q in remove:
        x, s = annihilate(x, int(q)); sgn *= s
        if s == 0.0:
            return det, 0.0
    for p in reversed(add):
        x, s = create(x, int(p)); sgn *= s
        if s == 0.0:
            return det, 0.0
    return x, sgn


def _spatial_and_spin(so: int, norb: int) -> Tuple[int, int]:
    return so % norb, so // norb


def _hso(data: FCIDumpData, p: int, q: int) -> float:
    op, sp = _spatial_and_spin(p, data.norb)
    oq, sq = _spatial_and_spin(q, data.norb)
    return float(data.h1[op, oq]) if sp == sq else 0.0


def _vso(data: FCIDumpData, p: int, r: int, q: int, s: int) -> float:
    op, sp = _spatial_and_spin(p, data.norb)
    or_, sr = _spatial_and_spin(r, data.norb)
    oq, sq = _spatial_and_spin(q, data.norb)
    os, ss = _spatial_and_spin(s, data.norb)
    if sp != sq or sr != ss:
        return 0.0
    # Same convention as the validated reconstruction of the public FCIDUMP.
    return float(data.eri[op, oq, or_, os])


def _barv(data: FCIDumpData, p: int, r: int, q: int, s: int) -> float:
    return _vso(data, p, r, q, s) - _vso(data, p, r, s, q)


def sector_hamiltonian(data: FCIDumpData, nalpha: int, nbeta: int) -> Tuple[np.ndarray, np.ndarray]:
    basis = sector_determinants(data.norb, nalpha, nbeta)
    index = {int(d): i for i, d in enumerate(basis)}
    n = len(basis)
    H = np.zeros((n, n), dtype=float)
    nso = 2 * data.norb

    for col, d0 in enumerate(basis):
        d = int(d0)
        occ = [p for p in range(nso) if (d >> p) & 1]
        vir = [p for p in range(nso) if not ((d >> p) & 1)]

        diag = data.ecore + sum(_hso(data, p, p) for p in occ)
        diag += 0.5 * sum(_barv(data, p, q, p, q) for p in occ for q in occ)
        H[col, col] = diag

        for q in occ:
            common = [j for j in occ if j != q]
            for p in vir:
                if p // data.norb != q // data.norb:
                    continue
                coeff = _hso(data, p, q) + sum(_barv(data, p, j, q, j) for j in common)
                if abs(coeff) < 1e-15:
                    continue
                d2, sgn = apply_excitation(d, (q,), (p,))
                row = index.get(d2)
                if row is not None:
                    H[row, col] += coeff * sgn

        for q, s in combinations(occ, 2):
            for p, r in combinations(vir, 2):
                coeff = _barv(data, p, r, q, s)
                if abs(coeff) < 1e-15:
                    continue
                d2, sgn = apply_excitation(d, (q, s), (p, r))
                row = index.get(d2)
                if row is not None:
                    H[row, col] += coeff * sgn

    defect = float(np.max(np.abs(H - H.T)))
    if defect > 1e-9:
        raise RuntimeError("Hamiltonian Hermiticity defect %.3e" % defect)
    H = 0.5 * (H + H.T)
    return basis, H


def splus_matrix(norb: int, nalpha: int, nbeta: int) -> Tuple[np.ndarray, np.ndarray, sp.csr_matrix]:
    """S_+ map from (nalpha,nbeta) to (nalpha+1,nbeta-1)."""
    if nbeta < 1 or nalpha >= norb:
        raise ValueError("S_+ target sector does not exist")
    src = sector_determinants(norb, nalpha, nbeta)
    dst = sector_determinants(norb, nalpha + 1, nbeta - 1)
    didx = {int(d): i for i, d in enumerate(dst)}
    rows, cols, vals = [], [], []
    for col, d0 in enumerate(src):
        d = int(d0)
        for p in range(norb):
            x, s1 = annihilate(d, norb + p)
            if s1 == 0.0:
                continue
            x, s2 = create(x, p)
            if s2 == 0.0:
                continue
            row = didx.get(x)
            if row is not None:
                rows.append(row); cols.append(col); vals.append(s1 * s2)
    S = sp.csr_matrix((vals, (rows, cols)), shape=(len(dst), len(src)), dtype=float)
    return src, dst, S


def spin_square_expectation(psi_ms: np.ndarray, splus: sp.csr_matrix, ms: float = 0.5) -> float:
    psi_ms = np.asarray(psi_ms, dtype=complex)
    norm = float(np.vdot(psi_ms, psi_ms).real)
    if norm <= 0.0:
        raise ValueError("zero state")
    y = splus @ psi_ms
    return float((np.vdot(y, y).real / norm) + ms * (ms + 1.0))


def highest_weight_subspace(norb: int, nalpha: int, nbeta: int, target_s: float = 0.5,
                            rcond: float = 1e-12) -> Tuple[np.ndarray, sp.csr_matrix]:
    ms = 0.5 * (nalpha - nbeta)
    if abs(ms - target_s) > 1e-12:
        raise ValueError("Highest-weight projector requires M_S=S; got M_S=%g, S=%g" % (ms, target_s))
    _, _, Splus = splus_matrix(norb, nalpha, nbeta)
    U = scipy.linalg.null_space(Splus.toarray(), rcond=rcond)
    defect = np.linalg.norm(Splus @ U)
    if defect > 1e-9:
        raise RuntimeError("S_+ null-space defect %.3e" % defect)
    return U, Splus


def pure_spin_hamiltonian(data: FCIDumpData, nalpha: int, nbeta: int, target_s: float = 0.5):
    basis, Hms = sector_hamiltonian(data, nalpha, nbeta)
    U, Splus = highest_weight_subspace(data.norb, nalpha, nbeta, target_s=target_s)
    Hs = U.conj().T @ Hms @ U
    Hs = 0.5 * (Hs + Hs.conj().T)
    return basis, Hms, U, Hs, Splus


def hf_determinant(norb: int, nalpha: int, nbeta: int) -> int:
    return sum(1 << p for p in range(nalpha)) | sum(1 << (norb + p) for p in range(nbeta))


def excitation_specs_from_hf(norb: int, nalpha: int, nbeta: int, max_rank: int = 3):
    """All HF S/D/T excitations that preserve N_alpha and N_beta.

    These preserve M_S.  Spin purity is enforced separately by projection to the
    highest-weight S=1/2 subspace.
    """
    det = hf_determinant(norb, nalpha, nbeta)
    nso = 2 * norb
    occ = [p for p in range(nso) if (det >> p) & 1]
    vir = [p for p in range(nso) if not ((det >> p) & 1)]
    specs = []
    for rank in range(1, max_rank + 1):
        for rem in combinations(occ, rank):
            nar = sum(p < norb for p in rem)
            for add in combinations(vir, rank):
                naa = sum(p < norb for p in add)
                if nar == naa:
                    specs.append((tuple(rem), tuple(add)))
    return specs


def excitation_generator(basis: np.ndarray, remove: Sequence[int], add: Sequence[int]) -> sp.csr_matrix:
    """Anti-Hermitian A=T-T^dagger in a determinant basis."""
    index = {int(d): i for i, d in enumerate(basis)}
    rows, cols, vals = [], [], []
    for col, d0 in enumerate(basis):
        d2, sgn = apply_excitation(int(d0), remove, add)
        if sgn == 0.0:
            continue
        row = index.get(d2)
        if row is None:
            continue
        rows.extend((row, col)); cols.extend((col, row)); vals.extend((sgn, -sgn))
    A = sp.csr_matrix((vals, (rows, cols)), shape=(len(basis), len(basis)), dtype=float)
    defect = sp.linalg.norm(A + A.T)
    if defect > 1e-10:
        raise RuntimeError("generator anti-Hermiticity defect %.3e" % defect)
    return A
