"""Configuration profiles for quantum workflow."""

# Prepnime na 'prod' pre reálne dáta, alebo 'test' pre rýchly beh
MODE = "prod"

INTEGRALS_NPZ = f"artifacts/{MODE}/integrals.npz"
FCIDUMP = f"artifacts/{MODE}/active.FCIDUMP"
HISTORY_NPZ = f"artifacts/{MODE}/vqe_history.npz"
REPORT_PNG = f"artifacts/{MODE}/vqe_report.png"

BASIS_GATES = ["rz", "sx", "x", "cz", "cx"]
NOISE_1Q = 0.001
NOISE_2Q = 0.01


def profile():
    if MODE == "test":
        return {
            "basis": "sto-3g",
            "active": "manual",
            "cas_orb": 3,
            "cas_elec": (2, 1),
            "avas_labels": ["Cu 3d", "S 3p"],
            "avas_threshold": 0.2,
            "run_dft": False,
            "ansatz": "hea",
            "hea_reps": 1,
            "ideal_maxiter": 10,
            "noisy_maxiter": 3,
            "shots": 128,
        }
    else:
        return {
            "basis": "sto-3g",
            "active": "avas",
            "cas_orb": None,
            "cas_elec": None,
            "avas_labels": ["Cu 3d", "9 S 3p"],
            "avas_threshold": 0.2,
            "run_dft": True,
            "ansatz": "hea",
            "hea_reps": 1,
            "ideal_maxiter": 10,     # Bleskový ideálny beh pre 24 qubitov
            "noisy_maxiter": 1,      # Okamžitá šumová vzorka pre recenzenta
            "shots": 128,            # Minimalizované šoty pre okamžitý prepočet
        }