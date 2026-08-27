"""Literature T1 'blue-copper' active-site model (first coordination sphere).

Idealised model of the oxidised Cu(II) type-1 site found in blue-copper
proteins (plastocyanin / azurin-like), used as a stand-in because no
experimental XYZ of the full laccase enzyme is available.

Donor set (N2S2, distorted tetrahedral):
    2x His  -> NH3     Cu-N ~ 2.00 A
    1x Cys  -> SH-     Cu-S ~ 2.15 A   (short, covalent; defines the site)
    1x Met  -> SH2     Cu-S ~ 2.88 A   (long axial)

Fidelity upgrades (swap in when available): His -> imidazole,
Cys -> CH3S-, Met -> (CH3)2S, or replace the whole block with coordinates
extracted from a laccase crystal structure.

Oxidised Cu(II) is d9 -> one unpaired electron -> spin doublet.
"""

# element  x       y       z    (Angstrom)
T1_MODEL_ATOMS = """
Cu   0.000   0.000   0.000
N    2.000   0.000   0.000
H    2.340   0.900   0.200
H    2.340  -0.620   0.740
H    2.340  -0.280  -0.800
N   -1.000   1.732   0.000
H   -1.280   2.200   0.800
H   -1.550   2.120  -0.620
H   -0.120   2.260   0.040
S   -0.600  -1.100   1.750
H   -1.420  -1.580   2.120
S    0.200  -0.500  -2.830
H    0.920  -1.180  -3.200
H   -0.540  -0.920  -3.280
"""

CHARGE = 1     # Cu(2+) + Cys(SH-, -1) + neutral N/S ligands
SPIN   = 1     # 2S = 1 unpaired electron (Cu(II) d9 doublet)

def t1_model():
    """Return (atom_string, charge, spin) for the T1 model site."""
    return T1_MODEL_ATOMS.strip(), CHARGE, SPIN
