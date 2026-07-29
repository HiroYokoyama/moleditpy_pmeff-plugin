"""
Cycloalkyne geometry: does a strained ring bend the alkyne by the right amount?

A nominally linear sp center in a small ring has to bend, and how much of the
ring strain it absorbs versus the sp3 linker is set by _K_ANGLE_LINEAR_SP.
That constant is empirical (see its comment in forcefield.py); these tests pin
it against experiment so a re-fit is a deliberate act rather than a drift.

Reference C-C#C angles are gas-phase electron diffraction / high-level values:
cyclooctyne 158.5 deg, cyclononyne 163.5, cyclodecyne 170.0. Acyclic alkynes
are linear. Tolerances are loose (a few degrees) because these are force-field
geometries, not spectroscopy -- the point is the trend and the regime, not the
last decimal.

These rings have several conformers whose C-C#C angle differs by >10 deg, so
every comparison here is against the GLOBAL minimum: relax from a dozen random
starts and keep the lowest-energy result. Fitting to a single MMFF-seeded
start instead lands in a higher-energy, more-bent conformer and inverts the
calibration curve entirely -- see the _K_ANGLE_LINEAR_SP comment.
"""

import math
import sys
import unittest

import numpy as np

sys.path.insert(0, __file__.rsplit("tests", 1)[0])

rdkit = None
try:
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom

    rdkit = Chem
except ImportError:  # pragma: no cover - exercised only without rdkit
    pass

from pmeff_plugin import forcefield as ff  # noqa: E402


def _angle_deg(x, i, j, k):
    a = x[i] - x[j]
    b = x[k] - x[j]
    cos = float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))


#: Medium rings have several conformers whose C-C#C angle differs by >10 deg,
#: and which one a single embedding lands in is arbitrary. Experiment measures
#: the global minimum, so optimize from a handful of starts and keep the
#: lowest-energy result -- otherwise the assertions pin a random conformer.
_N_STARTS = 12


def _embed(smiles, seed=0xC0FFEE):
    """Crude but instant starting geometry.

    ETKDG's experimental torsion preferences are pathologically slow on
    medium-ring alkynes -- cyclooctyne took 15 s and cyclononyne 74 s, and
    both needed several seed retries. Plain distance geometry embeds all of
    them first-try in under 10 ms. That is also the more honest starting
    point for these tests: what is under test is where PMEFF relaxes the ring
    to, not how good RDKit's conformer generator is, and PMEFF converges from
    the crude geometry in ~0.2 s.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = rdDistGeom.EmbedParameters()
    params.useRandomCoords = True
    params.useExpTorsionAnglePrefs = False
    params.useBasicKnowledge = False
    params.ignoreSmoothingFailures = True
    params.maxIterations = 200
    params.randomSeed = seed
    if rdDistGeom.EmbedMolecule(mol, params) == 0:
        return mol
    return None


#: Embedding + MMFF cleanup + a full PMEFF minimize costs a couple of seconds
#: per molecule, and most tests below want the same few geometries. Cache them
#: so the suite pays for each molecule once instead of once per assertion.
_GEOMETRY_CACHE = {}


def _optimized_alkyne_angles(smiles):
    """Return (avg C-C#C angle, mol, optimized coords) after a PMEFF minimize."""
    if smiles in _GEOMETRY_CACHE:
        return _GEOMETRY_CACHE[smiles]
    result = _compute_alkyne_angles(smiles)
    _GEOMETRY_CACHE[smiles] = result
    return result


def _compute_alkyne_angles(smiles):
    best_e, best_mol, best_coords = None, None, None
    for seed in range(_N_STARTS):
        mol = _embed(smiles, seed=0xC0FFEE + seed * 7919)
        if mol is None:
            continue
        topo = ff.topology_from_rdkit(mol)
        start = np.array(mol.GetConformer().GetPositions(), dtype=float)
        # The optimizer exits on f_tol; this is only a ceiling, so keep it
        # modest -- these molecules converge in well under 500 steps.
        coords, _res = ff.optimize(start, topo, max_iter=500)
        energy, _g = ff.energy_and_gradient(coords, topo)
        if best_e is None or energy < best_e:
            best_e, best_mol, best_coords = energy, mol, coords
    if best_mol is None:
        return None, None, None
    mol, coords = best_mol, best_coords

    triple = [b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.TRIPLE]
    c1, c2 = triple[0].GetBeginAtomIdx(), triple[0].GetEndAtomIdx()
    n1 = [a.GetIdx() for a in mol.GetAtomWithIdx(c1).GetNeighbors() if a.GetIdx() != c2]
    n2 = [a.GetIdx() for a in mol.GetAtomWithIdx(c2).GetNeighbors() if a.GetIdx() != c1]
    angles = []
    if n1:
        angles.append(_angle_deg(coords, n1[0], c1, c2))
    if n2:
        angles.append(_angle_deg(coords, n2[0], c2, c1))
    return sum(angles) / len(angles), mol, coords


@unittest.skipIf(rdkit is None, "rdkit not installed")
class TestCycloalkyneBending(unittest.TestCase):
    def test_cyclooctyne_bends_to_about_158_degrees(self):
        avg, _mol, _x = _optimized_alkyne_angles("C1CCCC#CCC1")
        self.assertIsNotNone(avg, "cyclooctyne failed to embed")
        self.assertAlmostEqual(avg, 158.5, delta=4.0)

    def test_cyclooctyne_is_neither_pinned_nor_collapsed(self):
        """Guard both failure modes: a too-stiff sp bend leaves it above 162
        deg, a too-floppy one (k_sp = 12) collapses it to ~146."""
        avg, _mol, _x = _optimized_alkyne_angles("C1CCCC#CCC1")
        self.assertGreater(avg, 150.0, "alkyne over-bent: sp bend too floppy")
        self.assertLess(avg, 162.0, "alkyne too straight: sp bend too stiff")

    def test_cyclononyne_bends_to_about_163_degrees(self):
        avg, _mol, _x = _optimized_alkyne_angles("C1CCCCC#CCC1")
        self.assertIsNotNone(avg, "cyclononyne failed to embed")
        self.assertAlmostEqual(avg, 163.5, delta=5.0)

    def test_cyclodecyne_bends_to_about_170_degrees(self):
        avg, _mol, _x = _optimized_alkyne_angles("C1CCCCCC#CCC1")
        self.assertIsNotNone(avg, "cyclodecyne failed to embed")
        self.assertAlmostEqual(avg, 170.0, delta=5.0)

    def test_bending_relaxes_monotonically_with_ring_size(self):
        """C8 more strained than C9 more strained than C10."""
        c8, _m, _x = _optimized_alkyne_angles("C1CCCC#CCC1")
        c9, _m, _x = _optimized_alkyne_angles("C1CCCCC#CCC1")
        c10, _m, _x = _optimized_alkyne_angles("C1CCCCCC#CCC1")
        self.assertLess(c8, c9)
        self.assertLess(c9, c10)

    def test_acyclic_alkyne_stays_linear(self):
        avg, _mol, _x = _optimized_alkyne_angles("CC#CC")
        self.assertAlmostEqual(avg, 180.0, delta=1.5)

    def test_terminal_alkyne_stays_linear(self):
        avg, _mol, _x = _optimized_alkyne_angles("C#CCCCCCC")
        self.assertAlmostEqual(avg, 180.0, delta=2.0)


@unittest.skipIf(rdkit is None, "rdkit not installed")
class TestRingStrainDistribution(unittest.TestCase):
    """Where the strain lands, not just how much."""

    def _ring_strain_split(self, smiles):
        avg, mol, coords = _optimized_alkyne_angles(smiles)
        self.assertIsNotNone(avg, f"{smiles} failed to embed")
        ring = list(mol.GetRingInfo().AtomRings()[0])
        n = len(ring)
        sp_dev = sp3_dev = 0.0
        for t in range(n):
            i, j, k = ring[t - 1], ring[t], ring[(t + 1) % n]
            is_sp = str(mol.GetAtomWithIdx(j).GetHybridization()) == "SP"
            dev = abs(_angle_deg(coords, i, j, k) - (180.0 if is_sp else 109.5))
            if is_sp:
                sp_dev += dev
            else:
                sp3_dev += dev
        return sp_dev, sp3_dev

    def test_the_alkyne_takes_a_real_share_of_the_strain(self):
        """It must bend, not stay pinned while the linker absorbs everything."""
        sp_dev, sp3_dev = self._ring_strain_split("C1CCCC#CCC1")
        self.assertGreater(sp_dev, 20.0)

    def test_the_sp3_linker_also_deforms(self):
        """The ring is strained everywhere, so the linker cannot sit at ideal
        tetrahedral while the alkyne absorbs literally all of it.

        No upper bound is asserted on the sp share: there is no reference
        value for how the strain *partitions*, only for the resulting angles,
        which the tests above pin against experiment directly.
        """
        _sp_dev, sp3_dev = self._ring_strain_split("C1CCCC#CCC1")
        self.assertGreater(sp3_dev, 5.0)


@unittest.skipIf(rdkit is None, "rdkit not installed")
class TestAlkyneGradient(unittest.TestCase):
    """A bent sp centre is exactly where a linear-bend gradient goes wrong."""

    def _fd_max_rel_error(self, smiles, h=1e-5):
        _avg, mol, _x = _optimized_alkyne_angles(smiles)
        self.assertIsNotNone(mol, f"{smiles} failed to embed")
        topo = ff.topology_from_rdkit(mol)
        x = np.array(mol.GetConformer().GetPositions(), dtype=float)
        x = x + np.random.default_rng(7).normal(0.0, 0.03, x.shape)
        _e, g_ana = ff.energy_and_gradient(x, topo)
        g_num = np.zeros_like(x)
        for i in range(x.shape[0]):
            for c in range(3):
                xp = x.copy()
                xp[i, c] += h
                xm = x.copy()
                xm[i, c] -= h
                g_num[i, c] = (
                    ff.energy_and_gradient(xp, topo)[0]
                    - ff.energy_and_gradient(xm, topo)[0]
                ) / (2 * h)
        return float((np.abs(g_ana - g_num) / np.maximum(np.abs(g_num), 1.0)).max())

    def test_gradient_is_analytic_for_a_strained_bent_alkyne(self):
        self.assertLess(self._fd_max_rel_error("C1CCCC#CCC1"), 1e-4)

    def test_gradient_is_analytic_near_the_linear_limit(self):
        """theta = pi is the cusp of the harmonic form; k*(1+cos) must be
        finite and correct there."""
        self.assertLess(self._fd_max_rel_error("CC#CC"), 1e-4)


if __name__ == "__main__":
    unittest.main()
