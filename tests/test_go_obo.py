"""Pre-GPU units for GO ontology helpers (bioreason_pro/go_obo.py).

Uses a tiny synthetic OBO (fast, deterministic, no gitignored data). An optional smoke test runs
against the real data/ files when present (they are gitignored, so skipped in CI).
"""

from pathlib import Path

import pytest

from bioreason_pro import go_obo

TINY_OBO = """format-version: 1.2

[Term]
id: GO:0008150
name: biological_process
namespace: biological_process

[Term]
id: GO:0000003
name: reproduction
namespace: biological_process
is_a: GO:0008150 ! biological_process

[Term]
id: GO:0019953
name: sexual reproduction
namespace: biological_process
is_a: GO:0000003 ! reproduction
"""


def test_load_go_ancestors_true_path(tmp_path):
    obo = tmp_path / "tiny.obo"
    obo.write_text(TINY_OBO)
    anc = go_obo.load_go_ancestors(obo)
    assert anc["GO:0019953"] == {"GO:0000003", "GO:0008150"}   # transitive ancestors
    assert anc["GO:0000003"] == {"GO:0008150"}
    assert anc["GO:0008150"] == set()                          # root has none


def test_load_ia_weights(tmp_path):
    ia_file = tmp_path / "IA.txt"
    ia_file.write_text("GO:0000001\t0.0\nGO:0000002\t3.10\n\nGO:0000003 3.44\n")
    ia = go_obo.load_ia_weights(ia_file)
    assert ia["GO:0000002"] == pytest.approx(3.10)
    assert ia["GO:0000003"] == pytest.approx(3.44)   # space-separated also parsed
    assert ia["GO:0000001"] == 0.0


REAL_OBO = Path("data/go-basic.obo")
REAL_IA = Path("data/IA.txt")


@pytest.mark.skipif(not REAL_IA.exists(), reason="data/IA.txt not present (gitignored)")
def test_real_ia_loads():
    ia = go_obo.load_ia_weights(REAL_IA)
    assert len(ia) > 10000
    assert all(isinstance(v, float) for v in list(ia.values())[:100])
