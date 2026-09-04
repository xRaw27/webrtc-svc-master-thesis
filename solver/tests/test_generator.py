"""M2 acceptance: validity, determinism, exact JSON round trip."""

import json

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sfu_alloc.generator import from_json, random_instance, to_json
from sfu_alloc.instance import instances_equal, weights_from_preferences


@given(seed=st.integers(0, 2**32 - 1), n=st.integers(2, 8))
def test_generated_instances_are_valid_and_round_trip(seed, n):
    """Construction runs full validation; the JSON round trip is exact."""
    inst = random_instance(seed, n)
    assert instances_equal(inst, from_json(to_json(inst)))


def test_identical_seed_gives_identical_instance():
    assert instances_equal(random_instance(42, n=6), random_instance(42, n=6))
    assert not instances_equal(random_instance(42, n=6), random_instance(43, n=6))


def test_accepts_generator_object():
    a = random_instance(np.random.default_rng(7), n=4)
    b = random_instance(np.random.default_rng(7), n=4)
    assert instances_equal(a, b)


def test_covers_history_and_preference_modes():
    insts = [random_instance(seed, n=4) for seed in range(40)]
    assert {inst.history is None for inst in insts} == {True, False}

    def has_default_preferences(inst):
        expected = weights_from_preferences(np.ones((inst.n, inst.n)), inst.r, inst.L)
        return bool(np.allclose(inst.w, expected, rtol=0.0, atol=1e-12))

    assert {has_default_preferences(inst) for inst in insts} == {True, False}


def test_weights_are_canonical_for_default_preferences():
    """In all-default-preference mode w must equal the canonical all-ones form."""
    inst = random_instance(11, n=5, p_default_prefs=1.0)
    expected = weights_from_preferences(np.ones((5, 5)), inst.r, inst.L)
    np.testing.assert_array_equal(inst.w, expected)


def test_json_document_shape():
    inst = random_instance(1, n=3)
    doc = json.loads(to_json(inst))
    assert doc["version"] == 1
    for i in range(3):
        # ladders are ragged: exactly L[i] + 1 entries, no NaN padding
        assert len(doc["r"][i]) == int(inst.L[i]) + 1


def test_from_json_rejects_unknown_version():
    doc = json.loads(to_json(random_instance(2, n=3)))
    doc["version"] = 99
    with pytest.raises(ValueError, match="version"):
        from_json(json.dumps(doc))


def test_rejects_bad_arguments():
    with pytest.raises(ValueError, match="n must be >= 2"):
        random_instance(0, n=1)
    with pytest.raises(ValueError, match="base ladder"):
        random_instance(0, n=3, L_choices=(6,))


def test_generates_five_level_svc_ladders():
    """L3T3 SVC ladders (5 levels) are reachable with the default pool."""
    for seed in range(10):
        inst = random_instance(seed, n=3, L_choices=(4, 5))
        assert np.all(inst.L >= 4)  # only ladders long enough are eligible
    inst = random_instance(0, n=3, L_choices=(5,))
    assert np.all(inst.L == 5)
    assert inst.r.shape == (3, 6)
    assert np.all(np.isfinite(inst.r))
