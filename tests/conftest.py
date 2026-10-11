import pytest

from rdgen.data import load


@pytest.fixture(scope="session")
def synth():
    items = load("synthetic", n_source=160, n_target=80, seed=3)
    return {
        "all": items,
        "train": [i for i in items if i.split == "train"],
        "test_id": [i for i in items if i.split == "test_id"],
        "shift": [i for i in items if i.split == "test_shift"],
    }
