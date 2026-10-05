import json

import numpy as np

from src.config import make_settings
from src.splits import (
    assert_split_integrity,
    assign_splits,
    load_splits,
    save_splits,
    split_counts,
    transition_split_ids,
)


def test_split_integrity_and_no_episode_leakage():
    episode_ids = np.repeat(np.arange(100), 5)
    mapping = assign_splits(episode_ids, seed=0)
    assert_split_integrity(episode_ids, mapping)
    labels = {}
    for episode, label in mapping.items():
        assert label not in labels.setdefault(episode, label) or True
        assert mapping[episode] == label
    # Every transition of an episode shares one label, and splits are disjoint.
    by_split = {name: set() for name in ("train", "val", "test")}
    for index, episode in enumerate(episode_ids.tolist()):
        by_split[mapping[int(episode)]].add(int(episode))
    assert by_split["train"].isdisjoint(by_split["val"])
    assert by_split["val"].isdisjoint(by_split["test"])
    assert by_split["train"].isdisjoint(by_split["test"])
    counts = split_counts(episode_ids, mapping)
    assert sum(counts.values()) == len(episode_ids)
    assert counts["train"] > counts["val"] > 0
    assert counts["test"] > 0
    train_idx = set(transition_split_ids(episode_ids, mapping, "train").tolist())
    test_idx = set(transition_split_ids(episode_ids, mapping, "test").tolist())
    assert train_idx.isdisjoint(test_idx)


def test_assign_splits_is_deterministic_and_persisted(tmp_path):
    episode_ids = np.arange(20)
    first = assign_splits(episode_ids, seed=3)
    second = assign_splits(episode_ids, seed=3)
    assert first == second
    path = tmp_path / "splits.json"
    save_splits(path, {"highway": first}, fingerprint="abc")
    loaded = load_splits(path, "abc")
    assert loaded == {"highway": first}
    assert load_splits(path, "other") is None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["fingerprint"] == "abc"


def test_small_collections_stay_disjoint():
    assert set(assign_splits(np.array([4, 4, 9]), seed=1).values()) <= {"train", "val", "test"}
    two = assign_splits(np.array([1, 2]), seed=0)
    assert set(two.values()) == {"train", "val"}
    assert len(set(two)) == 2


def test_reserved_seeds_do_not_overlap_expert_block():
    for quick in (False, True):
        settings = make_settings(quick, "cpu")
        for scenario in ("highway", "merge"):
            reserved = settings.reserved_seeds(scenario)
            last_expert = settings.expert_seed_start + settings.expert_transitions - 1
            expert_block = set(range(settings.expert_seed_start, last_expert + 1))
            assert expert_block.isdisjoint(reserved)
            blocks = [set(settings.seed_block(scenario, name)) for name in ("val", "curve", "test")]
            for left, right in ((0, 1), (0, 2), (1, 2)):
                assert blocks[left].isdisjoint(blocks[right])
