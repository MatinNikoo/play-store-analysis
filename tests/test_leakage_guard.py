"""The launch-time model must never see anything that was only known after launch."""

import json

import numpy as np
import pandas as pd
import pytest

from study import data, features as F, milestones as MS, prepare, ranking
from tests.fixture import make_raw


def test_launch_lists_pass_the_guard():
    assert F.check_launch_only(F.LAUNCH)
    assert F.check_launch_only(F.STRICT)
    assert set(F.STRICT) <= set(F.LAUNCH)


@pytest.mark.parametrize("feat", F.SNAPSHOT + F.POST + F.LEAKY_DEV + ["not_a_feature"])
def test_guard_rejects_post_launch_features(feat):
    with pytest.raises(ValueError):
        F.check_launch_only(F.LAUNCH + [feat])


def test_old_model_fails_the_guard():
    # the old model used monetization and size read off the 2021 snapshot
    with pytest.raises(ValueError):
        F.check_launch_only(F.OLD)


def test_every_used_feature_is_in_the_table():
    used = set(F.LAUNCH + F.OLD + F.SNAPSHOT + F.POST + F.LEAKY_DEV)
    assert used <= set(F.AVAILABILITY)
    for f in F.LAUNCH:
        assert F.known(f) == "launch"
    for f in F.SNAPSHOT:
        assert F.known(f) == "snapshot"


def test_ranking_launch_fit_calls_the_guard(built):
    train = built[built["released"].dt.year == 2018].reset_index(drop=True)
    with pytest.raises(ValueError):
        ranking.launch_fit("gbm", train, [train], F.LAUNCH + ["iap"], "top5")


def test_milestone_inputs_are_launch_time():
    for inp, feat in MS.INPUT_SOURCE.items():
        assert F.known(feat) == "launch", inp
    terms = MS.make_terms(["Tools", "Puzzle"], "Tools")
    assert {t["input"] for t in terms} <= set(MS.INPUT_SOURCE)


def test_exported_model_only_reads_launch_inputs(pipeline):
    model = json.loads((pipeline / "models" / "milestones.json").read_text())
    for t in model["terms"]:
        assert F.known(MS.INPUT_SOURCE[t["input"]]) == "launch"


def _launch_features(raw):
    df = data.clean(raw)
    for col in ["Category", "Content Rating", "model"]:
        df[col] = df[col].astype("category")
    return F.add_launch_features(df).set_index("app_id")


def test_features_ignore_the_future():
    """Change everything about apps launched after a cutoff (installs, flags, names, even delete some).
    Launch features of earlier apps must not move."""
    raw = make_raw(n=3000, devs=200, seed=3)
    base = _launch_features(raw)
    cut = pd.Timestamp("2019-01-01")
    rel = pd.to_datetime(raw["Released"], format="%b %d, %Y", errors="coerce")
    later = rel >= cut
    changed = raw.copy()
    changed.loc[later, "Maximum Installs"] = 10 ** 8
    changed.loc[later, "In App Purchases"] = True
    changed.loc[later, "App Name"] = "Totally Different Name 123"
    rng = np.random.default_rng(0)
    changed = changed[~(later & (rng.random(len(changed)) < 0.3))]
    after = _launch_features(changed)
    early = base.index[base["released"] < cut]
    pd.testing.assert_frame_equal(base.loc[early, F.LAUNCH], after.loc[early, F.LAUNCH], check_categorical=False)


def test_features_ignore_the_app_s_own_outcome_and_listing():
    """Shuffle each app's own installs, ratings, flags, size and update fields across apps. Launch features
    don't read any of them, so nothing may move (release date, id, developer, category and name stay put)."""
    raw = make_raw(n=3000, devs=200, seed=5)
    base = _launch_features(raw)
    changed = raw.copy()
    rng = np.random.default_rng(1)
    for col in ["Maximum Installs", "Rating", "Rating Count", "Free", "Price", "Size", "Minimum Android",
                "Content Rating", "Privacy Policy", "Developer Website", "Ad Supported", "In App Purchases",
                "Editors Choice", "Last Updated"]:
        ok = changed[col].notna().to_numpy()
        vals = changed.loc[ok, col].to_numpy()
        changed.loc[ok, col] = vals[rng.permutation(len(vals))]
    after = _launch_features(changed)
    assert set(base.index) == set(after.index)
    pd.testing.assert_frame_equal(base[F.LAUNCH], after.loc[base.index, F.LAUNCH], check_categorical=False)


def test_outcome_only_compares_apps_from_the_same_month(built):
    df = built[["released", "installs"]].copy()
    jan = df["released"].dt.to_period("M") == pd.Period("2019-01", "M")
    before = F.top_in_month(df, 0.05)[jan]
    df.loc[~jan, "installs"] = 0
    assert (F.top_in_month(df, 0.05)[jan] == before).all()


def test_docs_table_matches_code():
    """docs/method.md prints the availability table. It has to say the same thing as the code."""
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "docs" / "method.md").read_text()
    rows = {}
    for line in text.splitlines():
        if line.startswith("| `"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            rows[cells[0].strip("`")] = cells
    assert set(rows) == set(F.AVAILABILITY)
    for f, (src, k, rev, note) in F.AVAILABILITY.items():
        c = rows[f]
        assert (c[1], c[2], c[3], c[6]) == (src, k, rev, note), f
        assert (c[4] == "yes") == (f in F.OLD) and (c[5] == "yes") == (f in F.LAUNCH), f
