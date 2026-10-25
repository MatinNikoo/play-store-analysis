import numpy as np
import pandas as pd
import pytest

from study import features as F, ranking


def tiny():
    # developer A: two apps on day 0 (Tools, Tools), one on day 10 (Puzzle), one on day 50 (Tools)
    # developer B: one app
    d = pd.Timestamp("2018-01-01")
    return pd.DataFrame({
        "app_id": ["com.a.one", "com.a.two", "com.a.three", "com.a.four", "com.b.x"],
        "name": ["One", "Two 2", "Tres é", "", "X"],
        "dev": ["A", "A", "A", "A", "B"],
        "Category": ["Tools", "Tools", "Puzzle", "Tools", "Tools"],
        "released": [d, d, d + pd.Timedelta(days=10), d + pd.Timedelta(days=50), d],
        "installs": [10, 200_000, 50, 5, 1],
    })


def test_developer_history_by_hand():
    f = F.add_launch_features(tiny())
    assert f["dev_prior_apps"].tolist() == [0, 0, 2, 3, 0]
    assert f["dev_same_day"].tolist() == [1, 1, 0, 0, 0]
    assert f["dev_prior_30d"].tolist() == [0, 0, 2, 0, 0]          # day 10 is outside day 50's 30 day window
    assert f["dev_days_active"].tolist() == [0, 0, 10, 50, 0]
    assert f["dev_prior_cats"].tolist() == [0, 0, 1, 2, 0]
    assert f["dev_prior_same_cat"].tolist() == [0, 0, 0, 2, 0]
    assert f["pkg_prior"].tolist() == [0, 0, 2, 3, 0]              # com.a prefix
    assert f["name_words"].tolist() == [1, 2, 2, 0, 1]
    assert f["name_nonascii"].tolist() == [0, 0, 1, 0, 0]
    assert f["name_digit"].tolist() == [0, 1, 0, 0, 0]


def test_order_of_rows_does_not_matter():
    a = F.add_launch_features(tiny())
    b = F.add_launch_features(tiny().iloc[::-1].reset_index(drop=True)).iloc[::-1].reset_index(drop=True)
    pd.testing.assert_frame_equal(a[F.LAUNCH], b[F.LAUNCH])


def test_leaky_track_record_uses_earlier_apps_only():
    f = F.add_leaky_features(F.add_launch_features(tiny()).assign(
        hit100k=lambda d: (d["installs"] >= 100_000).astype(int),
        scraped=pd.Timestamp("2021-06-15"), updated=pd.Timestamp("2021-01-01")))
    assert np.isnan(f.loc[0, "dev_prior_hit_share"])
    assert f.loc[2, "dev_prior_hit_share"] == pytest.approx(0.5)
    assert f.loc[3, "dev_prior_hit_share"] == pytest.approx(1 / 3)
    assert f.loc[2, "dev_prior_max_installs"] == pytest.approx(np.log10(200_001))


def test_package_prefix():
    s = pd.Series(["com.andromo.dev1.app", "br.com.loja.app", "io.kodular.x.y", "jp.co.abc.d"])
    assert F.pkg_prefix(s).tolist() == ["com.andromo", "br.com.loja", "io.kodular", "jp.co.abc"]


def test_top5_rate_is_five_percent_per_month(built):
    g = built.groupby(built["released"].dt.to_period("M"))["top5"].agg(["mean", "size"])
    assert g.loc[g["size"] >= 60, "mean"].between(0.03, 0.08).all()


def test_split_integrity(built):
    train, test, later = ranking.split(built)
    assert train["released"].max() < test["released"].min()
    assert test["released"].max() < later["released"].min()
    assert not set(train["app_id"]) & set(test["app_id"])
    assert not set(test["app_id"]) & set(later["app_id"])
    assert len(train) + len(test) + len(later) == built["released"].dt.year.isin([2018, 2019, 2020]).sum()


def test_cleaning_drops_missing_and_duplicates(raw):
    from study import data
    df = data.clean(raw)
    c = df.attrs["counts"]
    assert c["raw_rows"] == len(raw)
    assert c["after_dropping_missing"] == len(raw) - 5
    assert c["after_dropping_duplicates"] == len(raw) - 7
    assert df["app_id"].is_unique
