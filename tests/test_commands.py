"""The playodds commands beyond plain scoring. These run on the model that ships in models/."""

import json

import pytest

from playodds import __version__, cli, tables
from playodds import model as PM


@pytest.fixture(scope="module")
def model():
    return PM.load()


def run(capsys, *args):
    code = cli.main(list(args))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_category_matching_ignores_case_and_and(model):
    assert PM.resolve_category(model, "puzzle") == "Puzzle"
    assert PM.resolve_category(model, "food and drink") == "Food & Drink"
    assert PM.resolve_category(model, "  Music ") == "Music"


def test_category_typo_gets_a_suggestion(model):
    with pytest.raises(ValueError, match="Did you mean Simulation"):
        PM.resolve_category(model, "simulaton")
    with pytest.raises(ValueError, match="required"):
        PM.resolve_category(model, "")
    with pytest.raises(ValueError, match="unknown category 'zzzz'"):
        PM.resolve_category(model, "zzzz")


def test_flags_without_a_command_still_score(capsys):
    code, out, _ = run(capsys, "--category", "puzzle", "--name", "Block Drop", "--earlier-apps", "3")
    assert code == 0 and "Block Drop (Puzzle)" in out and "coefficient range" in out


def test_old_list_flag_and_no_arguments(capsys):
    code, out, _ = run(capsys, "--list-categories")
    assert code == 0 and "Simulation" in out.splitlines()
    with pytest.raises(SystemExit) as e:
        cli.main([])
    assert e.value.code == 0 and "usage: playodds" in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_score_csv_and_json_alias_agree(capsys):
    _, js, _ = run(capsys, "score", "--category", "Tools", "--json", "--comparables", "0")
    _, cs, _ = run(capsys, "score", "--category", "Tools", "--format", "csv")
    head, row = cs.strip().split("\n")
    rec = dict(zip(head.split(","), row.split(",")))
    p = json.loads(js)["milestones"]["10000"]["p"]
    assert abs(float(rec["p_10K"]) - p) < 1e-5
    assert json.loads(js)["comparables"] == []


def test_compare_copies_a_into_b_and_lists_differences(capsys):
    code, out, _ = run(capsys, "compare", "--a", "category=Puzzle", "name=Block Drop", "earlier_apps=3",
                       "--b", "category=Tools", "--format", "json")
    r = json.loads(out)
    assert code == 0
    assert r["b"]["app"]["name"] == "Block Drop" and r["b"]["app"]["earlier_apps"] == 3
    assert r["a"]["app"]["category"] == "Puzzle" and r["b"]["app"]["category"] == "Tools"
    _, txt, _ = run(capsys, "compare", "--a", "category=Puzzle", "--b", "category=Tools", "earlier_apps=40")
    assert "Inputs that differ: category (Puzzle vs Tools), earlier_apps (0 vs 40)" in txt
    assert "does not say what changing an input would do" in txt


def test_compare_matches_scoring_each_idea_alone(capsys, model):
    _, out, _ = run(capsys, "compare", "--a", "category=Puzzle", "--b", "category=Action", "--format", "json")
    r = json.loads(out)
    for side, cat in [("a", "Puzzle"), ("b", "Action")]:
        alone = PM.score({"category": cat}, model=model)
        assert r[side]["milestones"]["100000"]["p"] == pytest.approx(alone["milestones"]["100000"]["p"])


@pytest.mark.parametrize("args,msg", [(["category=Puzzle", "oops"], "KEY=VALUE"),
                                      (["category=Puzzle", "earlier_aps=2"], "Did you mean earlier_apps"),
                                      (["category=Puzzle", "earlier_apps=2.5"], "whole number"),
                                      (["category=Puzzle", "months_since_first_app=x"], "must be a number")])
def test_compare_bad_specs_give_plain_errors(capsys, args, msg):
    code, _, err = run(capsys, "compare", "--a", *args, "--b", "category=Tools")
    assert code == 2 and msg in err and "Traceback" not in err


IDEAS = ("id,Category,name,Earlier-Apps,months_since_first_app\n"
         "one,puzzle,Block Drop,3,14\n"
         "two,Tools,,,\n"
         "three,Simulation,Truck Driver,0,0\n")


def test_batch_scores_every_row_like_score_does(tmp_path, capsys, model):
    f = tmp_path / "ideas.csv"
    f.write_text(IDEAS)
    code, out, _ = run(capsys, "batch", str(f))
    rows = [dict(zip(out.splitlines()[0].split(","), l.split(","))) for l in out.strip().splitlines()[1:]]
    assert code == 0 and [r["id"] for r in rows] == ["one", "two", "three"]
    assert rows[0]["category"] == "Puzzle"
    one = PM.score({"category": "Puzzle", "name": "Block Drop", "earlier_apps": 3, "earlier_categories": 1,
                    "months_since_first_app": 14}, model=model)
    assert float(rows[0]["p_100K"]) == pytest.approx(one["milestones"]["100000"]["p"], rel=1e-5)
    assert float(rows[0]["installs_hi"]) == pytest.approx(one["installs_interval"][1], rel=1e-5)


def test_batch_sort_and_output_file(tmp_path, capsys):
    f, o = tmp_path / "ideas.csv", tmp_path / "scored.csv"
    f.write_text(IDEAS)
    code, out, err = run(capsys, "batch", str(f), "-o", str(o), "--sort", "10k")
    lines = o.read_text().strip().splitlines()
    cols = lines[0].split(",")
    p = [float(l.split(",")[cols.index("p_10K")]) for l in lines[1:]]
    assert out == "" and "scored 3 of 3" in err and p == sorted(p, reverse=True)


def test_batch_lists_every_bad_row_and_scores_nothing(tmp_path, capsys):
    f = tmp_path / "bad.csv"
    f.write_text("category,earlier_apps\nPuzzle,1\nPuzzel,2\nTools,-4\nTools,1.5\n")
    code, out, err = run(capsys, "batch", str(f))
    assert code == 2 and out == ""
    assert "3 of 4 rows" in err and "row 2" in err and "Did you mean Puzzle" in err
    assert "row 3" in err and "row 4" in err and "row 1" not in err


def test_batch_keep_going_marks_bad_rows(tmp_path, capsys):
    f = tmp_path / "bad.csv"
    f.write_text("category\nPuzzle\nNope\n")
    code, out, err = run(capsys, "batch", str(f), "--keep-going")
    lines = out.strip().splitlines()
    assert code == 0 and "unknown category" in lines[2] and lines[1].endswith(",")
    assert "scored 1 of 2" in err


def test_batch_file_problems(tmp_path, capsys):
    code, _, err = run(capsys, "batch", str(tmp_path / "missing.csv"))
    assert code == 2 and "cannot find" in err
    f = tmp_path / "nocat.csv"
    f.write_text("name\nA\n")
    code, _, err = run(capsys, "batch", str(f))
    assert code == 2 and "category column" in err
    code, _, err = run(capsys, "batch")
    assert code == 2 and "--template" in err


def test_batch_template_scores_cleanly(tmp_path, capsys):
    _, tpl, _ = run(capsys, "batch", "--template")
    f = tmp_path / "t.csv"
    f.write_text(tpl)
    code, out, _ = run(capsys, "batch", str(f), "--format", "json")
    assert code == 0 and len(json.loads(out)) == 2


def test_category_explorer_matches_market_table(capsys):
    t = tables.market_table()
    code, out, _ = run(capsys, "category", "simulation", "--format", "json")
    p = json.loads(out)
    row = t[t["category"] == "Simulation"].iloc[0]
    assert code == 0 and p["category"] == "Simulation"
    assert p["top10_share"] == pytest.approx(row["top10_share"])
    assert p["pooled_share_reached"]["100K"]["12_months"] == pytest.approx(row["npmle_12m_100K"])
    assert p["first_apps"]["model_average_at_12_months"]["10K"] == pytest.approx(row["first_app_12m_10K"], abs=1e-5)
    assert p["rank_top10_share"] == 1
    _, txt, _ = run(capsys, "category", "Business")
    assert "observed*" in txt and "number 47 of 48" in txt


def test_category_typo(capsys):
    code, _, err = run(capsys, "category", "Bussiness")
    assert code == 2 and "Did you mean Business" in err


def test_category_table_sorts(capsys):
    _, out, _ = run(capsys, "categories", "--table", "--sort", "top10", "--format", "csv")
    lines = out.strip().splitlines()
    top = [float(l.split(",")[3]) for l in lines[1:]]
    assert len(lines) == 49 and top == sorted(top, reverse=True)
    _, out, _ = run(capsys, "categories", "--table", "--sort", "name", "--format", "json")
    names = [r["category"] for r in json.loads(out)]
    assert names == sorted(names)


def test_html_report_for_one_and_two_ideas(tmp_path, capsys):
    one, two = tmp_path / "one.html", tmp_path / "two.html"
    run(capsys, "score", "--category", "Tools", "--name", "<b>Tool</b>", "--html", str(one))
    run(capsys, "compare", "--a", "category=Puzzle", "--b", "category=Tools", "--html", str(two))
    h1, h2 = one.read_text(), two.read_text()
    assert "<b>Tool</b>" not in h1 and "&lt;b&gt;Tool&lt;/b&gt;" in h1
    assert h1.count("<polyline") == 3 and h2.count("<polyline") == 6
    for h in (h1, h2):
        assert h.startswith("<!doctype html>") and "http://" not in h and "https://" not in h
        assert "say nothing about what any change" in h


def test_shipped_data_files_are_found():
    for name in ["milestones.json", "comparables.csv.gz", "market.csv", "checks.json"]:
        assert PM.data_path(name) is not None
