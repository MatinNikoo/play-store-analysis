"""Writes paper/numbers.tex: every number the paper quotes, read straight from results/.
Run from the repo root:  python paper/make_numbers.py"""

import json
from pathlib import Path

RES = Path("results")


def load(name):
    p = RES / "study" / name if name != "summary.json" else RES / name
    return json.loads(p.read_text())


def get(d, path):
    for k in path.split("/"):
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d


def fmt(v, kind):
    out = fmt_raw(v, kind)
    return "$-$" + out[1:] if out.startswith("-") else out


def fmt_raw(v, kind):
    if kind == "auc":
        return f"{v:.3f}"
    if kind == "pct0":
        return f"{100 * v:.0f}\\%"
    if kind == "pct1":
        return f"{100 * v:.1f}\\%"
    if kind == "pct2":
        return f"{100 * v:.2f}\\%"
    if kind == "int":
        return f"{int(round(v)):,}".replace(",", "{,}")
    if kind == "x2":
        return f"{v:.2f}"
    if kind == "x3":
        return f"{v:.3f}"
    if kind == "x4":
        return f"{v:.4f}"
    if kind == "str":
        return str(v)
    raise ValueError(kind)


def macros():
    rk, ms, asc = load("ranking.json"), load("milestones.json"), load("associations.json")
    pop, old = load("populations.json"), load("summary.json")
    out = {}

    def add(name, src, path, kind):
        out[name] = fmt(get(src, path), kind)

    add("RawRows", pop, "raw_rows", "int")
    add("CleanApps", pop, "cleaned_apps", "int")
    add("CohortApps", pop, "cohort_2018_2019", "int")
    add("SimApps", pop, "simulator_population", "int")
    add("TrainApps", rk, "split/train_apps", "int")
    add("TestApps", rk, "split/test_apps", "int")
    add("AblTrainApps", rk, "split/ablation_train_apps", "int")
    add("WindowApps", ms, "apps", "int")
    add("Developers", pop, "developers", "int")
    add("NeverUpdated", pop, "share_never_updated", "pct1")
    add("OldReportedAUC", old, "model/auc_gradient_boosting", "auc")
    add("OldReportedLogit", old, "model/auc_logistic", "auc")

    names = {"gbm_launch": "Launch", "logit_launch": "LaunchLogit", "gbm_strict": "Strict",
             "base_category_dev": "BaseCatDev", "base_category": "BaseCat", "base_dev": "BaseDev",
             "gbm_old": "Old", "logit_old": "OldLogit"}
    for k, n in names.items():
        for f, kind in [("auc", "auc"), ("lo", "auc"), ("hi", "auc"), ("brier", "x4"), ("brier_skill", "x3"),
                        ("cal_slope", "x2"), ("cal_intercept", "x2"), ("ece", "x4"), ("top10_captures", "pct0"),
                        ("diff_vs_gbm_launch", "x3"), ("diff_lo", "x3"), ("diff_hi", "x3")]:
            tag = {"auc": "AUC", "lo": "Lo", "hi": "Hi", "brier": "Brier", "brier_skill": "BSS",
                   "cal_slope": "Slope", "cal_intercept": "Int", "ece": "ECE", "top10_captures": "TopTen",
                   "diff_vs_gbm_launch": "Diff", "diff_lo": "DiffLo", "diff_hi": "DiffHi"}[f]
            add(f"{tag}{n}", rk, f"main/{k}/{f}", kind)
    add("AUCOldHundredK", rk, "main_100k/gbm_old_100k/auc", "auc")
    add("WithinOldHundredK", rk, "main_100k/gbm_old_100k/within_month_auc", "auc")
    add("AUCLaunchHundredK", rk, "main_100k/gbm_launch_100k/auc", "auc")
    add("WithinLaunchHundredK", rk, "main_100k/gbm_launch_100k/within_month_auc", "auc")
    add("SimPopApps", rk, "simulator_population_test/apps", "int")
    add("SimPopLaunch", rk, "simulator_population_test/auc_gbm_launch", "auc")
    add("SimPopOld", rk, "simulator_population_test/auc_gbm_old", "auc")

    for k, n in [("old features (all)", "AblOld"), ("old minus monetization", "AblOldNoMon"),
                 ("old minus monetization and listing", "AblOldNoMonList"),
                 ("old minus all snapshot fields", "AblOldNoSnap"), ("monetization only", "AblMonOnly"),
                 ("launch features", "AblLaunch"), ("launch + monetization", "AblLaunchMon"),
                 ("launch + all snapshot fields", "AblLaunchSnap"),
                 ("launch + earlier apps' 2021 success", "AblLaunchLeakyDev"),
                 ("launch + post-launch fields", "AblLaunchPost"),
                 ("launch minus title", "AblNoTitle"), ("launch minus category", "AblNoCategory")]:
        add(n, rk, f"ablation/{k}", "auc")
    add("SnapShare", rk, "snapshot_share_of_old_auc_above_half", "pct0")
    add("GapShare", rk, "launch_gap_share_of_old_auc_above_half", "pct0")
    for m in ["launch", "old"]:
        n = m.capitalize()
        add(f"Seed{n}Mean", rk, f"seeds/{m}/mean", "auc")
        add(f"Seed{n}SD", rk, f"seeds/{m}/sd", "x4")
    for yr in ["2018", "2019", "2020"]:
        for m in ["launch", "old"]:
            add(f"Shift{m.capitalize()}{['A', 'B', 'C'][int(yr) - 2018]}", rk, f"shift/{yr}/{m}/auc", "auc")
    for k in ["top1", "top10"]:
        n = {"top1": "One", "top10": "Ten"}[k]
        add(f"Sens{n}Launch", rk, f"outcome_sensitivity/{k}/auc_launch", "auc")
        add(f"Sens{n}Old", rk, f"outcome_sensitivity/{k}/auc_old", "auc")
    add("RecallOverall", rk, "errors/recall_top10_overall", "pct0")
    add("PrecisionTop", rk, "errors/precision_top10", "pct1")

    add("Link", ms, "link", "str")
    for m, n in [("10K", "TenK"), ("100K", "HundredK"), ("1M", "OneM")]:
        for t, tn in [("12", "Twelve"), ("36", "ThirtySix")]:
            add(f"F{n}{tn}", ms, f"npmle/{m}/{t}/F", "pct1" if m != "1M" else "pct2")
            add(f"F{n}{tn}Lo", ms, f"npmle/{m}/{t}/lo", "pct1" if m != "1M" else "pct2")
            add(f"F{n}{tn}Hi", ms, f"npmle/{m}/{t}/hi", "pct1" if m != "1M" else "pct2")
        link = ms["link"]
        for v, vn in [("forward", "Fwd"), ("interleaved", "Int")]:
            add(f"{vn}AUC{n}", ms, f"validation/{v}/{m}/{link}/auc", "auc")
            add(f"{vn}WAUC{n}", ms, f"validation/{v}/{m}/{link}/within_month_auc", "auc")
            add(f"{vn}LL{n}", ms, f"validation/{v}/{m}/{link}/log_loss", "x4")
            add(f"{vn}AgeLL{n}", ms, f"validation/{v}/{m}/age_only_cloglog/log_loss", "x4")
        add(f"FwdSlopeYear{n}", ms, f"validation/forward/{m}/{link}/calibration_9_to_15_months/slope", "x2")
        add(f"FwdIntYear{n}", ms, f"validation/forward/{m}/{link}/calibration_9_to_15_months/intercept", "x2")
    for l in ["cloglog", "probit", "logit"]:
        add(f"LL{l.capitalize()}", ms, f"link_log_loss/{l}", "x4")
    add("CovEighty", ms, "conformal/80/coverage", "pct1")
    add("CovNinety", ms, "conformal/90/coverage", "pct1")
    add("CovFifty", ms, "conformal/50/coverage", "pct1")
    add("CovEightyFwd", ms, "conformal_forward/80/coverage", "pct1")
    add("WidthEighty", ms, "conformal/80/mean_width", "x2")
    add("CovGames", ms, "conformal/by_80/kind/Games", "pct1")
    add("WidthFifty", ms, "conformal/50/mean_width", "x2")
    add("WidthNinety", ms, "conformal/90/mean_width", "x2")
    add("WidthEightyFwd", ms, "conformal_forward/80/mean_width", "x2")
    add("FitSample", ms, "fit_sample", "int")
    add("FwdMeanPredHundredK", ms, f"validation/forward/100K/{ms['link']}/mean_pred", "pct1")
    add("FwdRateHundredK", ms, f"validation/forward/100K/{ms['link']}/rate", "pct1")
    add("IntSlopeHundredK", ms, f"validation/interleaved/100K/{ms['link']}/cal_slope", "x2")
    mk = load("market.json")
    cats = {c["category"]: c for c in mk["categories"]}
    add("SimFirstTenK", cats, "Simulation/first_app_12m_10K", "pct1")
    add("BusFirstTenK", cats, "Business/first_app_12m_10K", "pct1")
    add("TopTenCorr", mk, "corr_top10_vs_first_app_10K", "x2")
    ck = load("checks.json")
    cr = ck["cohort_rates"]
    add("RawApps", cr, "11 to 13 months/apps", "int")
    add("RawTenK", cr, "11 to 13 months/10K/observed", "pct1")
    add("RawHundredK", cr, "11 to 13 months/100K/observed", "pct2")
    add("RawOneM", cr, "11 to 13 months/1M/observed", "pct2")
    add("RawModelHundredK", cr, "11 to 13 months/100K/model", "pct1")
    flat = [cr[k]["10K"]["observed"] for k in ["5 to 7 months", "8 to 10 months", "11 to 13 months", "14 to 16 months"]]
    out["RawFlatLo"] = fmt(min(flat), "pct1")
    out["RawFlatHi"] = fmt(max(flat), "pct1")
    add("RawObsEarly", cr, "5 to 7 months/10K/observed", "pct1")
    add("RawModelEarly", cr, "5 to 7 months/10K/model", "pct1")
    add("RawObsLate", cr, "17 to 19 months/10K/observed", "pct1")
    add("RawModelLate", cr, "17 to 19 months/10K/model", "pct1")
    fa = ck["first_apps_2019_2020"]
    for cat, n in [("Simulation", "Sim"), ("Business", "Bus")]:
        add(f"{n}ObsApps", fa, f"{cat}/ages_9_to_15/apps", "int")
        add(f"{n}ObsTenK", fa, f"{cat}/ages_9_to_15/10K/observed", "pct1")
        add(f"{n}ModelTenK", fa, f"{cat}/ages_9_to_15/10K/model", "pct1")
    cs = ck["by_category_summary"]
    add("CatInsideTenK", cs, "10K/share_with_model_inside_interval", "pct0")
    add("CatInsideHundredK", cs, "100K/share_with_model_inside_interval", "pct0")
    add("CatRatioMin", cs, "10K/min", "x2")
    add("CatRatioMax", cs, "10K/max", "x2")
    add("NpmleSimCoverage", ck, "npmle_interval_simulation/coverage", "pct0")
    add("NpmleSimNominal", ck, "npmle_interval_simulation/nominal", "pct0")
    add("NpmleSimApps", ck, "npmle_interval_simulation/apps", "int")

    for g, n in [("Ads + in-app purchases", "AdsIap"), ("In-app purchases only", "IapOnly"),
                 ("Free, no ads or purchases", "NoMon"), ("Paid upfront", "Paid"),
                 ("Privacy policy", "Privacy"), ("Developer website", "Website")]:
        for block, bn in [("adjusted_all", "All"), ("adjusted_never_updated", "Never"), ("adjusted_updated", "Upd")]:
            r = asc[block].get(g, {})
            if r.get("rr") is None:
                continue
            add(f"RR{n}{bn}", asc, f"{block}/{g}/rr", "x2")
            add(f"RR{n}{bn}Lo", asc, f"{block}/{g}/lo", "x2")
            add(f"RR{n}{bn}Hi", asc, f"{block}/{g}/hi", "x2")
            add(f"E{n}{bn}", asc, f"{block}/{g}/e_value", "x2")
            add(f"ECI{n}{bn}", asc, f"{block}/{g}/e_value_ci", "x2")
    ib = asc["iap_share_by_update"]
    add("IapHitsUpd", ib, "updated_top5", "pct0")
    add("IapHitsNever", ib, "never_updated_top5", "pct0")
    out["IapHitsRatio"] = fmt(ib["updated_top5"] / ib["never_updated_top5"], "x2")
    add("NeverHits", asc, "never_updated_share/top5", "pct0")
    add("NeverRest", asc, "never_updated_share/rest", "pct0")
    return out


def main():
    lines = ["% written by paper/make_numbers.py from results/, do not edit by hand"]
    for k, v in macros().items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    Path("paper/numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines) - 1} numbers to paper/numbers.tex")


if __name__ == "__main__":
    main()
