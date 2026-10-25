# Method

This is the long version of how the study works. The short version is in the README and the paper is in
`paper/paper.pdf`. Code for each part is in `study/`.

## 1. The problem with the old model

The old model (README section 5, `analysis.py`) predicted whether an app released in 2018 or 2019 had 100K installs
by June 2021. It had two problems.

1. Leakage. Monetization, size, minimum Android version, content rating, privacy policy and website were all read off
   the June 2021 listing. An app that took off in 2019 can add in-app purchases in 2020, grow from 20 MB to 90 MB,
   and get a privacy policy because Google started asking for one. So the model was partly reading the answer.
2. Unequal time. An app from January 2018 had 41 months to reach 100K and an app from December 2019 had 18. With one
   snapshot the old outcome mixes "good app" with "old app". The share of apps past 100K falls from about 13% for
   2016 launches to about 4% for 2020 launches (`by_launch_month` in `results/study/populations.json`).

## 2. Feature availability

Every feature any model here touches, when it could first have been known, and whether success could change it.
The same table lives in code (`study/features.py`, `AVAILABILITY`) and in
`results/study/feature_availability.csv`. `tests/test_leakage_guard.py` fails if anything that isn't marked
"launch" gets into the launch-time model.

| feature | source column | first known | success can change it | old model | launch model | note |
|---|---|---|---|---|---|---|
| `Category` | Category | launch | low | yes | yes | picked at launch, can be changed later but rarely is |
| `rel_month` | Released | launch | none |  | yes | release date never changes |
| `rel_dow` | Released | launch | none |  | yes | release date never changes |
| `dev_prior_apps` | Developer Id + Released | launch | none |  | yes | apps by the same developer released earlier |
| `dev_same_day` | Developer Id + Released | launch | none |  | yes | other apps the developer released the same day |
| `dev_prior_30d` | Developer Id + Released | launch | none |  | yes | apps the developer released in the 30 days before |
| `dev_days_active` | Developer Id + Released | launch | none |  | yes | days since the developer's first app |
| `dev_prior_cats` | Developer Id + Category | launch | low |  | yes | distinct categories among earlier apps |
| `dev_prior_same_cat` | Developer Id + Category | launch | low |  | yes | earlier apps in the same category |
| `pkg_prior` | App Id + Released | launch | none |  | yes | earlier apps sharing the package prefix (app builders) |
| `pkg_segments` | App Id | launch | none |  | yes | package names can never change |
| `pkg_len` | App Id | launch | none |  | yes | package names can never change |
| `name_len` | App Name | launch | low |  | yes | title can be edited later (store optimization) |
| `name_words` | App Name | launch | low |  | yes | title can be edited later |
| `name_nonascii` | App Name | launch | low |  | yes | title can be edited later |
| `name_digit` | App Name | launch | low |  | yes | title can be edited later |
| `dev_apps` | Developer Id + Released | launch | none | yes |  | old feature, same as dev_prior_apps + 1 |
| `Content Rating` | Content Rating | snapshot | low | yes |  | questionnaire is redone when features like chat get added |
| `size_mb` | Size | snapshot | high | yes |  | apps grow with every update and hits update more |
| `min_android` | Minimum Android | snapshot | medium | yes |  | raised in updates, hits update more |
| `paid` | Free | snapshot | medium | yes |  | price can change, paid apps often go free |
| `price` | Price | snapshot | medium |  |  | price can change |
| `ads` | Ad Supported | snapshot | high | yes |  | ads are easy to add once an app has traffic |
| `iap` | In App Purchases | snapshot | high | yes |  | purchases get added after an app has users |
| `has_privacy` | Privacy Policy | snapshot | high | yes |  | policies got added as Google required them and as apps grew |
| `has_website` | Developer Website | snapshot | medium | yes |  | developers with a hit build a site |
| `dev_prior_hit_share` | Developer Id + Maximum Installs | after | high |  |  | earlier apps' installs measured in 2021 |
| `dev_prior_max_installs` | Developer Id + Maximum Installs | after | high |  |  | earlier apps' installs measured in 2021 |
| `rating` | Rating | after | high |  |  | needs users |
| `rating_count` | Rating Count | after | high |  |  | basically a second install count |
| `editors_choice` | Editors Choice | after | high |  |  | given to apps that already did well |
| `days_since_update` | Last Updated | after | high |  |  | hits keep getting updated |

"Low" for category and title is a judgment call. Both are listing fields a developer can edit, and the scrape only
has the 2021 value. I keep them in the launch model because a developer picks both before launch and there's no
obvious way a hit would cause a category switch. The title is the weaker case, since developers rewrite titles for
search after an app starts to get traffic, so some of its signal may come from edits made after launch. The strict
model drops both and is reported next to it. A separate ablation drops the title features alone (AUC 0.830 to 0.783)
and category alone (0.830 to 0.779), on a 40% sample of the training developers.

### Developer history, and how the old code built it

The old `dev_apps` used `groupby("Developer Id")["released"].rank(method="min")`, which is the number of the
developer's apps released strictly before this one, plus one. So the timing was already right, nothing from the
future went in. Two things were still off.

- It only counts earlier apps that were still listed in June 2021. Apps that got removed are invisible, so it runs low.
- The README and the simulator describe it as "apps already published", but the bucket labels count the app itself.
  The "6 to 20" bucket really means 5 to 19 earlier apps. I fixed the wording in the README. The simulator labels
  are listed as a follow-up since the page is live.

The new features (`study/features.py`) are all computed with one sort and binary search per developer, using only
apps released on earlier days. `tests/test_leakage_guard.py::test_features_ignore_the_future` changes every field
of apps released after a cutoff (and deletes some) and checks that no launch feature of an earlier app moves.

A developer's track record (how many of their earlier apps became hits) sounds like launch information, but it
isn't here. Those earlier apps' installs were counted in June 2021, after this app launched, and a new hit can pull
installs to the developer's older apps. I keep it out of every launch model and only use it in the ablation.

## 3. Outcomes with a consistent horizon

### Ranking target: top 5% of the launch month

An app is a hit if its installs are in the top 5% of all apps released in the same calendar month
(`study/features.py::top_in_month`). Every app in a month had the same time to grow, so the comparison is fair no
matter the launch date, and the base rate is 5% in every month by construction. I also report the top 1% and top 10%.

### Absolute target: time to N installs

For "how likely is it to hit 100K within a year" I treat the snapshot as current status data. App i launched
a_i months before the scrape. Installs only go up, so the scrape tells me whether the time to reach N installs,
T_i, is below a_i, and nothing else. That's a classic interval censored setup (current status data).

- Without covariates the NPMLE of F(t) = P(T <= t) is the isotonic regression of the 0/1 "crossed yet" indicator on
  age (pool adjacent violators, `study/milestones.py::pava`). The NPMLE converges at rate n^(1/3) and the plain
  bootstrap is not valid for it, so I use subsampling (Politis and Romano) with m = 10% of developers and the
  cube-root rate to get intervals.
- With launch covariates x I fit P(T <= t | x) = G(a + b log t + x'beta) as a binary GLM with log age as an input.
  G = cloglog is a Weibull proportional hazards model, G = probit is a log-normal accelerated failure time model,
  G = logit is log-logistic. Each one implies a time ratio exp(-beta_j / b) for covariate j.
- The key assumption is that the time to milestone doesn't depend on launch date once x is fixed (no cohort effect).
  With one snapshot, age and launch date are the same thing, so this can't be tested inside the fit. It can be tested
  out of sample, which is what the forward check does.

## 4. Evaluation

### Ranking (study/ranking.py)

- Split by launch date. Train on 2018 launches, test on 2019 launches, same as the old model. A developer can show
  up on both sides, since their history is legitimately known at launch. No app is on both sides
  (`tests/test_features.py::test_split_integrity`).
- One caveat for a PhD reader. The training labels are also measured in June 2021, after the test apps launched.
  A real deployment in January 2019 would only see 2018 apps' installs as of January 2019. With one snapshot I can't
  replicate that, so this is only a temporal split.
- Models: histogram gradient boosting (early stopping on a random 10% of training rows), a spline logistic
  regression, and three lookup baselines (category rate, developer history rate, category by developer rate, shrunk
  toward the overall rate with 50 pseudo apps).
- AUC with 95% intervals from a bootstrap that resamples whole developers (500 draws), paired differences on the
  same draws, Brier score and Brier skill against the training base rate, calibration slope and intercept (logistic
  recalibration, intercept with slope fixed at 1), reliability by decile with Wilson intervals, expected calibration
  error, and the share of hits found in the top 10% of scores.
- Stability: three refits, each on a different random half of the training developers and a different early
  stopping split.
- Subgroups: category, developer history, release quarter, games vs apps.
- Distribution shift: train on 2016 and 2017 launches, test on 2018, 2019 and 2020 separately.
- Leakage ablation: ten feature sets fit on the same 40% sample of training developers (to keep runtime sane on
  shared cores), from the old feature set down through removing monetization, listing and every snapshot field, and
  from the launch set up through adding each leaky group back.
- Error analysis: profiles of missed hits (bottom half of scores), caught hits and false alarms (top 10% of scores,
  not a hit), including 2021-only fields as description (not as inputs).

### Milestones (study/milestones.py)

- Window: launches from January 2016 to May 2021 (ages from two weeks to 65 months).
- Forward check: fit on launches before June 2019 (ages 24 months and up), test on launches after (ages under 24
  months). This is extrapolation in age and in calendar time at once, so it's the hard test of the no cohort effect
  assumption.
- Interleaved check: hold out every fourth launch month, fit on the rest.
- The link is picked by held-out log loss averaged over both checks and all three milestones.
- GLM fits use a random sample of 400K apps, because statsmodels keeps several float64 copies of the design matrix and
  the full 1.96M apps do not fit in 7 GB. The stage reads a slim copy of the feature table (16 columns, launch window
  only), builds the design in float32, frees each fit's data right after fitting, and peaks at about 5.3 GB. The
  association models use a chunked IRLS for the same reason (peak about 1.3 GB). The final fit uses developer-clustered
  sandwich covariance.
- Intervals for a single app: (1) probability intervals from the clustered covariance (parameter uncertainty only),
  and (2) split conformal intervals for where the app lands among apps launched the same month, within 20 bins of the
  launch score (Mondrian conformal), calibrated on apps the final fit never saw. Coverage is checked on random
  held-out apps and on launches after June 2019 with a forward-fit score, overall and by developer history,
  games vs apps, and category.

## 5. Associations (study/associations.py)

- Modified Poisson regression (log link, robust errors clustered by developer) of the top 5% indicator on 2021
  monetization group (reference "ads only"), privacy policy and website, with fixed effects for category, launch
  month and developer history bin. Exponentiated coefficients are adjusted risk ratios.
- E-values (VanderWeele and Ding 2017) for each ratio and for the confidence limit closest to 1. The E-value is the
  minimum strength of association, on the risk ratio scale, that an unmeasured factor (or a reverse path) would need
  with both the flag and the outcome to fully explain the observed ratio.
- Timing evidence: the same model fit separately on apps never updated after launch and on apps that were updated.
  A never-updated app's build is the launch build, so its in-app purchase flag is much closer to what it had on day
  one. Being updated is also more common for hits, so the never-updated group is selected on
  something downstream of success. I read the comparison as a hint about timing and nothing stronger.
