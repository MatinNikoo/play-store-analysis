# Model card

There are two models in this repo that someone could actually use. Both only take inputs that are known on the day
an app launches. Numbers below come from `results/study/ranking.json` and `results/study/milestones.json`.

## 1. Milestone model (`models/milestones.json`, `playodds`)

### What it does

Given a category, a title and the developer's release history, it gives the chance the app reaches 10K, 100K and 1M
installs within t months of launch, plus an interval for where it lands among apps launched the same month, and a few
real apps with a similar launch-day profile. The `playodds` command can also compare two ideas, score a CSV of ideas, print one category's market table and write an HTML report (README, "The playodds command"). The CLI accepts t from 0.5 to 66 months, which is the range of ages in the
fit, and refuses anything outside it.

### How it works

There is one binary GLM per milestone, fit to current status data. Each app launched between January 2016 and May 2021
is seen once at its age in June 2021, with a yes or no for whether it had crossed the milestone.
P(T <= t | x) = G(a + b log t + x'beta). Cloglog, probit and logit gave log losses of 0.1972, 0.1993 and 0.1976, which I
treat as a tie, and the file uses cloglog. It is fit on a random sample of 400,000 apps, with standard errors clustered
by developer. The full spec is in [models/FORMAT.md](../models/FORMAT.md).

### Inputs

Category, title (length, words, non-ASCII, digits), how many apps the developer already released, how many the same
day and in the last 30 days, months since their first app, how many categories and same-category apps they had, and how
many earlier apps share the package prefix (a sign of app builder templates). Nothing about price, ads, in-app
purchases, size, ratings or updates. Category and title are 2021 values, and a developer can edit both after launch.

### How well it works

Numbers are in `results/study/milestones.json` (`validation`) and `results/study/checks.json`. I ran two checks. In the
interleaved check I held out every fourth launch month and fit on the rest. For 100K installs the AUC was 0.811, the
calibration slope 0.98 and the calibration intercept 0.00. In the forward check I fit on launches before June 2019 and
scored launches after it. The AUC was 0.797 for 100K, but the model under-predicted. It averaged 2.7% against a real
3.6%, and near one year of age the calibration intercept was 0.40. Every app in that fit was at least 24 months old, so
the forward check also extrapolates the time curve to ages the fit never saw. I can't separate the two causes with one
snapshot. In both checks the model beat an age-only model on log loss for all three milestones.

The final model is pooled over cohorts, so it does not follow the raw rates by age exactly. Among apps 5 to 7 months
old the model's 10K average (10.4%) is about a fifth below the raw rate (13.2%). At 17 to 19 months it is about a
seventh above (16.6% against 14.4%). At 11 to 13 months the raw 100K rate is 3.45% and the model says 3.6%.

Per category, I compared the model's average with the raw rate for apps 9 to 15 months old. The model sat inside the
95% interval of the raw rate for 46% of the 48 categories at 10K and 52% at 100K. The ratio of observed to modeled
10K ran from 0.71 (House & Home) to 2.09 (Events). For first apps launched in 2019 and 2020 in Simulation, the model
says 43.0% for 10K at those ages and the raw rate was 36.3% (n=717). For Business the two were 4.5% and 4.5%
(n=7,628). So the model is fine on average and can be off by a factor of two in a single category.

### Intervals

- Probability intervals only show how precisely the coefficients are pinned down. With 400,000 apps they are narrow.
  They do not include model error, so they are too narrow as a statement about any single app.
- Percentile intervals are split conformal, inside 20 bins of the launch score, calibrated on apps the fit never saw.
  The 80% interval covered 80.1% of random held-out apps and 80.7% of launches after June 2019. Coverage is lower for
  games (75.2%) and for some game categories, such as Strategy (69%) and Action (71%).
- The intervals are wide. On the scale from 0 to 1 for the launch-month percentile, the mean width is 0.44 for the 50%
  interval, 0.73 for 80% and 0.84 for 90%. For a single app they only give a rough neighborhood.
- The install range the CLI prints converts the percentile interval using the cohort that was t months old in June
  2021. That conversion assumes an app keeps roughly its rank among same-month launches.
- The pooled time-to-milestone intervals come from subsampling and ignore cohort drift. In a simulation with a known
  curve, intervals meant to cover 95% covered 85%.

### Use it for

A rough prior for a new app before any user data exists, a comparison of launch-day profiles, and a baseline that a
better model with real early traction data should beat.

### Don't use it for

Choosing a business model. It has no monetization input on purpose, and the associations in the README are not
effects. It also should not be used to judge an app after launch, since a week of real installs says more than
anything here, or for any store other than Google Play or any period far from 2016 to 2021.

### Known failure modes

- Within the ranking model, first apps from new developers have a recall problem. Only 40% of their hits land in its
  top 10%, against 49% to 52% for developers with 1 to 99 earlier apps. Their AUC is 0.80, the same as developers with
  1 to 4 earlier apps (0.80) and below those with more earlier apps, so recall is the only measure on which they
  look hardest.
- Games have a high base rate (12.4% of 2019 games land in the top 5% against 4.0% of other apps), which is why more
  of their hits land in the top 10%. Ranking within games is a little worse than within other apps (AUC 0.805 against
  0.825).
- Apps removed before June 2021 are missing, which makes older cohorts look better. The 12-month numbers lean mostly
  on apps launched around mid 2020, which had less time to be removed than older ones.
- The model assumes time to a milestone depends on launch facts and not on the launch date. The forward check, the
  per-age comparison and the per-category comparison above are where that assumption looks weakest.
- Category and title are 2021 values. Taking out the title features lowers the ranking AUC from 0.830 to 0.783 and
  taking out category lowers it to 0.779 (40% sample of training developers), so part of what the model uses may have
  been edited after launch.

## 2. Ranking model (research only, `study/ranking.py`)

Histogram gradient boosting on 16 launch-time features, trained on 2018 launches, scored on 2019 launches, target
"top 5% of installs among apps launched the same month". It is not exported because the milestone model answers the
same question in a form a person can use. Its job is to measure how much launch-day information there is, compared
with the old model that read 2021 fields. Full results, calibration, ablation, stability, subgroups and shift are in
`results/study/ranking.json`.

## Ethics and scope

The data is public listing data under the MIT license. The comparables file includes real app names and ids, which
are already public on Google Play. Predictions are about apps, not people. A developer's history is used only as a
count of releases.
