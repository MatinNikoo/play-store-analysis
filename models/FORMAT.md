# models/milestones.json format (playodds-milestones/1)

This is the launch-time milestone model in plain JSON, so it can run in Python (`playodds/model.py`), in a
browser (`score.js`) or anywhere else. `tests/test_export.py` checks that both scorers give the same numbers as
the fitted statsmodels GLM to within 1e-9.

## What it predicts

For an app with launch-day inputs x, the chance it reaches N installs within t months of launch:

```
eta = intercept + log_time * ln(t) + sum_j coef[j] * term_j(x)
P(T_N <= t | x) = G(eta)
```

`G` is the inverse link named in `link`:

| link | G(eta) | reading |
|---|---|---|
| `cloglog` | 1 - exp(-exp(eta)) | Weibull, proportional hazards |
| `probit` | standard normal cdf | log-normal time to milestone |
| `logit` | 1 / (1 + exp(-eta)) | log-logistic time to milestone |

The link was picked by held-out log loss (see `results/study/milestones.json`, key `link_log_loss`).

## Top level fields

| field | meaning |
|---|---|
| `format` | always `playodds-milestones/1` for this layout |
| `link` | `cloglog`, `probit` or `logit` |
| `time_unit` | `months` (30.44 days). Both scorers accept 0.5 to 66 months, the range of ages in the fit, and raise an error outside it |
| `categories` | the Play categories the model knows |
| `reference_category` | the category with no term of its own (folded into the intercept) |
| `inputs` | input names and their defaults |
| `terms` | how inputs turn into numbers, in coefficient order |
| `milestone_list` | `["10000", "100000", "1000000"]` |
| `milestones` | per milestone: `intercept`, `log_time`, `coef`, `cov` |
| `index_milestone` | the milestone whose `coef` defines the launch score used by `conformal` |
| `z` | normal quantiles for the 50, 80 and 90 percent intervals |
| `conformal` | per level: `edges`, `lo`, `hi` for the launch-month percentile interval |
| `percentile_installs` | per horizon in months: install counts at each launch-month percentile |
| `meta` | fit window, sample sizes, validation AUCs, notes |

## Inputs

| input | what it is | default |
|---|---|---|
| `category` | Play category, must be one of `categories` | none |
| `name` | app title (only its length, word count, non-ASCII and digit flags are used) | "" |
| `earlier_apps` | apps the developer released on Play before this one | 0 |
| `same_day_apps` | other apps the developer released the same day | 0 |
| `apps_last_30_days` | apps the developer released in the 30 days before | 0 |
| `months_since_first_app` | months since the developer's first app, 0 for a first app | 0 |
| `earlier_categories` | distinct categories among earlier apps | 0 |
| `earlier_apps_same_category` | earlier apps in the same category | 0 |
| `builder_apps` | earlier Play apps sharing the package prefix (app builder templates) | 0 |

Every input is known on launch day. Nothing about price, ads, in-app purchases, size, content rating, ratings or
updates goes in, since those were only seen in the June 2021 scrape.

## Terms

Each term turns one input into one number. Kinds:

| kind | value |
|---|---|
| `onehot` | 1 if input equals `level`, else 0 |
| `bin` | 1 if `lo <= input <= hi`, else 0 |
| `log1p` | ln(1 + min(max(input, 0), cap)) |
| `linear` | min(max(input, 0), cap) / scale |

Name inputs come from the title: `name_len` is the number of characters (code points), `name_words` the number
of whitespace separated words, `name_nonascii` is 1 if any character is above code point 127, `name_digit` is 1
if the title has a 0 to 9 digit.

## Intervals

- Probability intervals: `cov` is the developer-clustered covariance of `[intercept, log_time, coef...]`. The
  interval is G(eta -/+ z * se) with se from g' cov g, g = [1, ln t, terms]. It only reflects how precisely the
  coefficients are estimated, not whether the model is right.
- Percentile intervals: split conformal inside 20 bins of the launch score s = coef(index_milestone) . terms.
  Find the bin with `edges` (right side search), then read `lo` and `hi`. These are where the app is expected to
  land among apps launched the same month. Coverage on held-out apps is in `results/study/milestones.json`.
- `percentile_installs[t]` maps a launch-month percentile to installs for the cohort that was t months old in
  June 2021 (linear interpolation over `pct`).

## Order of milestones

Each milestone is fit on its own, so in rare cases P(1M) could come out above P(100K). Both scorers apply a running
minimum in `milestone_list` order (10K, then 100K, then 1M) to the point estimate and both interval ends.
