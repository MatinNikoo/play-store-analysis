# Data card

Every number here comes from `results/study/populations.json` (written by `python -m study.prepare`).

## What it is

A scrape of Google Play listings made on June 15 and 16, 2021 by Gautham Prakash, published at
[github.com/gauthamp10/Google-Playstore-Dataset](https://github.com/gauthamp10/Google-Playstore-Dataset) under the
MIT license. `get_data.sh` fetches commit `423196c20029a69d6bc68f1eafc3342ec79dcf8a` of that repo, stacks its three
csv parts into `data/google_playstore_2021.csv` (about 650 MB, never committed here) and checks the file against
SHA256 `f8830426d88687801d5874dfd1e1dfc209a1324e56eb837ab46e942dc4d328e1`.

One row is one app as it looked on the scrape day. There is no history. Each app is seen exactly once.

| | count |
|---|---:|
| Raw rows | 2,312,944 |
| Rows with no release date, install count or category (dropped) | 71,053 |
| Duplicate app ids (dropped) | 0 |
| Apps after cleaning | 2,241,891 |
| Developers | 736,124 |
| First and last release date | Jan 28, 2010 and Jun 16, 2021 |
| Apps released 2018 and 2019 (the old README cohort, "815K") | 815,231 |
| Of those, known size and one of the four main content ratings (the live simulator, "about 800,000") | 799,763 |
| Apps released Jan 2016 to May 2021 (milestone model window) | 1,959,938 |

The 815,231 and 799,763 numbers are both right. They describe different groups. The simulator drops 15,422 apps whose
size is "Varies with device" and 46 apps rated Unrated or Adults only 18+, because it asks for those as inputs.

## Columns I use and when they were true

| column | what it means | when it was true |
|---|---|---|
| App Id | package name | fixed forever at launch |
| Released | first release date | fixed at launch |
| Developer Id | developer account name | launch, used only to link a developer's apps |
| Category, App Name | listing fields | set at launch, editable later |
| Maximum Installs | near exact install count | scrape day (this is the outcome) |
| Free, Price, Ad Supported, In App Purchases | monetization | scrape day only |
| Size, Minimum Android, Content Rating | build and rating | scrape day only, change with updates |
| Privacy Policy, Developer Website | links on the listing | scrape day only |
| Rating, Rating Count, Editors Choice | reviews and awards | scrape day, need users first |
| Last Updated | last update date | scrape day |

The full feature availability table, with which model uses what, is in [method.md](method.md).

## Who it covers and who it misses

- Only apps still listed in June 2021. Apps that were taken down before that, by Google or by their developer, are
  not here at all. Failed apps are more likely to be gone, and the older the launch year the more of them are gone.
  So older cohorts look more successful than they really were. This is the biggest bias in the data and it can't
  be fixed from a single snapshot.
- Developer history counts only earlier apps that survived to 2021, so it runs low, more so for older developers.
- Google Play only. No iOS, no Chinese Android stores.
- 98.0% of apps are free, 50.6% show ads and 8.5% have in-app purchases (all as of 2021).
- 21.6% of apps were never updated after their release date. 11,035 apps have zero installs.
- 2.3% of apps have no numeric size ("Varies with device").

## Known quirks

- Install counts are cumulative, so they only go up. That's what makes the current status (one look per app)
  survival analysis valid.
- Release dates are days, not times.
- The scrape spans two days, so app age at observation uses each row's own scrape date.
- Median installs by launch month drop sharply between May and June 2019 (from 662 to 392, see `by_launch_month`
  in populations.json). I don't know why. It could be a scrape artifact or a change on Google's side. The milestone
  model's forward check covers launches after that date, so any effect on the model shows up there.
- Raw milestone rates by app age are not a smooth function of age. Among apps 5 to 16 months old the share past 10K
  only moves between 13.0% and 14.1%, which suggests cohorts differ (see `cohort_rates` in `results/study/checks.json`).

## Licensing

The dataset is MIT licensed. App names and ids in `models/comparables.csv.gz` are a sample of public listing data
from it.
