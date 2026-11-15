# Data Validation Report

*Generated:* 2026-11-15T08:19:46.640942+00:00  
*Ingestion batch:* `20261115T081946-4b2041`  
*Overall status:* **PASS**

## Summary

| Table | Input rows | Valid rows | Quarantined | Rows with warnings | Pass rate | Pandera contract |
|---|---:|---:|---:|---:|---:|---|
| users | 957 | 955 | 2 | 12 | 99.791% | PASS |
| items | 1691 | 1690 | 1 | 14 | 99.941% | PASS |
| interactions | 100055 | 100020 | 35 | 25 | 99.965% | PASS |

## Rule results

| Table | Rule | Severity | Failed rows | Description |
|---|---|---|---:|---|
| users | `missing_user_id` | error | 1 | user_id is null/blank |
| users | `invalid_type_user_id` | error | 1 | user_id is not a positive integer |
| users | `missing_age` | warning | 3 | age is null/blank |
| users | `invalid_age` | warning | 2 | age outside [5, 100] or non-numeric |
| users | `invalid_gender` | warning | 2 | gender not in ['M', 'F'] |
| users | `missing_occupation` | warning | 0 | occupation is null/blank |
| users | `non_standard_occupation` | warning | 2 | occupation has whitespace/upper-case |
| users | `missing_zip_code` | warning | 0 | zip_code is null/blank |
| users | `exact_duplicate` | warning | 3 | Identical user record appears more than once |
| users | `conflicting_duplicate_key` | warning | 0 | Same user_id with different attribute values |
| items | `missing_item_id` | error | 0 | item_id is null/blank |
| items | `invalid_type_item_id` | error | 1 | item_id is not a positive integer |
| items | `missing_title` | error | 0 | title is null/blank |
| items | `invalid_genre_flags` | error | 0 | genre indicator not in {0,1} |
| items | `missing_release_date` | warning | 4 | release_date is null/blank |
| items | `invalid_release_date` | warning | 1 | release_date not a valid DD-Mon-YYYY date |
| items | `non_standard_title` | warning | 5 | title has leading/trailing whitespace |
| items | `missing_imdb_url` | warning | 3 | imdb_url is null/blank |
| items | `exact_duplicate` | warning | 2 | Identical item record appears more than once |
| items | `conflicting_duplicate_key` | warning | 0 | Same item_id with different attribute values |
| interactions | `missing_user_id` | error | 0 | user_id is null/blank |
| interactions | `missing_item_id` | error | 0 | item_id is null/blank |
| interactions | `missing_rating` | error | 4 | rating is null/blank |
| interactions | `missing_timestamp` | error | 1 | timestamp is null/blank |
| interactions | `invalid_type_user_id` | error | 0 | user_id is not an integer |
| interactions | `invalid_type_item_id` | error | 0 | item_id is not an integer |
| interactions | `invalid_type_rating` | error | 5 | rating is not numeric |
| interactions | `invalid_rating` | error | 8 | rating not an integer in [1, 5] |
| interactions | `invalid_timestamp` | error | 5 | timestamp non-numeric or outside [1997-01-01, 1999-01-01) |
| interactions | `unknown_user_fk` | error | 6 | user_id not present in validated users (referential integrity) |
| interactions | `unknown_item_fk` | error | 6 | item_id not present in validated items (referential integrity) |
| interactions | `exact_duplicate` | warning | 15 | Identical interaction record appears more than once |
| interactions | `conflicting_duplicate_key` | warning | 10 | Same user_id+item_id with different attribute values |

Errors are quarantined in `data/errors/<table>_errors.csv` (with an `error_reasons` column).
Warnings are listed in `data/errors/<table>_warnings.csv` and resolved in the preparation stage.

## Column profile of validated data

### users

| Column | dtype | Nulls | Unique |
|---|---|---:|---:|
| user_id | int64 | 0 | 952 |
| age | float64 | 5 | 61 |
| gender | str | 0 | 3 |
| occupation | str | 0 | 23 |
| zip_code | str | 0 | 795 |

### items

| Column | dtype | Nulls | Unique |
|---|---|---:|---:|
| item_id | int64 | 0 | 1688 |
| title | str | 0 | 1666 |
| release_date | datetime64[ns] | 5 | 240 |
| video_release_date | str | 1690 | 0 |
| imdb_url | str | 3 | 1660 |
| unknown | int64 | 0 | 2 |
| Action | int64 | 0 | 2 |
| Adventure | int64 | 0 | 2 |
| Animation | int64 | 0 | 2 |
| Children | int64 | 0 | 2 |
| Comedy | int64 | 0 | 2 |
| Crime | int64 | 0 | 2 |
| Documentary | int64 | 0 | 2 |
| Drama | int64 | 0 | 2 |
| Fantasy | int64 | 0 | 2 |
| Film-Noir | int64 | 0 | 2 |
| Horror | int64 | 0 | 2 |
| Musical | int64 | 0 | 2 |
| Mystery | int64 | 0 | 2 |
| Romance | int64 | 0 | 2 |
| Sci-Fi | int64 | 0 | 2 |
| Thriller | int64 | 0 | 2 |
| War | int64 | 0 | 2 |
| Western | int64 | 0 | 2 |

### interactions

| Column | dtype | Nulls | Unique |
|---|---|---:|---:|
| user_id | int64 | 0 | 943 |
| item_id | int64 | 0 | 1682 |
| rating | int64 | 0 | 5 |
| timestamp | int64 | 0 | 49287 |
| event_time | datetime64[ns, UTC] | 0 | 49287 |
