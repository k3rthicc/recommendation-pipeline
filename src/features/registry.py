"""Feature registry - the single source of truth for every feature the pipeline produces.

The catalog in ``reports/feature_catalog.md`` is generated from this list, and the feature store
checks that every materialised column is registered here (and vice versa).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.ingestion.ingest import GENRES


@dataclass(frozen=True)
class FeatureDef:
    name: str
    entity: str          # user | item | user_item
    dtype: str
    description: str
    logic: str
    source: str
    used_by: str


USER_FEATURES = [
    FeatureDef("user_rating_count", "user", "int", "Number of ratings given before the snapshot time", "count(ratings)", "interactions", "hybrid, analysis"),
    FeatureDef("user_avg_rating", "user", "float", "Mean rating given by the user (global mean for cold-start users)", "mean(rating)", "interactions", "hybrid, item-kNN"),
    FeatureDef("user_rating_std", "user", "float", "Standard deviation of the user's ratings", "std(rating)", "interactions", "analysis"),
    FeatureDef("user_positive_ratio", "user", "float", "Share of the user's ratings that are >= relevance threshold", "count(rating>=4)/count", "interactions", "analysis"),
    FeatureDef("user_active_days", "user", "int", "Distinct calendar days with at least one rating", "nunique(event_date)", "interactions", "analysis"),
    FeatureDef("user_tenure_days", "user", "float", "Days between first and last rating", "max(ts)-min(ts)", "interactions", "analysis"),
    FeatureDef("user_days_since_last_interaction", "user", "float", "Recency: days between the last rating and the snapshot time", "as_of - max(ts)", "interactions", "analysis, monitoring"),
    FeatureDef("user_preferred_genre", "user", "category", "Genre with the most positive ratings (min. support applies)", "argmax_g count(rating>=4 & genre=g)", "interactions + items", "hybrid (explainability)"),
    FeatureDef("user_genre_entropy", "user", "float", "Shannon entropy of the user's genre distribution (taste breadth)", "-sum p_g log2 p_g", "interactions + items", "analysis"),
    FeatureDef("user_mainstreamness", "user", "float", "Mean log-popularity of items the user rated", "mean(log1p(item_rating_count))", "interactions + item features", "hybrid"),
    FeatureDef("user_avg_release_year", "user", "float", "Mean release year of items the user rated", "mean(release_year)", "interactions + items", "hybrid"),
    FeatureDef("age", "user", "int", "User age (imputed by occupation median where missing/invalid)", "cleaned u.user.age", "users", "cold-start"),
    FeatureDef("age_bucket", "user", "category", "Age band <18, 18-24, 25-34, 35-44, 45-54, 55+", "cut(age)", "users", "cold-start segments"),
    FeatureDef("gender", "user", "category", "M / F / U (unknown)", "cleaned u.user.gender", "users", "cold-start segments"),
    FeatureDef("occupation", "user", "category", "Lower-cased occupation", "cleaned u.user.occupation", "users", "analysis"),
    FeatureDef("is_cold_start", "user", "bool", "True if the user has no rating before the snapshot time", "user_rating_count == 0", "derived", "routing (hybrid)"),
]

ITEM_FEATURES = [
    FeatureDef("item_rating_count", "item", "int", "Number of ratings received (popularity)", "count(ratings)", "interactions", "popularity, hybrid"),
    FeatureDef("item_avg_rating", "item", "float", "Raw mean rating (NaN if never rated)", "mean(rating)", "interactions", "analysis"),
    FeatureDef("item_bayesian_avg", "item", "float", "Bayesian-average rating shrunk toward the global mean (m = 20)", "(n*mean + m*mu)/(n+m)", "interactions", "popularity, hybrid"),
    FeatureDef("item_rating_std", "item", "float", "Standard deviation of the item's ratings (controversy)", "std(rating)", "interactions", "analysis"),
    FeatureDef("item_popularity_rank", "item", "int", "Rank by rating count (1 = most popular)", "rank(count, desc)", "interactions", "popularity"),
    FeatureDef("item_log_popularity", "item", "float", "log(1 + rating count)", "log1p(count)", "interactions", "hybrid"),
    FeatureDef("item_days_since_last_rating", "item", "float", "Days between the item's last rating and the snapshot time", "as_of - max(ts)", "interactions", "analysis, monitoring"),
    FeatureDef("item_release_year", "item", "int", "Release year (imputed from title when release date is missing)", "year(release_date)", "items", "hybrid"),
    FeatureDef("item_age_years", "item", "float", "Years between release date and snapshot time", "(as_of - release_date)/365.25", "items", "analysis"),
    FeatureDef("item_primary_genre", "item", "category", "First listed genre of the item", "first genre flag = 1", "items", "diversity, analysis"),
    FeatureDef("item_n_genres", "item", "int", "Number of genres the item belongs to", "sum(genre flags)", "items", "analysis"),
    *[FeatureDef(f"genre_{g}", "item", "int", f"1 if the item is tagged '{g}'", "one-hot genre flag", "items", "content model, diversity") for g in GENRES],
]

USER_ITEM_FEATURES = [
    FeatureDef("ui_genre_affinity", "user_item", "float", "Mean of the user's shrunk genre preference over the item's genres", "mean_{g in G(i)} sum(r-mu_u)/(n_ug+5)", "user genre profile + items", "hybrid"),
    FeatureDef("ui_popularity_gap", "user_item", "float", "Item log-popularity minus the user's mainstreamness", "item_log_popularity - user_mainstreamness", "user + item features", "hybrid"),
    FeatureDef("ui_release_year_gap", "user_item", "float", "Item release year minus user's average release year", "item_release_year - user_avg_release_year", "user + item features", "analysis"),
]

ALL_FEATURES = USER_FEATURES + ITEM_FEATURES + USER_ITEM_FEATURES


def catalog_rows() -> list[dict]:
    return [asdict(f) for f in ALL_FEATURES]
