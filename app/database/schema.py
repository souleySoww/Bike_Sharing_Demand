from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


ERROR_CATEGORIES = {
    "completeness": ("null_required_value",),
    "validity": ("invalid_range",),
    "consistency": (
        "duplicate_instant",
        "holiday_workingday_inconsistency",
    ),
    "schema": ("missing_required_column",),
    "type": ("invalid_numeric_type",),
}
INTEGER_FEATURES = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
]
FLOAT_FEATURES = ["temp", "atemp", "hum", "windspeed"]


def _category_rows() -> str:
    rows = []
    for category, counters in ERROR_CATEGORIES.items():
        total = " + ".join(
            f"COALESCE((source.counts->>'{counter}')::int, 0)"
            for counter in counters
        )
        rows.append(f"('{category}', {total})")
    return ",\n            ".join(rows)


def _feature_columns() -> str:
    columns = [
        f"(features::jsonb->>'{name}')::int AS {name}"
        for name in INTEGER_FEATURES
    ] + [
        f"(features::jsonb->>'{name}')::double precision AS {name}"
        for name in FLOAT_FEATURES
    ]
    return ",\n        ".join(columns)


# error_counts is JSON or JSONB depending on whether the API or Airflow created
# ingestion_stats first, so every view casts through jsonb.
VIEWS = {
    "v_ingestion_errors": f"""
        SELECT
            stat.id AS ingestion_id,
            stat.created_at,
            stat.file_name,
            stat.total_rows,
            category.name AS category,
            category.error_count
        FROM ingestion_stats AS stat
        CROSS JOIN LATERAL (
            SELECT stat.error_counts::jsonb AS counts
        ) AS source
        CROSS JOIN LATERAL (
            VALUES
            {_category_rows()}
        ) AS category(name, error_count)
    """,
    "v_ingestion_error_types": """
        SELECT
            stat.id AS ingestion_id,
            stat.created_at,
            stat.file_name,
            error.key AS error_type,
            error.value::int AS error_count
        FROM ingestion_stats AS stat
        CROSS JOIN LATERAL jsonb_each_text(stat.error_counts::jsonb) AS error
    """,
    "v_baseline_numeric": """
        SELECT
            stats.model_version,
            stats.is_champion,
            stats.created_at,
            feature.key AS feature,
            (feature.value->>'mean')::double precision AS baseline_mean,
            (feature.value->>'std')::double precision AS baseline_std,
            (feature.value->>'min')::double precision AS baseline_min,
            (feature.value->>'max')::double precision AS baseline_max
        FROM training_stats AS stats
        CROSS JOIN LATERAL jsonb_each(
            stats.statistics::jsonb->'numerical'
        ) AS feature
    """,
    "v_baseline_categorical": """
        SELECT
            stats.model_version,
            stats.is_champion,
            stats.created_at,
            feature.key AS feature,
            category.key AS category,
            category.value::double precision AS baseline_proportion
        FROM training_stats AS stats
        CROSS JOIN LATERAL jsonb_each(
            stats.statistics::jsonb->'categorical'
        ) AS feature
        CROSS JOIN LATERAL jsonb_each_text(feature.value) AS category
    """,
    "v_prediction_features": f"""
        SELECT
            id,
            created_at,
            source,
            model_name,
            model_version,
            prediction,
            {_feature_columns()}
        FROM predictions
    """,
    # One row per baseline bin: numbered bins for numerical features and the
    # target, one bin per category for categorical features.
    "v_baseline_bins": """
        SELECT
            stats.model_version,
            stats.is_champion,
            stats.created_at,
            feature.key AS feature,
            'numerical' AS kind,
            share.position::text AS bin_key,
            (feature.value->'edges'->>(share.position::int - 1))
                ::double precision AS lower_edge,
            (feature.value->'edges'->>(share.position::int))
                ::double precision AS upper_edge,
            share.value::double precision AS baseline_proportion
        FROM training_stats AS stats
        CROSS JOIN LATERAL jsonb_each(stats.statistics::jsonb->'bins')
            AS feature
        CROSS JOIN LATERAL jsonb_array_elements_text(
            feature.value->'proportions'
        ) WITH ORDINALITY AS share(value, position)
        UNION ALL
        SELECT
            stats.model_version,
            stats.is_champion,
            stats.created_at,
            feature.key AS feature,
            'categorical' AS kind,
            category.key AS bin_key,
            NULL AS lower_edge,
            NULL AS upper_edge,
            category.value::double precision AS baseline_proportion
        FROM training_stats AS stats
        CROSS JOIN LATERAL jsonb_each(
            stats.statistics::jsonb->'categorical'
        ) AS feature
        CROSS JOIN LATERAL jsonb_each_text(feature.value) AS category
    """,
}

# PSI from course 5: sum of (actual - expected) * ln(actual / expected) over
# the champion's bins, per time bucket. p_feature is a feature name, or
# 'prediction' to compare predictions with the training target. Empty bins
# use 0.0001 so the logarithm stays finite.
FUNCTIONS = {
    "feature_psi": """
        CREATE OR REPLACE FUNCTION feature_psi(
            p_feature text,
            p_bucket interval,
            p_from timestamptz,
            p_to timestamptz
        )
        RETURNS TABLE (
            bucket_start timestamptz,
            psi double precision,
            sample_size bigint
        )
        LANGUAGE sql
        STABLE
        AS $psi$
        WITH baseline AS (
            SELECT bins.bin_key, bins.lower_edge, bins.baseline_proportion
            FROM v_baseline_bins AS bins
            WHERE bins.is_champion
              AND bins.feature = CASE
                  WHEN p_feature = 'prediction' THEN 'cnt'
                  ELSE p_feature
              END
        ),
        lower_edges AS (
            SELECT array_agg(lower_edge ORDER BY bin_key::int) AS edges
            FROM baseline
            WHERE lower_edge IS NOT NULL
        ),
        serving AS (
            SELECT
                date_bin(p_bucket, rows.created_at, p_from) AS bucket_start,
                CASE
                    WHEN lower_edges.edges IS NULL THEN rows.raw_value
                    ELSE GREATEST(
                        1,
                        width_bucket(
                            rows.raw_value::double precision,
                            lower_edges.edges
                        )
                    )::text
                END AS bin_key
            FROM (
                SELECT
                    created_at,
                    CASE
                        WHEN p_feature = 'prediction' THEN prediction::text
                        ELSE features::jsonb->>p_feature
                    END AS raw_value
                FROM predictions
                WHERE created_at >= p_from AND created_at < p_to
            ) AS rows
            CROSS JOIN lower_edges
            WHERE rows.raw_value IS NOT NULL
        ),
        counts AS (
            SELECT bucket_start, bin_key, count(*) AS n
            FROM serving
            GROUP BY bucket_start, bin_key
        ),
        totals AS (
            SELECT bucket_start, sum(n) AS total
            FROM counts
            GROUP BY bucket_start
        ),
        bin_keys AS (
            SELECT bin_key FROM baseline
            UNION
            SELECT bin_key FROM counts
        ),
        shares AS (
            SELECT
                totals.bucket_start,
                totals.total,
                GREATEST(
                    COALESCE(counts.n, 0)::double precision / totals.total,
                    0.0001
                ) AS actual,
                GREATEST(COALESCE(baseline.baseline_proportion, 0), 0.0001)
                    AS expected
            FROM totals
            CROSS JOIN bin_keys
            LEFT JOIN counts
                ON counts.bucket_start = totals.bucket_start
               AND counts.bin_key = bin_keys.bin_key
            LEFT JOIN baseline ON baseline.bin_key = bin_keys.bin_key
        )
        SELECT
            bucket_start,
            sum((actual - expected) * ln(actual / expected)) AS psi,
            max(total)::bigint AS sample_size
        FROM shares
        WHERE EXISTS (SELECT 1 FROM baseline)
        GROUP BY bucket_start
        ORDER BY bucket_start
        $psi$
    """,
}


def ensure_monitoring_schema(engine: Engine) -> None:
    """Upgrade tables that create_all cannot change, then rebuild the views."""
    with engine.begin() as connection:
        columns = {
            column["name"]
            for column in inspect(connection).get_columns("predictions")
        }
        if "model_version" not in columns:
            connection.execute(
                text(
                    "ALTER TABLE predictions "
                    "ADD COLUMN model_version VARCHAR(64)"
                )
            )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_predictions_model_version "
                "ON predictions (model_version)"
            )
        )
        # The views use PostgreSQL JSON functions; local SQLite only needs
        # the tables.
        if connection.dialect.name != "postgresql":
            return
        for name, query in VIEWS.items():
            connection.execute(text(f"DROP VIEW IF EXISTS {name}"))
            connection.execute(text(f"CREATE VIEW {name} AS {query}"))
        for definition in FUNCTIONS.values():
            connection.execute(text(definition))
