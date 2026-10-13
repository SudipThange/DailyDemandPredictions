import pytest

CATALOG = "supply_chain_daily_demand"
GOLD_SCHEMA = "gold"

TABLES = [
    f"{CATALOG}.{GOLD_SCHEMA}.aggregated_daily_demand",
    f"{CATALOG}.{GOLD_SCHEMA}.features_daily_demand",
]


@pytest.mark.parametrize("table_name", TABLES)
def test_gold_table_exists(spark, table_name):
    """Verify that the Gold table exists."""
    assert spark.catalog.tableExists(table_name), (
        f"Gold table does not exist: {table_name}"
    )


@pytest.mark.parametrize("table_name", TABLES)
def test_gold_table_has_records(spark, table_name):
    """Verify that the Gold table contains at least one record."""
    df = spark.table(table_name)

    assert df.limit(1).count() > 0, (
        f"Gold table is empty: {table_name}"
    )


@pytest.mark.parametrize("table_name", TABLES)
def test_gold_table_has_unique_column_names(spark, table_name):
    """Verify that column names are unique."""
    df = spark.table(table_name)

    assert len(df.columns) == len(set(df.columns)), (
        f"Duplicate column names found in: {table_name}"
    )


def test_aggregated_gold_required_columns(spark):
    """Verify required columns in the aggregated Gold table."""
    table_name = (
        f"{CATALOG}.{GOLD_SCHEMA}.aggregated_daily_demand"
    )

    actual_columns = set(spark.table(table_name).columns)

    required_columns = {
        "transaction_date",
        "product_type",
        "target_city",
        "daily_demand",
    }

    missing_columns = required_columns - actual_columns

    assert not missing_columns, (
        f"Missing columns in {table_name}: "
        f"{sorted(missing_columns)}"
    )
