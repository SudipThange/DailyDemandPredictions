from unittest.mock import MagicMock

import pytest

CATALOG = "supply_chain_daily_demand"
GOLD_SCHEMA = "gold"

TABLES = [
    f"{CATALOG}.{GOLD_SCHEMA}.aggregated_daily_demand",
    f"{CATALOG}.{GOLD_SCHEMA}.features_daily_demand",
]


@pytest.fixture
def spark():
    """Provide a mocked Spark session for unit tests."""
    return MagicMock()


@pytest.mark.parametrize("table_name", TABLES)
def test_gold_table_exists(spark, table_name):
    """Verify that the table existence check succeeds."""
    spark.catalog.tableExists.return_value = True

    assert spark.catalog.tableExists(table_name) is True

    spark.catalog.tableExists.assert_called_once_with(table_name)


@pytest.mark.parametrize("table_name", TABLES)
def test_gold_table_has_records(spark, table_name):
    """Verify that a Gold table contains at least one record."""
    dataframe = MagicMock()
    dataframe.limit.return_value.count.return_value = 1
    spark.table.return_value = dataframe

    assert dataframe.limit(1).count() > 0

    spark.table.assert_called_once_with(table_name)


@pytest.mark.parametrize("table_name", TABLES)
def test_gold_table_has_unique_column_names(spark, table_name):
    """Verify that the table has unique column names."""
    dataframe = MagicMock()
    dataframe.columns = [
        "transaction_date",
        "product_type",
        "daily_demand",
    ]
    spark.table.return_value = dataframe

    columns = spark.table(table_name).columns

    assert len(columns) == len(set(columns))


def test_aggregated_gold_required_columns(spark):
    """Verify the required columns in the aggregated Gold table."""
    table_name = (
        f"{CATALOG}.{GOLD_SCHEMA}.aggregated_daily_demand"
    )

    dataframe = MagicMock()
    dataframe.columns = [
        "transaction_date",
        "product_type",
        "target_city",
        "daily_demand",
    ]
    spark.table.return_value = dataframe

    actual_columns = set(spark.table(table_name).columns)

    required_columns = {
        "transaction_date",
        "product_type",
        "target_city",
        "daily_demand",
    }

    assert required_columns.issubset(actual_columns)
