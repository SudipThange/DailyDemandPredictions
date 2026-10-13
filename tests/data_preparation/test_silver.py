
import logging
from unittest.mock import MagicMock, patch

import pytest
import yaml

from data_preparation.silver.silver_transformation import Silver


@pytest.fixture
def config():
    with open("config/config.yml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


@pytest.fixture
def silver(config):
    settings = config["data_preparation"]
    source = settings["silver"]["source"]
    target = settings["silver"]["target"]

    return Silver(
        spark=MagicMock(),
        logging=logging.getLogger("test_silver"),
        catalog_name=settings["catalog_name"],
        bronze_schema_name=source["bronze_schema_name"],
        bronze_table_name=source["bronze_table_name"],
        silver_schema_name=target["silver_schema_name"],
        silver_table_name=target["silver_table_name"],
    )


def test_load_from_bronze(silver):
    silver.load_from_bronze()

    silver.spark.read.table.assert_called_once_with(
        f"{silver.catalog_name}."
        f"{silver.bronze_schema_name}."
        f"{silver.bronze_table_name}"
    )


@patch("data_preparation.silver.silver_transformation.Window")
def test_remove_duplicates(mock_window, silver):
    mock_window.partitionBy.return_value.orderBy.return_value = MagicMock()

    dataframe = MagicMock()
    silver.data = dataframe

    dataframe.count.side_effect = [5, 3]
    dataframe.withColumn.return_value = dataframe

    filtered_dataframe = MagicMock()
    dataframe.withColumn.return_value.filter.return_value = (
        filtered_dataframe
    )
    filtered_dataframe.drop.return_value = filtered_dataframe

    # Execute the method
    silver.remove_duplicates()

    # Verify the Window configuration
    mock_window.partitionBy.assert_called_once_with("transaction_id")
    mock_window.partitionBy.return_value.orderBy.assert_called_once_with(
        "transaction_date"
    )


@patch(
    "data_preparation.silver.silver_transformation.F.current_timestamp"
)
def test_write_to_silver(mock_current_timestamp, silver):
    dataframe = MagicMock()
    dataframe.withColumn.return_value = dataframe
    silver.data = dataframe

    silver.write_to_silver()

    # Verify timestamp creation
    mock_current_timestamp.assert_called_once_with()
    dataframe.withColumn.assert_called_once_with(
        "silver_ingestion_timestamp",
        mock_current_timestamp.return_value,
    )

    # Verify Delta write configuration
    dataframe.write.format.assert_called_once_with("delta")
    dataframe.write.format.return_value.mode.assert_called_once_with(
        "overwrite"
    )

    (
        dataframe.write.format.return_value
        .mode.return_value.saveAsTable
        .assert_called_once_with(
            f"{silver.catalog_name}."
            f"{silver.silver_schema_name}."
            f"{silver.silver_table_name}"
        )
    )
