
import logging
from unittest.mock import MagicMock, patch

import pytest
import yaml

from data_preparation.bronze.bronze_ingestion import Bronze


@pytest.fixture
def config():
    with open("config/config.yml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


@pytest.fixture
def bronze(config):
    settings = config["data_preparation"]
    source = settings["bronze"]["source"]
    target = settings["bronze"]["target"]

    return Bronze(
        spark=MagicMock(),
        logging=logging.getLogger("test_bronze"),
        catalog_name=settings["catalog_name"],
        bronze_schema_name=target["schema_name"],
        raw_schema_name=source["schema_name"],
        volume_name=source["volume_name"],
        bronze_table_name=target["table_name"],
        file_name=source["file_name"],
    )


def test_load_raw_data(bronze):
    bronze.load_raw_data()

    reader = bronze.spark.read.format.return_value

    bronze.spark.read.format.assert_called_once_with("csv")

    # Verify the chained CSV reader options
    reader.option.assert_called_once_with("header", "true")

    option_1 = reader.option.return_value
    option_1.option.assert_called_once_with("sep", ",")

    option_2 = option_1.option.return_value
    option_2.option.assert_called_once_with("inferSchema", "true")

    expected_path = (
        f"/Volumes/{bronze.catalog_name}/"
        f"{bronze.raw_schema_name}/"
        f"{bronze.volume_name}/"
        f"{bronze.file_name}"
    )

    option_2.option.return_value.load.assert_called_once_with(
        expected_path
    )


@patch(
    "data_preparation.bronze.bronze_ingestion.F.current_timestamp"
)
def test_write_to_bronze(mock_current_timestamp, bronze):
    dataframe = MagicMock()

    dataframe.drop.return_value = dataframe
    dataframe.withColumn.return_value = dataframe

    bronze.data = dataframe

    bronze.write_to_bronze()

    # Verify unwanted columns are removed
    dataframe.drop.assert_called_once_with(
        "ingestion_timestamp",
        "source_system",
        "operation_type",
    )

    # Verify the ingestion timestamp is added
    mock_current_timestamp.assert_called_once_with()

    dataframe.withColumn.assert_called_once_with(
        "bronze_ingestion_timestamp",
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
            f"{bronze.catalog_name}."
            f"{bronze.bronze_schema_name}."
            f"{bronze.bronze_table_name}"
        )
    )
