
import logging
from unittest.mock import MagicMock

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

    reader.option.assert_any_call("header", "true")
    reader.option.assert_any_call("sep", ",")
    reader.option.assert_any_call("inferSchema", "true")

    expected_path = (
        f"/Volumes/{bronze.catalog_name}/"
        f"{bronze.raw_schema_name}/"
        f"{bronze.volume_name}/"
        f"{bronze.file_name}"
    )

    reader.load.assert_called_once_with(expected_path)


def test_write_to_bronze(bronze):
    dataframe = MagicMock()
    dataframe.drop.return_value = dataframe
    dataframe.withColumn.return_value = dataframe

    bronze.data = dataframe

    bronze.write_to_bronze()

    dataframe.drop.assert_called_once_with(
        "ingestion_timestamp",
        "source_system",
        "operation_type",
    )

    dataframe.withColumn.assert_called_once()
    assert dataframe.withColumn.call_args.args[0] == (
        "bronze_ingestion_timestamp"
    )

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
