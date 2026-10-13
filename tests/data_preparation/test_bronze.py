
from unittest.mock import MagicMock, patch

from data_preparation.bronze.bronze_ingestion import Bronze


def test_load_raw_data(bronze):
    bronze.load_raw_data()

    reader = bronze.spark.read.format.return_value

    bronze.spark.read.format.assert_called_once_with("csv")

    # Each chained option() call returns a new mock
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

    # Verify timestamp creation without a real Spark context
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
