import pyspark.sql.functions as F
from pyspark.sql.window import Window


class Silver:
    def __init__(self, spark, logging, catalog_name, bronze_schema_name, bronze_table_name, silver_schema_name, silver_table_name):
        self.spark = spark
        self.logging = logging
        self.catalog_name = catalog_name
        self.bronze_schema_name = bronze_schema_name
        self.bronze_table_name = bronze_table_name
        self.silver_schema_name = silver_schema_name
        self.silver_table_name = silver_table_name

    def load_from_bronze(self):
        bronze_table_path = f"{self.catalog_name}.{self.bronze_schema_name}.{self.bronze_table_name}"

        self.logging.info(f"Loading the delta table from path: {bronze_table_path}.")
        self.data = self.spark.read.table(bronze_table_path)
        self.logging.info("Bronze delta table is loaded.")

    def remove_duplicates(self):
        window_config = Window.partitionBy("transaction_id").orderBy("transaction_date")

        count_before_removing = self.data.count()

        self.logging.info(f"Duplication removing started, total record count: {count_before_removing}")
        self.data = self.data.withColumn(
            "duplicate_count",
            F.row_number().over(window_config)
        ).filter(
            F.col("duplicate_count") == 1
        ).drop("duplicate_count")

        count_after_removing = self.data.count()
        self.logging.info(f"Duplicates are removed: {count_before_removing - count_after_removing}")

    def write_to_silver(self):
        self.logging.info("Adding the silver_ingestion_timestamp column.")
        self.data = self.data.withColumn("silver_ingestion_timestamp", F.current_timestamp())

        silver_table_path = f"{self.catalog_name}.{self.silver_schema_name}.{self.silver_table_name}"
        self.logging.info(f"Writing the dataframe to the delta table. Path: {silver_table_path}")

        self.data.write.format("delta")\
                       .mode("overwrite")\
                       .saveAsTable(silver_table_path)

        self.logging.info("Table saved.")
