import pyspark.sql.functions as F


# bronze class
class Bronze:
    def __init__(self, spark, logging, catalog_name, bronze_schema_name, raw_schema_name, volume_name, bronze_table_name, file_name):
        self.spark = spark
        self.logging = logging
        self.catalog_name = catalog_name
        self.bronze_schema_name = bronze_schema_name
        self.raw_schema_name = raw_schema_name
        self.volume_name = volume_name
        self.bronze_table_name = bronze_table_name
        self.file_name = file_name

    def load_raw_data(self):
        file_path = f"/Volumes/{self.catalog_name}/{self.raw_schema_name}/{self.volume_name}/{self.file_name}"

        self.logging.info(f"File loading from the file path: {file_path}.")
        self.data = self.spark.read.format("csv")\
                              .option("header", "true")\
                              .option("sep", ",")\
                              .option("inferSchema", "true")\
                              .load(file_path)

        self.logging.info(f"File loaded and saved in the dataframe.")
        #self.data.show()

    def write_to_bronze(self):
        bronze_table_path = f"{self.catalog_name}.{self.bronze_schema_name}.{self.bronze_table_name}"

        self.logging.info("Dropping columns: ingestion_timestamp, source_system, operation_type")
        self.data = self.data.drop("ingestion_timestamp", "source_system", "operation_type")

        self.logging.info("Adding the ingesting timestamp columns: bronze_ingestion_timestamp")
        self.data = self.data.withColumn("bronze_ingestion_timestamp", F.current_timestamp())

        self.logging.info(f"Writing the loaded dataframe to delta table, table_path: {bronze_table_path}.")
        self.data.write.format("delta")\
                       .mode("overwrite")\
                       .saveAsTable(bronze_table_path)
        self.logging.info(f"Data saved in delta table, path: {bronze_table_path}")
