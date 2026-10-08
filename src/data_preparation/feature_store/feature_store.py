import pyspark.sql.functions as F
from databricks.feature_engineering import FeatureEngineeringClient


class FeatureStore:
    def __init__(self, spark, logging, catalog_name, gold_schema_name, featured_gold_table_name, feature_schema, feature_table_name, label_table_name):
        self.spark = spark
        self.logging = logging
        self.fe = FeatureEngineeringClient()
        self.catalog_name = catalog_name
        self.gold_schema_name = gold_schema_name
        self.featured_gold_table_name = featured_gold_table_name
        self.feature_schema = feature_schema
        self.feature_table_name = feature_table_name
        self.label_table_name = label_table_name

    def read_gold_table(self):
        gold_table_path = f"{self.catalog_name}.{self.gold_schema_name}.{self.featured_gold_table_name}"

        self.logging.info(f"Loading the feature table from the {gold_table_path}")
        self.data = self.spark.read.table(gold_table_path)
        self.logging.info("Gold feature table loaded in the dataframe.")

    def add_primary_key(self):
        self.logging.info("Adding primary key: feature_uid")
        self.data = self.data.withColumn("feature_uid", F.concat(F.col("transaction_date").cast("string"),F.lit('_'), F.col("product_name"),F.lit('_'),F.col("destination_city")))
        self.logging.info("Primary key is added. Key: feature_uid")

    def register_feature_table(self):
        feature_table_path = f"{self.catalog_name}.{self.feature_schema}.{self.feature_table_name}"
        
        self.logging.info("Selected the req columns for the feature table")
        feature_data = self.data.select(["feature_uid", "transaction_date", "product_name", "destination_city", "avg_unit_price", "total_inventory", "day_of_week", "month", "demand_lag_1", "demand_lag_7", "moving_avg_7"])

        if not self.spark.catalog.tableExists(feature_table_path):
            tags = {'name':"daily_demand_features", 'domain':"oil&gas", 'key_columns':"feature_uid, transaction_date, product_name, destination_city","categorical_columns":"product_name, destination_city", "numeric_cols":"total_inventory,day_of_week,month,demand_lag_1,demand_lag_7,moving_avg_7"}
            
            schema = feature_data.schema

            self.logging.info(f"Creating the Feature table at {feature_table_path}")
            self.fe.create_table(name=feature_table_path, primary_keys=["feature_uid"], schema=schema, description="Raw copy of the Feature data", tags=tags)
            self.logging.info("Feature table is created.")

        self.logging.info(f"Inserting the table with the data. Table: {feature_table_path}")
        self.fe.write_table(name=feature_table_path, df=feature_data, mode="merge")
        self.logging.info(f"Data Inserted Table: {feature_table_path}")

    def save_label_data(self):
        label_table_path = f"{self.catalog_name}.{self.gold_schema_name}.{self.label_table_name}"

        self.logging.info("Selected the req columns for the label data table")
        label_data = self.data.select(["feature_uid", "total_demand"])

        self.logging.info(f"Writing the label data. Path: {label_table_path}")
        label_data.write.format("delta")\
                        .mode("overwrite")\
                        .saveAsTable(label_table_path)
        self.logging.info(f"Label data saved at {label_table_path}")
        