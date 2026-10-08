import logging

import yaml
from data_preparation.bronze.bronze_ingestion import Bronze
from data_preparation.feature_store.feature_store import FeatureStore
from data_preparation.gold.gold_aggregations import Gold
from pyspark.sql import SparkSession
from data_preparation.silver.silver_transformation import Silver

# logging configurations
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# yaml load function
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config

def run_bronze(config, spark, logging):
    catalog_name = config["data_preparation"]["catalog_name"]
    raw_schema_name = config["data_preparation"]["bronze"]["source"]["schema_name"]
    volume_name = config["data_preparation"]["bronze"]["source"]["volume_name"]
    file_name = config["data_preparation"]["bronze"]["source"]["file_name"]
    bronze_schema_name = config["data_preparation"]["bronze"]["target"]["schema_name"]
    bronze_table_name = config["data_preparation"]["bronze"]["target"]["table_name"]

    bronze_obj = Bronze(spark=spark, logging=logging, catalog_name=catalog_name, bronze_schema_name=bronze_schema_name, raw_schema_name=raw_schema_name, volume_name=volume_name, bronze_table_name=bronze_table_name, file_name=file_name)

    bronze_obj.load_raw_data()
    bronze_obj.write_to_bronze()

def run_silver(config, spark, logging):
    catalog_name = config["data_preparation"]["catalog_name"]
    bronze_schema_name = config["data_preparation"]["silver"]["source"]["bronze_schema_name"]
    bronze_table_name = config["data_preparation"]["silver"]["source"]["bronze_table_name"]
    silver_schema_name = config["data_preparation"]["silver"]["target"]["silver_schema_name"]
    silver_table_name = config["data_preparation"]["silver"]["target"]["silver_table_name"]

    silver_obj = Silver(spark=spark, logging=logging, catalog_name=catalog_name, bronze_schema_name=bronze_schema_name, bronze_table_name=bronze_table_name, silver_schema_name=silver_schema_name, silver_table_name=silver_table_name)

    silver_obj.load_from_bronze()
    silver_obj.remove_duplicates()
    silver_obj.write_to_silver()

def run_gold(config, spark, logging):
    catalog_name = config["data_preparation"]["catalog_name"]
    silver_schema_name = config["data_preparation"]["gold"]["source"]["silver_schema_name"]
    silver_table_name = config["data_preparation"]["gold"]["source"]["silver_table_name"]
    gold_schema_name = config["data_preparation"]["gold"]["target"]["gold_schema_name"]
    aggregated_gold_table_name = config["data_preparation"]["gold"]["target"]["aggregated_gold_table_name"]
    featured_gold_table_name =  config["data_preparation"]["gold"]["target"]["featured_gold_table_name"]

    gold_obj = Gold(catalog_name=catalog_name, silver_schema_name=silver_schema_name, silver_table_name=silver_table_name,gold_schema_name=gold_schema_name, aggregated_gold_table_name=aggregated_gold_table_name, featured_gold_table_name=featured_gold_table_name,spark=spark, logging=logging)

    gold_obj.load_silver_table()
    gold_obj.write_aggregated_table()
    gold_obj.write_feature_table()

def run_feature_store(config, spark, logging):
    catalog_name = config["data_preparation"]["catalog_name"]
    gold_schema_name = config["data_preparation"]["feature_store"]["source"]["gold_schema_name"]
    featured_gold_table_name = config["data_preparation"]["feature_store"]["source"]["featured_gold_table_name"]
    feature_schema = config["data_preparation"]["feature_store"]["target"]["feature_schema"]
    feature_table_name = config["data_preparation"]["feature_store"]["target"]["feature_table_name"]
    label_table_name = config["data_preparation"]["feature_store"]["target"]["label_table_name"]

    feature_store_obj = FeatureStore(spark=spark, logging=logging, catalog_name=catalog_name, gold_schema_name=gold_schema_name, featured_gold_table_name=featured_gold_table_name, feature_schema=feature_schema, feature_table_name=feature_table_name, label_table_name=label_table_name)

    feature_store_obj.read_gold_table()
    feature_store_obj.add_primary_key()
    feature_store_obj.register_feature_table()
    feature_store_obj.save_label_data()

# main execution part
if __name__  == "__main__":
    spark = SparkSession.builder.getOrCreate()
    config = load_config(r"/Workspace/Users/sudipthange856@gmail.com/supply_chain_project/config/config.yml")
    #print(config["data_preparation"]["bronze"])

    logger.info("Bronze - raw data ingestion started. 🔃")
    run_bronze(config, spark, logger)
    logger.info("Bronze layer completed. ✅")

    logger.info("Silver - transformation started on the bronze raw data. 🔃")
    run_silver(config, spark, logger)
    logger.info("Silver layer completed. ✅")

    logger.info("Gold - aggregations & features engineering started on the transformed data. 🔃")
    run_gold(config, spark, logger)
    logger.info("Gold layer completed. ✅")

    logger.info("Feature Store - started registering the feature data in the feature store. 🔃")
    run_feature_store(config, spark, logger)
    logger.info("Feature Store layer completed. ✅")


   
