import logging

import yaml
from pyspark.sql import SparkSession
from train_model.model_training import ModelTraining

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

def run_model_training(config, spark, logger):
    catalog_name = config["model_training"]["catalog_name"]
    feature_schema = config["model_training"]["source"]["feature_schema"]
    feature_table_name = config["model_training"]["source"]["feature_table_name"]
    label_schema = config["model_training"]["source"]["lable_schema"]
    label_table_name = config["model_training"]["source"]["label_table_name"]

    model_training_obj = ModelTraining(spark=spark, logging=logger, catalog_name=catalog_name, feature_schema=feature_schema, feature_table_name=feature_table_name, lable_schema=label_schema, label_table_name=label_table_name)

    model_training_obj.load_data()
    model_training_obj.start_training()
    model_training_obj.register_model()


if __name__  == "__main__":
    spark = SparkSession.builder.getOrCreate()
    config = load_config(r"/Workspace/Users/sudipthange856@gmail.com/supply_chain_project/config/config.yml")

    logger.info("Model training & registering started. 🔃")
    run_model_training(config, spark, logger)
    logger.info("Model training & registration completed. ✅")

