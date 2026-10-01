import yaml
from pyspark.sql import SparkSession
import logging
from model_predict.model_predictions import ModelPredictions

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

def run_model_predictions(config, spark, logger):
    catalog_name = config["model_predictions"]["catalog_name"]
    schema_name = config["model_predictions"]["source"]["schema_name"]
    table_name = config["model_predictions"]["source"]["table_name"]
    model_path = config["model_predictions"]["source"]["model_path"]
    output_table_name = config["model_predictions"]["source"]["output_table_name"]

    model_pred_obj = ModelPredictions(spark=spark, logging=logger, catalog_name=catalog_name, schema_name=schema_name, table_name=table_name, model_path=model_path, output_table_name=output_table_name)

    model_pred_obj.predict_demand()

if __name__  == "__main__":
    spark = SparkSession.builder.getOrCreate()
    config = load_config(r"/Workspace/Users/sudipthange856@gmail.com/supply_chain_project/config/config.yml")

    logging.info("Model prediction started. 🔃")
    run_model_predictions(config, spark, logger)
    logging.info("Model predictions completed. ✅")

