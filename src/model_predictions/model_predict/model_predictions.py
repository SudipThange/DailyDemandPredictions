import mlflow
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from mlflow import MlflowClient
import joblib
import pandas as pd

class ModelPredictions:
    def __init__(self, spark, logging, catalog_name, schema_name, table_name, model_path, output_table_name):
        self.spark = spark
        self.logging = logging
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.table_name = table_name
        self.model_path = model_path
        self.output_table_name = output_table_name
        self.client = MlflowClient()

    def load_data(self):
        table_path = f"{self.catalog_name}.{self.schema_name}.{self.table_name}"

        self.data = self.spark.read.table(table_path)

    def get_champion_model(self):
        model_name = "supply_chain_daily_demand.ml_model.random_forest_demand"

        champion = self.client.get_model_version_by_alias(name=model_name,alias="champion")

        return champion

    def get_run_id(self):
        champion = self.get_champion_model()

        return champion.run_id
    
    def encode_categories(self):
        self.load_data()

        encoder_path = self.client.download_artifacts(run_id=self.get_run_id(),path="encoder/encoder.pkl")
        encoder = joblib.load(encoder_path)

        categorical_cols = ["product_name", "destination_city"]

        categorical_data = self.data.select(["feature_uid"] + categorical_cols).toPandas()

        encoded_data = encoder.transform(categorical_data[categorical_cols])

        encoded_cols = encoder.get_feature_names_out(categorical_cols)
        encoded_df = pd.DataFrame(encoded_data,columns=encoded_cols)
        encoded_df["encoded_feature_uid"] = categorical_data["feature_uid"]
        encoded_spark_df = self.spark.createDataFrame(encoded_df)
        
        encoded_final_df = self.data.join(encoded_spark_df, self.data.feature_uid  == encoded_spark_df.encoded_feature_uid, "inner")\
                               .drop("encoded_feature_uid", "product_name", "destination_city")

        columns = encoded_final_df.columns
        cleaned_columns = []
        for column in columns:
            cleaned_columns.append(column.replace(" ", "_"))

        encoded_final_df = encoded_final_df.toDF(*cleaned_columns)
        return encoded_final_df

    def convert_to_pandas(self, encoded_final_df):
        X_test = encoded_final_df
        X_test = X_test.drop("feature_uid", "total_demand", "transaction_date")
        y_test = encoded_final_df.select("total_demand")

        X_test = X_test.toPandas()
        y_test = y_test.toPandas()

        y_test = y_test["total_demand"]

        return (X_test, y_test)
    
    def predict_demand(self):
        encoded_final_df = self.encode_categories()
        X_test, y_test = self.convert_to_pandas(encoded_final_df)

        model = mlflow.pyfunc.load_model(self.model_path)

        predictions = model.predict(X_test)

        print("MAE:", mean_absolute_error(predictions, y_test))
        print("MSE:", mean_squared_error(predictions, y_test))
        print("R2:", r2_score(predictions, y_test))

        output_table_path = f"{self.catalog_name}.{self.schema_name}.{self.output_table_name}"
        output_data = self.data.select("feature_uid", "transaction_date", "product_name", "destination_city", "total_demand").toPandas()

        output_data["predicted_demand"] = predictions

        output_data = self.spark.createDataFrame(output_data)               

        output_data.write.format("delta")\
                         .mode("overwrite")\
                         .saveAsTable(output_table_path)

        print(f"Table loaded to the path: {output_table_path}")