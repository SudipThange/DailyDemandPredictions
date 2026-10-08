import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import pyspark.sql.functions as F
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.feature_store import FeatureLookup
from mlflow.client import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql.window import Window
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder


# encoding class
class LabelEncoding:
    def __init__(self, spark):
        self.spark = spark
        self.encoder = OneHotEncoder(sparse_output=False,handle_unknown="ignore")

    def encode_categorical_data(self, data):
        name = data["name"]
        data = data["data"]

        categorical_cols = ["product_name", "destination_city"]

        categorical_data = data.select(["feature_uid"] + categorical_cols).toPandas()

        if name == "training_data":
            encoded_data = self.encoder.fit_transform(categorical_data[categorical_cols])
        else:
            encoded_data = self.encoder.transform(categorical_data[categorical_cols])

        encoded_cols = self.encoder.get_feature_names_out(categorical_cols)
        encoded_df = pd.DataFrame(encoded_data,columns=encoded_cols)
        encoded_df["encoded_feature_uid"] = categorical_data["feature_uid"]
        encoded_spark_df = self.spark.createDataFrame(encoded_df)
        
        encoded_final_df = data.join(encoded_spark_df, data.feature_uid  == encoded_spark_df.encoded_feature_uid, "inner")\
                               .drop("encoded_feature_uid", "product_name", "destination_city")

        columns = encoded_final_df.columns
        cleaned_columns = []
        for column in columns:
            cleaned_columns.append(column.replace(" ", "_"))

        encoded_final_df = encoded_final_df.toDF(*cleaned_columns)
        # print("Encoded Data")
        # display(encoded_final_df)

        return encoded_final_df
    
    def get_encoder(self):
        return self.encoder

# training class
class ModelTraining:
    def __init__(self, spark, logging, catalog_name, feature_schema, feature_table_name, lable_schema, label_table_name):
        self.spark = spark
        self.logging = logging
        self.fe = FeatureEngineeringClient()
        self.catalog_name = catalog_name
        self.feature_schema = feature_schema
        self.feature_table_name = feature_table_name
        self.lable_schema = lable_schema
        self.label_table_name = label_table_name
        mlflow.set_experiment("/Users/sudipthange856@gmail.com/supply_chain_demand")

    def load_data(self):
        feature_table_path =  f"{self.catalog_name}.{self.feature_schema}.{self.feature_table_name}"   
        label_table_path = f"{self.catalog_name}.{self.lable_schema}.{self.label_table_name}"

        # feature_table = fe.read_table(name=feature_table_path)     

        lable_table = self.spark.read.table(label_table_path)
        # lable_table = lable_table.withColumnRenamed("feature_uid", "labels_feature_uid")

        # self.data = feature_table.join(lable_table, feature_table.feature_uid == lable_table.labels_feature_uid, "inner")\
        #                          .drop(F.col("labels_feature_uid"))
        
        feature_lookups = [
            FeatureLookup(table_name=feature_table_path, lookup_key="feature_uid")
        ]

        training_set = self.fe.create_training_set(
            df=lable_table,feature_lookups=feature_lookups,label="total_demand"
        )

        self.data = training_set.load_df()

        self.data = self.data.fillna(0, subset=["demand_lag_1", "demand_lag_7"])

        # self.data.show()
        #print("Count after loading", self.data.count)
    
    def split_data(self, data):
        window = Window.orderBy("transaction_date")

        data = data.withColumn("row_number", F.row_number().over(window))

        total_count = data.count()
        threshold = int(total_count * 0.80)
        print("Spliting Threshold is:", threshold)

        training_data = data.filter(F.col("row_number") <= threshold)\
                                    .drop("row_number")

        testing_data = data.filter(F.col("row_number") > threshold)\
                                    .drop("row_number")

        # display(training_data)
        # display(testing_data)
        # print("Training data:", training_data.count())
        # print("Testing Data:", testing_data.count())

        return (training_data, testing_data)

    def convert_into_sets(self, encoded_training_data, encoded_testing_data, encoded_validation_data):
        X_train = encoded_training_data
        X_train = X_train.drop("feature_uid", "total_demand", "transaction_date")
        y_train = encoded_training_data.select("total_demand")

        X_validation = encoded_validation_data
        X_validation = X_validation.drop("feature_uid", "total_demand", "transaction_date")
        y_validation = encoded_validation_data.select("total_demand")

        X_test = encoded_testing_data
        X_test = X_test.drop("feature_uid", "total_demand", "transaction_date")
        y_test = encoded_testing_data.select("total_demand")

        X_train = X_train.toPandas()
        y_train = y_train.toPandas()
        X_validation = X_validation.toPandas()
        y_validation = y_validation.toPandas()
        X_test = X_test.toPandas()
        y_test = y_test.toPandas()

        y_train = y_train["total_demand"]
        y_validation = y_validation["total_demand"]
        y_test = y_test["total_demand"]

        return (X_train, y_train, X_validation, y_validation, X_test, y_test)

    def validate_model(self, y_validation, predictions):
        mae = mean_absolute_error(y_validation,predictions)
        mse = mean_squared_error(y_validation,predictions)
        r2 = r2_score(y_validation,predictions)

        return (mae, mse, r2)

    def save_to_table(self, info):
        table_path = info["table_path"]
        data = info["data"]

        columns = data.columns
        cleaned_columns = []
        for column in columns:
            cleaned_columns.append(column.replace(" ", "_"))

        data = data.toDF(*cleaned_columns)

        data.write.format("delta")\
                  .mode("overwrite")\
                  .saveAsTable(table_path)

    def start_training(self):
        with mlflow.start_run(run_name="RandomForestRegressor"):
            training_data, testing_data = self.split_data(self.data)
            
            training_data_subset, validation_data = self.split_data(training_data)

            mlflow.log_params({"total_records":self.data.count(),"training_records":training_data_subset.count(),"validation_records":validation_data.count(),"testing_records":testing_data.count()})
            
            encoder = LabelEncoding(self.spark)
            encoded_training_data = encoder.encode_categorical_data({"name":"training_data", "data":training_data_subset})
            encoded_testing_data = encoder.encode_categorical_data({"name":"testing_data", "data":testing_data})
            encoded_validation_data = encoder.encode_categorical_data({"name":"validatation_data", "data":validation_data})

            fitted_encoder = encoder.get_encoder()
            joblib.dump(fitted_encoder, "/tmp/encoder.pkl")
            mlflow.log_artifact("/tmp/encoder.pkl", artifact_path="encoder")

            X_train, y_train, X_validation, y_validation, _X_test, _y_test = self.convert_into_sets(encoded_training_data, encoded_testing_data, encoded_validation_data)

            model = RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1, max_depth=15,min_samples_split=5,min_samples_leaf=2)

            model.fit(X_train,y_train)
            mlflow.log_params({"model_type":"RandomForestRegressor","n_estimators":100,"random_state":42,"n_jobs":-1,"max_depth":15,"min_samples_split":5,"min_samples_leaf":2})

            predictions = model.predict(X_validation)
            mae, mse, r2 = self.validate_model(y_validation, predictions)
            self.logging.info(f"Metrices Score on validatation data: mae={mae}, mse={mse}, r2={r2}")
            mlflow.log_metrics({"validation_mae":mae,"validation_mse":mse,"validation_r2":r2})

            sign = infer_signature(X_validation, predictions)
            mlflow.sklearn.log_model(sk_model=model,name="random_forest_model",skops_trusted_types=["sklearn.tree._tree.Tree"],signature=sign)

            self.save_to_table({"table_path":"supply_chain_daily_demand.ml_model.training_data", "data":training_data})
            self.save_to_table({"table_path":"supply_chain_daily_demand.ml_model.testing_data", "data":testing_data})
            self.save_to_table({"table_path":"supply_chain_daily_demand.ml_model.validation_data", "data":validation_data})

            self.register_model(run_id=mlflow.active_run().info.run_id,validation_r2=r2,validation_mae=mae,validation_mse=mse)

    def register_model(self,run_id,validation_r2,validation_mae,validation_mse):
        client = MlflowClient()

        MODEL_NAME = "supply_chain_daily_demand.ml_model.random_forest_demand"

        # 1. Minimum quality threshold
        MIN_R2 = 0.90

        if validation_r2 < MIN_R2:
            self.logging.info(f"Model rejected. Validation R2={validation_r2:.4f} is below threshold={MIN_R2}")

            mlflow.set_tag("model_status", "rejected")

        self.logging.info(f"Model passed quality gate. Validation R2={validation_r2:.4f}")

        # 2. Register candidate model
        model_uri = (f"runs:/{run_id}/random_forest_model")

        registered_model = mlflow.register_model(model_uri=model_uri,name=MODEL_NAME)

        candidate_version = registered_model.version

        self.logging.info(f"Candidate model registered. Version={candidate_version}")

        # 3. Check current Champion
        try:
            champion = client.get_model_version_by_alias(
                MODEL_NAME,
                "champion"
            )

            champion_version = champion.version
            champion_run_id = champion.run_id

            self.logging.info(f"Current Champion found. Version={champion_version}")
        except mlflow.exceptions.MlflowException:
            champion = None
            self.logging.info("No Champion model exists.")

        if champion is None:

            client.set_registered_model_alias(name=MODEL_NAME,alias="champion",version=candidate_version)

            self.logging.info(f"Candidate version {candidate_version} | promoted to Champion.")

            return {"status": "CHAMPION","version": candidate_version,"run_id": run_id}

        # 5. Get Champion metrics
        champion_run = client.get_run(champion_run_id)

        champion_r2 = champion_run.data.metrics.get("validation_r2")

        champion_mae = champion_run.data.metrics.get("validation_mae")

        self.logging.info(f"Champion metrics: R2={champion_r2:.4f}, MAE={champion_mae:.4f}")

        # 6. Compare Candidate vs Champion
        candidate_is_better = (
            validation_r2 >= MIN_R2
            and validation_mae < champion_mae
        )

        # 7. Promote Candidate
        if candidate_is_better:

            # Old Champion becomes Challenger
            client.set_registered_model_alias(name=MODEL_NAME,alias="challenger",version=champion_version)

            # Candidate becomes Champion
            client.set_registered_model_alias(name=MODEL_NAME,alias="champion",version=candidate_version)

            self.logging.info(f"Candidate version {candidate_version} is better than Champion version {champion_version}.")

            self.logging.info(f"Version {candidate_version} promoted to Champion.")

        # 8. Candidate remains Challenger
        else:

            client.set_registered_model_alias(
                name=MODEL_NAME,
                alias="challenger",
                version=candidate_version
            )

            self.logging.info(f"Candidate version {candidate_version} did not outperform Champion.")