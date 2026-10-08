class Gold:
    def __init__(self, catalog_name, silver_schema_name, silver_table_name,gold_schema_name, aggregated_gold_table_name, featured_gold_table_name,spark, logging):
        self.catalog_name = catalog_name
        self.silver_schema_name = silver_schema_name
        self.silver_table_name = silver_table_name
        self.gold_schema_name = gold_schema_name
        self.aggregated_gold_table_name = aggregated_gold_table_name
        self.featured_gold_table_name = featured_gold_table_name
        self.spark = spark
        self.logging = logging

    def load_silver_table(self):
        silver_table_path = f"{self.catalog_name}.{self.silver_schema_name}.{self.silver_table_name}"

        self.logging.info(f"Loading the delta table from {silver_table_path}.")
        self.data = self.spark.read.table(silver_table_path)
        self.logging.info("Silver table loaded to dataframe.")

    def write_aggregated_table(self):
        silver_table_path = f"{self.catalog_name}.{self.silver_schema_name}.{self.silver_table_name}"
        aggregated_gold_table_path = f"{self.catalog_name}.{self.gold_schema_name}.{self.aggregated_gold_table_name}"

        self.logging.info(f"Writing the aggregated table to {aggregated_gold_table_path}")
        self.spark.sql(f"""
            CREATE OR REPLACE TABLE {aggregated_gold_table_path} AS
            SELECT 
                transaction_date,
                product_name,
                destination_city,
                ROUND(SUM(demand_quantity), 2) AS total_demand,
                ROUND(AVG(unit_price_usd), 2) AS avg_unit_price,
                SUM(available_inventory) as total_inventory
            FROM {silver_table_path}
            GROUP BY transaction_date, product_name, destination_city       
        """)
        self.logging.info("Aggregated table saved in the gold schema.")

    def write_feature_table(self):
        aggregated_gold_table_path = f"{self.catalog_name}.{self.gold_schema_name}.{self.aggregated_gold_table_name}"
        features_gold_table_path = f"{self.catalog_name}.{self.gold_schema_name}.{self.featured_gold_table_name}"

        self.logging.info(f"Writing the feature table to {features_gold_table_path}")
        self.spark.sql(f"""
            CREATE OR REPLACE TABLE {features_gold_table_path} AS 
            SELECT 
                transaction_date,
                product_name,
                destination_city,
                total_demand,
                avg_unit_price,
                total_inventory,
                dayofweek(transaction_date) AS day_of_week,
                month(transaction_date) AS month,
                ROUND(LAG(total_demand) OVER (PARTITION BY product_name, destination_city ORDER BY transaction_date), 2) AS demand_lag_1,
                ROUND(LAG(total_demand, 7) OVER (PARTITION BY product_name, destination_city ORDER BY transaction_date), 2) AS demand_lag_7,
                ROUND(AVG(total_demand) OVER (PARTITION BY product_name, destination_city ORDER BY transaction_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) AS moving_avg_7
            FROM {aggregated_gold_table_path}    
        """)
        self.logging.info("Feature table saved in the gold schema.")
