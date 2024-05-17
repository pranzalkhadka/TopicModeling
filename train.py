from src.data_ingestion import DataIngestion
from src.data_transformation import DataTransformation
from src.data_transformation import clean_text
from src.feature_engineering import FeatureEngineering
from src.model_trainer import ModelTrainer
from pyspark.sql import SparkSession



# if __name__ == "__main__":

#     ngout = DataIngestion().import_data()

#     sdf = DataTransformation(clean_text).data_processing(ngout)

#     news_df, first_row = FeatureEngineering().feature_engineering(sdf)

#     ll, lp = ModelTrainer().model_training(news_df)

#     # print(type(news_df))
#     #print(news_df.show(5))
#     # print(first_row)
#     print(ll, lp)

if __name__ == "__main__":
    # Initialize Spark session
    spark = SparkSession.builder.appName("TopicModeling").getOrCreate()

    # Data Ingestion
    ngout = DataIngestion().import_data()

    # Data Transformation
    data_transformation = DataTransformation(spark, clean_text)
    sdf = data_transformation.data_processing(ngout)

    # Feature Engineering
    feature_engineering = FeatureEngineering()
    news_df, first_row = feature_engineering.feature_engineering(sdf)

    # Model Training
    model_trainer = ModelTrainer(spark)
    ll, lp = model_trainer.model_training(news_df)

    print(ll, lp)

    # Stop Spark session
    spark.stop()
