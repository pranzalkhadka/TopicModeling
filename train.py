from src.data_ingestion import DataIngestion
from src.data_transformation import DataTransformation
from src.feature_engineering import FeatureEngineering
from src.model_trainer import ModelTrainer
from pyspark.sql import SparkSession


if __name__ == "__main__":

    spark = SparkSession.builder.appName("TopicModeling").getOrCreate()
    ngout = DataIngestion().import_data()
    data_transformation = DataTransformation(spark)
    sdf = data_transformation.data_processing(ngout)
    feature_engineering = FeatureEngineering()
    news_df = feature_engineering.feature_engineering(sdf)
    model_trainer = ModelTrainer(spark)
    ll, lp = model_trainer.model_training(news_df)
    print(ll)
    print(lp)


