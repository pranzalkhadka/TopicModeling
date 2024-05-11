from src.data_ingestion import DataIngestion
from src.data_transformation import DataTransformation
from src.data_transformation import clean_text


if __name__ == "__main__":

    ngout = DataIngestion().import_data()
    sdf = DataTransformation(clean_text).data_processing(ngout)
    print(type(sdf))