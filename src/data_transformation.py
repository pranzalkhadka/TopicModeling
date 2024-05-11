import re
from pyspark.sql.functions import split, udf
from pyspark.sql.functions import split, monotonically_increasing_id, col, length
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType


def clean_text(in_string):
    # Remove email
    remove_email = re.sub('\S*@\S*\s?', '', in_string)
    # Remove extra whitespaces
    remove_nl = re.sub('\s+', ' ', remove_email)
    # Remove miscellaneous symbols
    remove_othr = re.sub("\'|\>|\:|\-", "", remove_nl)
    return remove_othr


class DataTransformation:

    def __init__(self, clean_func):
        self.spark = SparkSession.builder.appName("DataTransformation").getOrCreate()
        self.clean_udf = udf(clean_func, StringType())


    def data_processing(self, ngout):
        sdf = self.spark.createDataFrame(ngout)
        sdf = sdf.withColumn("text_sep", split(sdf.text, "\n\n")).select(col('text'), col('target'), col('title'), col('text_sep').getItem(1), col('text_sep').getItem(2)).withColumn("id", monotonically_increasing_id())
        sdf = sdf.withColumn("cleaned_text", self.clean_udf(col("text")))
        sdf = sdf.where(length(col('text')) > 100)
        return sdf