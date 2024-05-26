import re
from pyspark.sql.functions import split, udf, monotonically_increasing_id, col, length, when, concat, lit
from pyspark.sql.types import StringType


def clean_text(in_string):
    remove_email = re.sub('\S*@\S*\s?', '', in_string)
    remove_nl = re.sub('\s+', ' ', remove_email)
    remove_othr = re.sub("\'|\>|\:|\-", "", remove_nl)
    return remove_othr


class DataTransformation:


    def __init__(self, spark):

        self.spark = spark


    def data_processing(self, ngout):
        
        sdf = self.spark.createDataFrame(ngout)
        sdf = sdf.withColumn("text_sep", split(sdf.text, "\n\n")).select(col('text'), col('target'), col('title'), col('text_sep').getItem(1), col('text_sep').getItem(2)).withColumn("id", monotonically_increasing_id())
        sdf = sdf.select(
            when(col("text_sep[2]").isNull(), col("text_sep[1]"))
            .when(col("text_sep[1]") == " ", col("text_sep[2]"))
            .otherwise(concat(col("text_sep[1]"), lit(" "), col("text_sep[2]"))).alias("text"),
            col("target"),
            col("title"),
            col("id")
        ).filter(
            col("text_sep[2]").isNotNull() & (col("text_sep[1]") != "")
        )

        clean_text_udf = udf(clean_text, StringType())
        sdf = sdf.withColumn("text", clean_text_udf(col("text")))
        sdf = sdf.where(length(col('text')) > 100)
        
        return sdf