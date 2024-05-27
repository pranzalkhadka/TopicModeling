from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml.clustering import LDA


class ModelTrainer:


    def __init__(self, spark):

        self.spark = spark
        self.idf = IDF(inputCol="rawFeatures", outputCol="features")
        self.lda = LDA(k=20, maxIter=100, optimizer="em")


    def model_training(self, news_df):

        cv = CountVectorizer(inputCol="word", outputCol="rawFeatures", vocabSize=10000, minDF=5)
        cvmodel = cv.fit(news_df)
        featurizedData = cvmodel.transform(news_df)
        idfModel = self.idf.fit(featurizedData)
        rescaledData = idfModel.transform(featurizedData)
        corpus = rescaledData.select('id', 'features')
        model = self.lda.fit(corpus)
        ll = model.logLikelihood(corpus)
        lp = model.logPerplexity(corpus)

        return ll, lp