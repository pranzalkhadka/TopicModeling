from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType
# from pyspark.sql.functions import col

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

porter = PorterStemmer()
lemma = WordNetLemmatizer()

spwordlist = ["article", "write", "entry", "date", "udel", "said", "tell", "think", "know", "just", "isnt", "line", "like", "does", "going", "make", "thanks", "also"]

class FeatureEngineering:

    def __init__(self):
        self.word_tokenize_udf = udf(self.word_tokenize, ArrayType(StringType()))


    def word_tokenize(self, text):
        pos = nltk.pos_tag(text)
        final = [lemma.lemmatize(word[0]) if (lemma.lemmatize(word[0]).endswith(('e','ion')) or len(word[0]) < 4 ) else porter.stem(word[0]) for word in pos]
        return final


    def feature_engineering(self, sdf):
        tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern="\W+", minTokenLength=4, toLowercase=True)
        tokenized = tokenizer.transform(sdf)
        spremover = StopWordsRemover(inputCol="tokens", outputCol="spfiltered")
        spremoved = spremover.transform(tokenized)
        # stemmed = spremoved.rdd.map(lambda tup: (tup[1], tup[2], tup[3], self.word_tokenize(tup[5])))
        # stemmed.collect()
        # news_df = stemmed.toDF(schema=['target', 'title', 'id', 'word'])
        #spremoved = spremoved.withColumn("stemmed_tokens", self.word_tokenize_udf("spfiltered"))
        spremoved = spremoved.withColumn("stemmed_tokens", self.word_tokenize_udf("spfiltered"))
        news_df = spremoved.select(
            spremoved['target'],
            spremoved['title'],
            spremoved['id'],
            spremoved['stemmed_tokens'].alias('word')
        )
        filtered_df = news_df.filter(col('title') == 'rec.autos')
        first_row = filtered_df.select('word').first()

        return news_df, first_row
