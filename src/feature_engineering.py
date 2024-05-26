from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

porter = PorterStemmer()
lemma = WordNetLemmatizer()


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
        spremoved = spremoved.withColumn("stemmed_tokens", self.word_tokenize_udf("spfiltered"))
        news_df = spremoved.select(
            spremoved['target'],
            spremoved['title'],
            spremoved['id'],
            spremoved['stemmed_tokens'].alias('word')
        )

        return news_df