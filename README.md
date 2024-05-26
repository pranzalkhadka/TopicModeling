# TopicModeling
This project is aimed to apply topic modeling which is an unsupervised Machine Learning technique to identify groups of similar words within a body of text. 

# Datset
The data set used is 20 newsgroups collection dataset from kaggle which is a popular datatset for text classification and text clustering. This data set is a collection of 20,000 messages, collected from 20 different newsgroups. One thousand messages from each of the twenty newsgroups were chosen at random and partitioned by newsgroup. Some of the categories of newsgroup are comp.graphics, sci.electronics, talk.politics.guns, sci.space and more.


# Techniques used

1. TF-IDF :- TF-IDF is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents. It helps to identify the most significant words in each document by balancing the frequency of words against their overall presence in the corpus. TF-IDF helps to highlight the most relevant words in each document, which is then be used as features for topic modeling.

2. LDA :- LDA is a probabilistic model used to discover hidden topics in a collection of documents. LDA uses the TF-IDF weighted terms to identify patterns and group words into topics and assigns probabilities to words within each topic.

# How to run?
1. Clone the repo, create and activate a virtual environment.
2. Install the dependencies from the requirements file.
3. Run the train.py file from the project directory.