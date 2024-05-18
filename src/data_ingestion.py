import pandas as pd
from sklearn.datasets import fetch_20newsgroups


class DataIngestion:

    def import_data(self):

        newsgroup_train = fetch_20newsgroups(subset="train")
        newsgroup_test = fetch_20newsgroups(subset="test")

        df = pd.DataFrame([newsgroup_train.data, newsgroup_train.target.tolist()]).T
        df.columns = ['text', 'target']
        targets = pd.DataFrame(newsgroup_train.target_names)
        targets.columns = ['title']
        ngout = pd.merge(df, targets, left_on='target', right_index=True)

        return ngout

if __name__ == "__main__":
    
    d = DataIngestion().import_data()
    print(len(d))