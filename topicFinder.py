"""
Topic modelling with LDA and NMF
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

import sys

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

path = sys.argv[1:]
data = fetch_20newsgroups(data_home=path[0],remove=('headers', 'footers', 'quotes'))

# Printing the 20 topic labels
print("\nTopic Labels: \n{}".format(data.target_names))

# Printing a newsgroup article
print("\nSample news article: \n{}".format(data.data[0]))

# Construct the text feature matrix for both LDA and NMF
# Using TfidfVectorizer for NMF and CountVectorizer for LDA
tfidfVec = TfidfVectorizer(max_df=0.95, min_df=5, max_features=1000, stop_words='english')
countVec = CountVectorizer(max_df=0.95, min_df=5, max_features=1000, stop_words='english')

# This is the TF-IDF weighted matrix of [samples X features]
tfidfMatrix = tfidfVec.fit_transform(data.data)
tfidfFeatureNames = tfidfVec.get_feature_names()

# This is the raw count matrix of [samples X features]
countMatrix = countVec.fit_transform(data.data)
countFeatureNames = countVec.get_feature_names()

print("\nTF-IDF features: \n{}".format(tfidfFeatureNames))

print("\nCount based features: \n{}".format(countFeatureNames))

print("\nTop words in each topic per LDA model:")
lda = LatentDirichletAllocation(n_components=10, learning_method='online', random_state=0).fit(countMatrix)
print_top_words(lda,countFeatureNames, 10)

print("\nTop words in each topic per NMF model:")
nmf = NMF(n_components=10, random_state=1, alpha=0.1, l1_ratio=0.5).fit(tfidfMatrix)
print_top_words(nmf,tfidfFeatureNames, 10)