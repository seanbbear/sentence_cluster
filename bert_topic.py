from numpy import vectorize
from bertopic import BERTopic
import csv
from sklearn.feature_extraction.text import CountVectorizer
import jieba

def tokenize_zh(text):
    words = jieba.lcut(text)
    return words

vectorizer = CountVectorizer(tokenizer=tokenize_zh)

corpus = []
with open("predict_result.tsv", encoding="utf-8") as file:
    tsv_file = csv.reader(file, delimiter="\t")

    for line in tsv_file:
        corpus.append(line[0])

topic_model = BERTopic(language="multilingual", calculate_probabilities=True, verbose=True, vectorizer_model=vectorizer)
topics, probs = topic_model.fit_transform(corpus)

freq = topic_model.get_topic_info(); freq.head(5)
print(freq)

print(topic_model.get_topic(0))
print(topic_model.get_topic(1))
# print(topic_model.visualize_topics())