import csv
from top2vec import Top2Vec

corpus = []
with open("predict_result.tsv", encoding="utf-8") as file:
    tsv_file = csv.reader(file, delimiter="\t")

    for line in tsv_file:
        corpus.append(line[0])

model = Top2Vec(corpus, min_count=1, embedding_model='universal-sentence-encoder-multilingual')

print(model.get_num_topics())

print(model.get_topics(0))
print(model.get_topics(1))
print(model.get_topics(2))