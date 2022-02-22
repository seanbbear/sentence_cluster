from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import csv
import json

def get_cluster(corpus, speaker):
    # embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embedder = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

    corpus_embeddings = embedder.encode(corpus)

    # Normalize the embeddings to unit length
    corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    # Perform kmean clustering
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.2) #, affinity='cosine', linkage='average', distance_threshold=0.4)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(corpus[sentence_id])

    result = []
    for i, cluster in clustered_sentences.items():
        print("Cluster ", i+1)
        print(cluster)
        print("")
        result.append(cluster)

    with open("agglo/"  + "sum_" + str(speaker)  + "_1.2" + ".json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    
if __name__=="__main__":
    # corpus = []
    # with open("predict_result.tsv") as file:
    #     tsv_file = csv.reader(file, delimiter="\t")

    #     for line in tsv_file:
    #         corpus.append(line[0])
    with open("evergreen_speakers_with_summary.json") as file:
        data = json.load(file)
        speaker_list = (data["_source"]["voice_data"])
        for speaker in speaker_list:
            corpus = []
            for i in range(len(speaker)):
                speaker_name = speaker[i]["speaker"]
                corpus.append(speaker[i]["summary"])
            if len(corpus)>5:
                get_cluster(corpus, speaker_name)