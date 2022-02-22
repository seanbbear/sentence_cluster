from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import csv
import json

def get_cluster(corpus, speaker):
    embedder = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    # embedder = SentenceTransformer('all-MiniLM-L6-v2')

    corpus_embeddings = embedder.encode(corpus)

    # Perform kmean clustering
    num_clusters = 10
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(corpus[sentence_id])

    for i, cluster in enumerate(clustered_sentences):
        print("Cluster ", i+1)
        print(cluster)
        print("")

    with open("kmeans/"  + "sum_" + str(speaker) + ".json", "w", encoding="utf-8") as f:
        json.dump(clustered_sentences, f, ensure_ascii=False)
    

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