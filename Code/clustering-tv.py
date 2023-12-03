import pandas as pd
from collections import Counter
from bertopic import BERTopic
from itertools import chain
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from umap import UMAP
from hdbscan import HDBSCAN

def load_data(file_path):
    return pd.read_pickle(file_path)

def cluster_passages(passages):
    umap_model = UMAP(n_neighbors=6, min_dist=0.25)
    hdbscan_model = HDBSCAN(min_cluster_size=500, gen_min_span_tree=True, prediction_data=True)
    
    topic_model = BERTopic(n_gram_range=(1, 3), top_n_words=20,
                          umap_model=umap_model, hdbscan_model=hdbscan_model, verbose=True)

    print("Running the clustering")
    topics, _ = topic_model.fit_transform(passages)
    print("Clustering completed")
    
    return topic_model, topics

if __name__ == "__main__":
    
    tv_df = load_data('TV/tv_data_processed.pkl')
    flattened_passages = list(chain.from_iterable(tv_df['Passages']))
    topic_model, topics = cluster_passages(flattened_passages)
    tv_passage_splits = [len(embedding) for embedding in tv_df['Passages']]
    
    passage_mapping = {}
    j = 0
    for i in range(len(tv_passage_splits)):
        passages_count = tv_passage_splits[i]
        passage_mapping[i] = topics[j:j+passages_count]
        j += passages_count
    
    tv_df['cluster_ids'] = passage_mapping.values()
    
    tv_df.to_pickle('TV/tv_cluster_ids.pkl')
    print("ids df saved")
    
    topic_model.get_topic_info().to_pickle('TV/tv-bert-clust.pkl')
    print("clusters df saved")