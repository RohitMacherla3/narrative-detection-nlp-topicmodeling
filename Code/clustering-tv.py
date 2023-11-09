import pandas as pd
import numpy as np
from collections import Counter
import time

from bertopic import BERTopic
from itertools import chain
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from umap import UMAP
from hdbscan import HDBSCAN
from memory_profiler import memory_usage

tv_df = pd.read_pickle('TV/tv_data_processed.pkl')

flattened_passages = list(chain.from_iterable(tv_df['Passages']))

print(len(flattened_passages))


import random
samples = random.sample(flattened_passages, len(flattened_passages))

print("data sampled")


umap_model = UMAP(n_neighbors=8, min_dist=0.3)
hdbscan_model = HDBSCAN(min_cluster_size=600, gen_min_span_tree=True, prediction_data=True)

topic_model = BERTopic(n_gram_range=(1, 3), top_n_words=20,
                      umap_model=umap_model, hdbscan_model=hdbscan_model, verbose=True)

print("running the clustering")
topics, probs = topic_model.fit_transform(samples)

mem_usage = memory_usage(-1, interval=1)
print(f"Memory usage: {mem_usage[0]} MB")

print("clustering completed")

print("# of topics: ", topic_model.get_topic_info().shape)

tv_passage_splits = [len(embedding) for embedding in tv_df['Passages']]
passage_mapping ={}
j = 0
for i in range(len(tv_passage_splits)):
    passages_count = tv_passage_splits[i]
    passage_mapping[i] = topics[j:j+passages_count]
    j += passages_count

tv_df['cluster_ids'] = passage_mapping.values()
tv_df.to_pickle('TV/tv_cluster_ids.pkl')

topic_model.get_topic_info().to_pickle('TV/tv-bert-clust.pkl')
topic_model.visualize_hierarchy().write_html('TV/tv-bert-clust.html')