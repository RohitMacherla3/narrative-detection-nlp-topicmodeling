import pandas as pd
from itertools import chain, product
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def cluster_passages(data, cluster_id):
    indices = [i for i, value in enumerate(chain.from_iterable(data['cluster_ids'])) if value == cluster_id]
    passages = [''.join(passage) for i, passage in enumerate(chain.from_iterable(data['Passages'])) if i in indices]
    return ''.join(passages)

def get_unique_unigrams(corpus):
    stop_words = set(stopwords.words('english'))
    return {word for word in word_tokenize(corpus.lower()) if word not in stop_words}

def cal_jaccard_sim(data1, data2, clus_ids1, clus_ids2, source1, source2):
    result = []

    for idx1, idx2 in product(clus_ids1, clus_ids2):
        topic1_passages = cluster_passages(data1, idx1)
        topic2_passages = cluster_passages(data2, idx2)
        
        topic1_words = get_unique_unigrams(topic1_passages)
        topic2_words = get_unique_unigrams(topic2_passages)
        
        intersection_size = len(topic1_words.intersection(topic2_words))
        union_size = len(topic1_words.union(topic2_words))
        
        if union_size > 0:
            jaccard_score = intersection_size / union_size
            result.append({'Source1': source1, 'Topic1': idx1, 'Source2': source2, 'Topic2': idx2, 'Similarity': jaccard_score})

    similarity_df = pd.DataFrame(result)
    similarity_df = similarity_df.sort_values(by='Similarity', ascending=False)
    similarity_df = similarity_df.loc[(similarity_df['Similarity'] != 0) & (similarity_df['Similarity'] >= 0.2)]
    return similarity_df

if __name__ == "__main__":
    print("Script Started")
    
    email_clusters = pd.read_pickle('Email/email_cluster_bert.pkl')
    podcast_clusters = pd.read_pickle('Podcast/podcast-bert-clust.pkl')
    tv_clusters = pd.read_pickle('TV/tv-bert-clust.pkl')
    
    email_clus_ids = email_clusters['Topic'].unique()[1:][:10]
    podcast_clus_ids = podcast_clusters['Topic'].unique()[1:][:10]
    tv_clus_ids = tv_clusters['Topic'].unique()[1:][:10]
    
    email_data = pd.read_pickle('Email/email_cluster_ids.pkl')
    podcast_data = pd.read_pickle('Podcast/podcast_cluster_ids.pkl')
    tv_data = pd.read_pickle('TV/tv_cluster_ids.pkl')
    
    print("Data imported")
    
    sim_pod_email = cal_jaccard_sim(podcast_data, email_data, podcast_clus_ids, email_clus_ids, 'Podcast', 'Email')
    print("Scores calculated for podcast-email")
    sim_pod_email.to_pickle('Podcast/sim_pod_email.pkl')
    print("Pod-Email file saved")
    
    sim_tv_email = cal_jaccard_sim(tv_data, email_data, tv_clus_ids, email_clus_ids, 'TV', 'Email')
    print("Scores calculated for tv-email")
    sim_tv_email.to_pickle('TV/sim_tv_email.pkl')
    print("TV-Email file saved")
    
    sim_tv_pod = cal_jaccard_sim(tv_data, podcast_data, tv_clus_ids, podcast_clus_ids, 'TV', 'Podcast')
    print("Scores calculated for tv-pod")
    sim_tv_pod.to_pickle('TV/sim_tv_pod.pkl')
    print("TV-Pod file saved")