import pandas as pd
from itertools import chain, product
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle

def cal_jaccard_sim(data1, data2, source1, source2):
    result = []
    clus_ids1 = list(data1.keys())
    clus_ids2 = list(data2.keys())

    for idx1, idx2 in product(clus_ids1, clus_ids2):
        topic1_words = data1[idx1]
        topic2_words = data2[idx2]
        
        intersection_size = len(topic1_words.intersection(topic2_words))
        union_size = len(topic1_words.union(topic2_words))
        
        if union_size > 0:
            jaccard_score = intersection_size / union_size
            result.append({'Source1': source1, 'Topic1': idx1, 'Source2': source2, 'Topic2': idx2, 'Similarity': jaccard_score})

    similarity_df = pd.DataFrame(result)
    similarity_df = similarity_df.sort_values(by='Similarity', ascending=False)
    similarity_df = similarity_df.loc[(similarity_df['Similarity'] != 0) & (similarity_df['Similarity'] > 0.18)]
    return similarity_df

if __name__ == "__main__":
    print("Script Started")
    
    with open('Email/email_cluster_unigrams.pkl', 'rb') as pickle_file:
        email_data = pickle.load(pickle_file)
        
    with open('Podcast/podcast_cluster_unigrams.pkl', 'rb') as pickle_file:
        podcast_data = pickle.load(pickle_file)
        
    with open('TV/tv_cluster_unigrams.pkl', 'rb') as pickle_file:
        tv_data = pickle.load(pickle_file)
    
    sim_pod_email = cal_jaccard_sim(podcast_data, email_data, 'Podcast', 'Email')
    print("Scores calculated for podcast-email")
    
    sim_tv_email = cal_jaccard_sim(tv_data, email_data, 'TV', 'Email')
    print("Scores calculated for tv-email")
    
    sim_tv_pod = cal_jaccard_sim(tv_data, podcast_data, 'TV', 'Podcast')
    print("Scores calculated for tv-pod")
    
    jaccard_sims = pd.concat([sim_pod_email, sim_tv_email,sim_tv_pod], axis=0, ignore_index=True)
    
    jaccard_sims.to_pickle('TV/jaccard_similarities.pkl')
    print("Similarities file saved")