import pandas as pd
import numpy as np
import linktransformer as lt
import os
from itertools import chain

print("packages imported")

email_df = pd.read_pickle('Email/emails_data_processed.pkl')
podcast_df = pd.read_pickle('Podcast/podcast_data_processed.pkl')

print("data loaded")

flt_pas_email = pd.DataFrame(columns= ['Passages'])
flt_pas_email['Passages'] = list(chain.from_iterable(email_df['Passages']))

flt_pas_podcast = pd.DataFrame(columns= ['Passages'])
flt_pas_podcast['Passages'] = list(chain.from_iterable(podcast_df['Passages']))

print("data flattened")

lt_pod_em = lt.merge(flt_pas_podcast, flt_pas_email, merge_type='1:m', on="Passages", model="all-MiniLM-L6-v2", 
                         left_on=None, right_on=None)

print("podcast-email completed")
                         
lt_pod_em['Pod_Pass'] = lt_pod_em['Passages_x']
lt_pod_em['Pod_Ids'] = lt_pod_em['id_lt_x']
lt_pod_em['Em_Pass'] = lt_pod_em['Passages_y']
lt_pod_em['Em_Ids'] = lt_pod_em['id_lt_y']
lt_pod_em['Scores'] = lt_pod_em['score']

lt_pod_em = lt_pod_em.drop(columns = ['Passages_x', 'Passages_y', 'id_lt_x', 'id_lt_y', 'score'])
lt_pod_em.to_pickle('Podcast/podcast-email-passage-sim.pkl')

print("dataframe for pod-email created")