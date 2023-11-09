import pandas as pd
import numpy as np
import linktransformer as lt
import os
from itertools import chain

print("packages imported")

podcast_df = pd.read_pickle('Podcast/podcast_data_processed.pkl')
tv_df = pd.read_pickle('TV/tv_data_processed.pkl')

print("data loaded")

flt_pas_podcast = pd.DataFrame(columns= ['Passages'])
flt_pas_podcast['Passages'] = list(chain.from_iterable(podcast_df['Passages']))

flt_pas_tv = pd.DataFrame(columns= ['Passages'])
flt_pas_tv['Passages'] = list(chain.from_iterable(tv_df['Passages']))

print("data flattened")

lt_tv_pod = lt.merge(flt_pas_tv, flt_pas_podcast, merge_type='1:m', on="Passages", model="all-MiniLM-L6-v2", 
                         left_on=None, right_on=None)
print("tv-pod completed")

lt_tv_pod['Tv_Pass'] = lt_tv_pod['Passages_x']
lt_tv_pod['Tv_Ids'] = lt_tv_pod['id_lt_x']
lt_tv_pod['Pod_Pass'] = lt_tv_pod['Passages_y']
lt_tv_pod['Pod_Ids'] = lt_tv_pod['id_lt_y']
lt_tv_pod['Scores'] = lt_tv_pod['score']

lt_tv_pod = lt_tv_pod.drop(columns = ['Passages_x', 'Passages_y', 'id_lt_x', 'id_lt_y', 'score'])
lt_tv_pod.to_pickle('TV/tv-podcast-passage-sim.pkl')

print("dataframe for tv-pod created")