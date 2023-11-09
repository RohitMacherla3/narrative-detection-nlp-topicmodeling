import pandas as pd
import numpy as np
import linktransformer as lt
import os
from itertools import chain

print("packages imported")

email_df = pd.read_pickle('Email/emails_data_processed.pkl')
tv_df = pd.read_pickle('TV/tv_data_processed.pkl')

print("data loaded")

flt_pas_email = pd.DataFrame(columns= ['Passages'])
flt_pas_email['Passages'] = list(chain.from_iterable(email_df['Passages']))

flt_pas_tv = pd.DataFrame(columns= ['Passages'])
flt_pas_tv['Passages'] = list(chain.from_iterable(tv_df['Passages']))

print("data flattened")

lt_tv_em = lt.merge(flt_pas_tv, flt_pas_email, merge_type='1:m', on="Passages", model="all-MiniLM-L6-v2", 
                         left_on=None, right_on=None)

print("tv-em completed")

lt_tv_em['Tv_Pass'] = lt_tv_em['Passages_x']
lt_tv_em['Tv_Ids'] = lt_tv_em['id_lt_x']
lt_tv_em['Em_Pass'] = lt_tv_em['Passages_y']
lt_tv_em['Em_Ids'] = lt_tv_em['id_lt_y']
lt_tv_em['Scores'] = lt_tv_em['score']

lt_tv_em = lt_tv_em.drop(columns = ['Passages_x', 'Passages_y', 'id_lt_x', 'id_lt_y', 'score'])
lt_tv_em.to_pickle('TV/tv-email-passage-sim.pkl')

print("dataframe for tv-email created")