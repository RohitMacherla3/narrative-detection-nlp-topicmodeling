print("Initialized")

import pandas as pd
from itertools import chain, islice
from collections import Counter
from nltk import word_tokenize, pos_tag
import spacy
import pickle

print("Libraries imported")

nlp = spacy.load("en_core_web_sm")

tv_data = pd.read_pickle('TV/tv_data_processed.pkl')
flattened_passages = list(chain.from_iterable(tv_data['Passages']))

print("Data loaded")

def tag_subject_object_frequencies(text, target_word):
    print("Code executing")
    words = word_tokenize(text)
    tagged_words = pos_tag(words)

    subjects = []
    objects = []

    for i, (word, pos) in enumerate(tagged_words):
        if word.lower() == target_word.lower():
            subject = next((tagged_words[j][0] for j in range(i - 1, -1, -1) if 'NN' in tagged_words[j][1]), '')
            object_ = next((tagged_words[j][0] for j in range(i + 1, len(tagged_words)) if 'NN' in tagged_words[j][1]), '')

            subjects.append(subject)
            objects.append(object_)

    subject_freq = Counter(subjects)
    object_freq = Counter(objects)

    return subject_freq, object_freq

corpus = ''.join(flattened_passages)
print("Corpus created")

subject_freq, object_freq = tag_subject_object_frequencies(corpus, "hate")

top_subjects = dict(sorted(subject_freq.items(), key=lambda item: item[1], reverse=True)[:20])
top_objects = dict(sorted(object_freq.items(), key=lambda item: item[1], reverse=True)[:20])

print("Top subjects:", top_subjects)
print("Top objects:", top_objects)

subject_objects = {'subjects': top_subjects, 'objects': top_objects}

file_path = 'TV/TV_sub_obj_freqs.pkl'

with open(file_path, "wb") as file:
    pickle.dump(subject_objects, file)

print("File created")