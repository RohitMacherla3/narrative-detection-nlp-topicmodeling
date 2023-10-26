# narrative-detection-nlp-topicmodeling
This project aims to detect narrative spread across diverse platforms such as Emails, Podcasts, and TV. The techniques used for this project are LLM for obtaining the semantic meaning of the text data and DP-Means Clustering to identify different narratives.

## This week's Action Items:
1. Obtain Clusters for all data sources using BERTopic
- Experiment with hyperparameters to figure out the right one.
- Get around 150-200 clusters for each data source.

2. Similarities
- Get Jaccard Similarities for the clusters to compare the data sources.
- Compare the clusters with >0.4 similarity score and do the date tracking

3. Link Transformer Similarities
- Run the code to get the combined data frame to compare and verify the cosine similarities. (Use GPU)

4. Keyword Analysis
- Top 10 users using certain keywords
- Top 10 TV Shows using certain keywords
- Context words analysis for Email data
- General statistics, time series plots for all the data sources
- POS tagging to identify the subjects, objects and causes for all the data sources

5. Get Top unigrams and bigrams for all data sources
- Generate unigrams and bigrams and compare the top of them across all data sources to identify unique unigrams/bigrams to certain data sources.


## Action Items: Oct 20 - Oct 26:

1. Amarel Issues
- Get the Job running issues resolved - Cleared
- Get the required packages installed - Cleared

2. TV Embedding
- Obtain TV Embeddings - on Hold
- Note: As we are not using DBSCAN or K-Means, putting it on hold

3. Keyword Analysis
  Refer to the paper to perform keyword analysis across the data sources (for example - “HATE”, and “RESENT”)
  Paper - https://cdn.theconversation.com/static_files/files/1255/Hate_on_Fox_News_draft_report_9-28-20.pdf

- Top 10 Channels using certain keywords - Done
- Top 10 Podcasts using certain keywords - Done
- Context words analysis for TV data - Done
- Context words analysis for Podcast data - Done
  

## Action Items: Oct 13 - Oct 19:

1. Obtain Clusters for all data sources using BERTopic
- Clusters obtained for Emails and Podcast data. Pending on Amarel for running it for TV data.

2. Keyword Analysis
- Implemented code for extracting unigrams and bigrams for TV data channel wise.
- Extracted top antipathy words using sentiment analysis LLM and plotted the comparisons for 3 TV channels
- Performed context-word analysis to identify before and after words for a particular keyword. 


## Action Items: Oct 3 - Oct 12:
1. Get the Amarel Cluster issue resolved - carry on to next week
2. Run the MPNet LLM and get embeddings at passages level - In progress (TV Remaining) - carry on to next week
3. Try FAISS on K-Means - Completed (Moved on to BERTopic)
4. Try Link Transformer - carry on to next week
5. Get the next Steps from the paper - carry on to next week
6. Try BERTopic for Clustering - carry on to next week
7. Run the DP-Means Clustering and obtain results - not moving forward with this approach.
