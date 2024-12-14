import pandas as pd
import re
from sklearn.decomposition import LatentDirichletAllocation
from statsmodels.tsa.arima.model import ARIMA
from gensim.models import HdpModel
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfTransformer
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
import plotly.figure_factory as ff
import numpy as np
import matplotlib.pyplot as plt



datadf = pd.read_excel(r'datav2.xlsx')

df = datadf.copy()
## Data Preprocessing
## Combining title and abstract into a single text column

df['text'] = df['title'] + " " + df['abstract']
df['text'] = df['text'].astype(str)

## Preprocessing function
## Function to preprocess the text: lowercase, remove stopwords and punctuations

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join([word.lower() for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS])
    return text

df['cleaned_text'] = df['text'].apply(preprocess_text)


## Feature Extraction
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

doc_term_matrix = vectorizer.fit_transform(df['cleaned_text'])
feature_names = vectorizer.get_feature_names_out()

## Applying tf-idf

tfidf_transformer = TfidfTransformer()
tfidf_matrix = tfidf_transformer.fit_transform(doc_term_matrix)

### LDA Modelling

n_topics = 300
lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda_output = lda_model.fit_transform(tfidf_matrix)


## Hierarchial Strucure Modelling
def calculate_topic_distances(lda_model, feature_names):
    ## topic-term matrix
    topic_term_matrix = lda_model.components_


    topic_distances = pdist(topic_term_matrix, metric='cosine')
    return topic_distances


def create_topic_hierarchy(lda_model, feature_names, n_topics=300):
    topic_term_matrix = lda_model.components_

    ## Calculating distances between topics
    topic_distances = calculate_topic_distances(lda_model, feature_names)

    ## Hierarchial Clustering
    linkage_matrix = sch.linkage(topic_distances, method='ward')

    ## Creating Visualization: Dendogram
    plt.figure(figsize=(15, 10))
    dendrogram = sch.dendrogram(
        linkage_matrix,
        labels=[f'Topic {i}' for i in range(n_topics)],
        leaf_rotation=90,
        leaf_font_size=8
    )
    plt.title('Hierarchical Structure of Topics')
    plt.xlabel('Topics')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()

    return linkage_matrix

## Function to get the top topics in the reseach papers
def print_hierarchical_topics(lda_model, feature_names, linkage_matrix, n_clusters=10):
    clusters = sch.fcluster(linkage_matrix, n_clusters, criterion='maxclust')

    topic_hierarchy = {}
    for topic_idx, cluster_id in enumerate(clusters):
        if cluster_id not in topic_hierarchy:
            topic_hierarchy[cluster_id] = []
        topic_hierarchy[cluster_id].append(topic_idx)

    print("Hierarchical Topic Structure:")
    for cluster_id, topics in topic_hierarchy.items():
        print(f"\nCluster {cluster_id}:")
        for topic_idx in topics:
            top_words = [feature_names[i] for i in lda_model.components_[topic_idx].argsort()[:-10:-1]]
            print(f"  Topic {topic_idx}: {', '.join(top_words)}")


linkage_matrix = create_topic_hierarchy(lda_model, feature_names)
print_hierarchical_topics(lda_model, feature_names, linkage_matrix)

## Evaluating HDP by calculating coherence scores for the clusters
def calculate_cluster_coherence(topic_hierarchy, lda_model, feature_names):
    coherence_scores = {}
    for cluster_id, topics in topic_hierarchy.items():
        # Get topic-term matrices for topics in this cluster
        cluster_matrices = lda_model.components_[topics]
        # Calculate average cosine similarity within cluster
        cluster_distances = pdist(cluster_matrices, metric='cosine')
        coherence_scores[cluster_id] = 1 - np.mean(cluster_distances)
    return coherence_scores


clusters = sch.fcluster(linkage_matrix, 10, criterion='maxclust')
topic_hierarchy = {}
for topic_idx, cluster_id in enumerate(clusters):
    if cluster_id not in topic_hierarchy:
        topic_hierarchy[cluster_id] = []
    topic_hierarchy[cluster_id].append(topic_idx)

coherence_scores = calculate_cluster_coherence(topic_hierarchy, lda_model, feature_names)
print("\nCluster Coherence Scores:")
for cluster_id, score in coherence_scores.items():
    print(f"Cluster {cluster_id}: {score:.3f}")

## Fetching top 10 topics
topic_prevalence = lda_output.sum(axis=0) / lda_output.sum()
top_10_topics = topic_prevalence.argsort()[-10:][::-1]

df['dominant_topic'] = lda_output.argmax(axis=1)

## Time series forecasting for the top 10 topics
print("\nTop 10 Topics and Their Forecasts:")
for topic_idx in top_10_topics:
    top_words = [feature_names[i] for i in lda_model.components_[topic_idx].argsort()[:-5:-1]]
    print(f"\nTopic {topic_idx}: {', '.join(top_words)}")

    yearly_data = df[df['dominant_topic'] == topic_idx].groupby('year').size().reset_index(name='count')
    yearly_data['proportion'] = yearly_data['count'] / yearly_data['count'].sum()

    try:
        ### Applying arima model
        model = ARIMA(yearly_data['proportion'], order=(1 ,1 ,1))
        model_fit = model.fit()

        ## Forecasting for 3 years (2018-2020)
        forecast = model_fit.forecast(steps=3)

        plt.figure(figsize=(10, 6))
        plt.plot(yearly_data['year'], yearly_data['proportion'], label='Historical', marker='o')
        forecast_years = range(yearly_data['year'].max() + 1, yearly_data['year'].max() + 4)
        plt.plot(forecast_years, forecast, label='Forecast', marker='x', linestyle='--', color='red')

        plt.title(f'Topic {topic_idx} Time Series Forecast\n({", ".join(top_words)})')
        plt.xlabel('Year')
        plt.ylabel('Topic Proportion')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print("Forecasted proportions:")
        for year, value in zip(forecast_years, forecast):
            print(f"Year {year}: {value:.4f}")

    except Exception as e:
        print(e)
