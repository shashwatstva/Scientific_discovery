import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

datadf = pd.read_excel(r'datav2.xlsx')
df = datadf.copy()
## Data Preprocessing
## Combining title and abstract into a single text column
df['text'] = df['title'] + " " + df['abstract']
df['text'] = df['text'].astype(str)

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

## Applying tf-idf
tfidf_transformer = TfidfTransformer()
tfidf_matrix = tfidf_transformer.fit_transform(doc_term_matrix)

### LDA Modelling
n_topics = 300

lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda_output = lda_model.fit_transform(tfidf_matrix)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        topic_list = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in topic_list]
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")


feature_names = vectorizer.get_feature_names_out()
print_top_words(lda_model, feature_names, 10)

## Visualization
topic_distribution = lda_output.sum(axis=0) / lda_output.sum()
plt.figure(figsize=(10, 5))
sns.barplot(x=range(1, n_topics + 1), y=topic_distribution)
plt.title('Topic Distribution')
plt.xlabel('Topic')
plt.ylabel('Distribution')
plt.show()

## Viz2: Topic distribution
df['dominant_topic'] = lda_output.argmax(axis=1)
topic_by_year = df.groupby('year')['dominant_topic'].value_counts(normalize=True).unstack()

## Viz 2 : Topic evaluation
plt.figure(figsize=(12, 6))
topic_by_year.plot(kind='area', stacked=True)
plt.title('Topic Evolution Over Time')
plt.xlabel('Year')
plt.ylabel('Topic Proportion')
plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


def calculate_coherence(model, feature_names, doc_term_matrix, top_n=10):
    term_rankings = []
    for topic_idx, topic in enumerate(model.components_):
        topic_list = topic.argsort()[:-top_n - 1:-1]
        term_rankings.append([feature_names[i] for i in topic_list])

    coherence = 0
    for topic in term_rankings:
        word_vectors = doc_term_matrix[:, [feature_names.tolist().index(word) for word in topic]]
        word_vectors = word_vectors.tocsc()
        sims = cosine_similarity(word_vectors.T)
        coherence += np.mean(sims[np.triu_indices(len(topic), k=1)])

    return coherence / len(term_rankings)


coherence_score = calculate_coherence(lda_model, feature_names, doc_term_matrix)
print("Topic Coherence: ",coherence_score)



