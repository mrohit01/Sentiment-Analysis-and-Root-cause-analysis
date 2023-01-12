# importing libraries
from langdetect import detect#supports around 55 language
from googletrans import Translator
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.util import ngrams
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import contractions
import yake
from rake_nltk import Rake
from unidecode import unidecode
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#reading the dataset
data = pd.read_csv('dataset_es_train.csv')
data.head(2)

#checking the shape of dataset
data.shape

#selecting 100 rows
sample = data.sample(100)
sample.reset_index(drop=True, inplace=True)

#making a dataframe of review and stars
df = sample[['review_body','stars']]

#checking the languages of reviews
def lang_detect(data):
    lang = detect(data)
    return lang

#returns the language code
print(df.review_body[0])
lang_detect(df.review_body[0])
df['language'] = df.review_body.apply(lang_detect)

#returns the unique languages of reviews
df.language.unique()
df.language.value_counts()

#translating the review to english
def lang_trans(data):
    translor = Translator()
    translated_text = translor.translate(data)
    return translated_text.text
print(df.review_body[0])
lang_trans(df.review_body[0])
df['translated_reviews'] = df.review_body.apply(lang_trans)

# performing the exploratory data analysis
def ngram_extractor(data,ngram_range):
    tokens = word_tokenize(data)
    ngram = ngrams(tokens,ngram_range)
    ngram_list1 = []
    for ngram1 in ngram:
        ngram_list1.append(' '.join(ngram1))
    return ngram_list1
list_unigrams = df.translated_reviews.apply(lambda x : ngram_extractor(x,1))

final_unigram = []
for unigram in list_unigrams:
    final_unigram.extend(unigram)
    
cnt = Counter(final_unigram).most_common(25)#frequent unigrams

# preprocessing the data
def expand_text(data):
    expanded_text = contractions.fix(data)
    return expanded_text
stopword_list = stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('nor')
stopword_list.remove('not')#to maintain negative review as negative reviews 
def clean_data(data):
    tokens = word_tokenize(data)
    clean_text = [word.lower() for word in tokens if (word not in punctuation) and (word.lower() not in stopword_list) and (len(word)>2) and (word.isalpha())]
    return clean_text
clean_text = df.translated_reviews.apply(expand_text)
clean_text = clean_text.apply(clean_data)

#getting the list of unigrams
list_unigrams = clean_text.apply(lambda x : ngram_extractor(' '.join(x),1))

final_unigram = []
for unigram in list_unigrams:
    final_unigram.extend(unigram)
    
cnt = Counter(final_unigram).most_common(25)

#getting the list of trigrams
list_trigrams = clean_text.apply(lambda x : ngram_extractor(' '.join(x),3))

final_trigram = []
for trigram in list_trigrams:
    final_trigram.extend(trigram)
    
cnt = Counter(final_trigram).most_common(25)

test = df[(df.stars<3)].reset_index(drop=True)

clean_text = test.translated_reviews.apply(expand_text)
clean_text = clean_text.apply(clean_data)
list_trigrams = clean_text.apply(lambda x : ngram_extractor(' '.join(x),3))

final_trigram = []
for trigram in list_trigrams:
    final_trigram.extend(trigram)
    
cnt = Counter(final_trigram).most_common(25)

#creating the wordcloud
def wordcloud(data,column):
    df_ = data[column].str.cat(sep = ' ')
    text = ' '.join([word for word in df_.split()])
    wordcloud = WordCloud(width = 700, height = 500, background_color='white').generate(text)
    plt.figure(figsize=(10,16))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
wordcloud(df,'translated_reviews')

#extracting the keywords using yake
def yake_extractor(data):
    key_extractor = yake.KeywordExtractor()
    keywords = key_extractor.extract_keywords(data)
    keyword_list = []
    for kw in keywords:
        keyword_list.append(kw[0])
    return keyword_list
keywords = df.translated_reviews.apply(yake_extractor)
all_keywords = []
for kw in keywords:
    all_keywords.extend(kw)
cnt = Counter(all_keywords).most_common(1000)

#extracting the keywords using rakw
def rake_extractor(data):
    keyword_extractor = Rake()
    keyword_extractor.extract_keywords_from_text(data)
    return keyword_extractor.get_ranked_phrases()
rake_keywords = df.translated_reviews.apply(rake_extractor)
all_keywords = []
for kw in rake_keywords:
    all_keywords.extend(kw)
cnt = Counter(all_keywords).most_common(100)

# preprocessing the data
# 1. remove spaces,newlines
def remove_spaces(data):
    clean_text = data.replace('\\n',' ').replace("\t",' ').replace('\\',' ')
    return clean_text

# 2. contraction mapping
def expand_text(data):
    expanded_text = contractions.fix(data)
    return expanded_text

# 3.handling accented character
def handling_accented(data):
    fixed_text = unidecode(data)
    return fixed_text

# 4. Cleaning 
stopword_list = stopwords.words("english")
stopword_list.remove('no')
stopword_list.remove('nor')
stopword_list.remove('not')

def clean_data(data):
    tokens = word_tokenize(data)
    clean_text = [word.lower() for word in tokens if (word not in punctuation) and(word.lower() not in stopword_list) and(len(word)>2) and (word.isalpha())]
    return clean_text                   # and(word.lower() not in stopword_list) and(len(word)>2) and (word.isalpha())]

# 5.autocorrect 
def autocorrection(data):
    spell = Speller(lang='en')
    corrected_text = spell(data)
    return corrected_text

# 6. lemmatization
def lemmatization(data):
    lemmatizer = WordNetLemmatizer()
    final_data = []
    for word in data :
        lemmatized_word = lemmatizer.lemmatize(word)
        final_data.append(lemmatized_word)
    return " ".join(final_data)

    clean_text_train = df.translated_reviews.apply(remove_spaces)

clean_text_train = clean_text_train.apply(expand_text)

clean_text_train = clean_text_train.apply(handling_accented)

clean_text_train = clean_text_train.apply(clean_data)

clean_text_train = clean_text_train.apply(lemmatization)

#creating the text vectorization using count vectoriser
count_vect = CountVectorizer()
bow = count_vect.fit_transform(clean_text_train).A
pd.DataFrame(bow, columns=count_vect.get_feature_names())

#creating the text vectorization using tfidf
tfidf_vect = TfidfVectorizer()
tfidf = tfidf_vect.fit_transform(clean_text_train).A
pd.DataFrame(tfidf, columns=tfidf_vect.get_feature_names())

sent = clean_text_train.tolist()
splitted_sent = [sen.split() for sen in sent]
print(splitted_sent)
#creating the text vectorization using word2vec
word_2vec_model = Word2Vec(splitted_sent,min_count=2,window=3)
word_2vec_model.save('word2vec.model')
def vectorizer(list_of_docs,model):
    feature = []#to save vector of reviews
    for rew in list_of_docs:
        zero_vector = np.zeros(model.vector_size)#if word2vec didn't make vector of any word
        vectors = []#to save vector of words
        for word in rew:
            if word in model.wv:
                try:
                    vectors.append(model.wv[word])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            feature.append(avg_vec)
        else:
            feature.append(zero_vector)
    return feature
#creating the doc2vec
vectorized_docs = vectorizer(splitted_sent,word_2vec_model)
x_emb = np.array(vectorized_docs)
x_emb

#creating the target column using k-means
def build_kmeans(clusters,data):
    kmeans_model = KMeans(n_clusters=clusters)
    y_pred = kmeans_model.fit_predict(data)
    return kmeans_model,y_pred

# kmeans -count-vectorizer
kmeans_model_count,count_pred = build_kmeans(3,bow)

# kmeans-tfidf
kmeans_model_tfidf,tfidf_pred = build_kmeans(3,tfidf)

# kmeans-word2vec
kmeans_model_word2vec,word2vec_pred= build_kmeans(3,x_emb)

# evaluation of clusters
print(f"Silhouette score with kmeans-count : {silhouette_score(bow,count_pred)}")
print(f"Silhouette score with kmeans-tfidf : {silhouette_score(tfidf,tfidf_pred)}")
print(f"Silhouette score with kmeans-word2vec : {silhouette_score(x_emb,word2vec_pred)}")

def visulize_silhouette(data,model,title1):
    visualizer = SilhouetteVisualizer(model,colors='yellowbrick')
    visualizer.fit(data)
    plt.title(f"Silhouette visualizer for {title1}")
# kmeans-count
visulize_silhouette(bow,kmeans_model_count,'Kmeans_count_vectorizer')

# kmeans-tfidf
visulize_silhouette(tfidf,kmeans_model_tfidf,'Kmeans_tfidf_vectorizer')

# kmeans-word2vec
visulize_silhouette(x_emb,kmeans_model_word2vec,'Kmeans_Word2vec_vectorizer')

target = pd.Series(count_pred)

df["Target"] = target
final_df = df[["translated_reviews","Target"]]
final_df
#splitting the dataset into training and testing 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(final_df.translated_reviews, final_df.Target, test_size=0.25,random_state=42)

# count_vectorizer
count = CountVectorizer(max_df=0.95)
count_val_train = count.fit_transform(x_train)
count_val_test = count.transform(x_test)

count_val_train # .A or  toarray() have to convert sparse matrix into array
#sparse matrix is a matrix in which many or most of the elements have zero value

pd.DataFrame(count_val_train.A,columns = count.get_feature_names())
#checking the model accuracy with the count vectoriser
count_mnb = MultinomialNB()
count_mnb.fit(count_val_train.A,y_train)
predict_count = count_mnb.predict(count_val_test.A)
accuracy_count = accuracy_score(y_test,predict_count)*100
accuracy_count

# tfidf 
tfidf = TfidfVectorizer(max_df=0.95)
tfidf_train = tfidf.fit_transform(x_train)
tfidf_test = tfidf.transform(x_test)
tfidf_train.toarray()
tfidf_train.A
pd.DataFrame(tfidf_train.A, columns = tfidf.get_feature_names())
#checking the model accuracy with the tfidf
tfidf_mnb = MultinomialNB()
tfidf_mnb.fit(tfidf_train.A,y_train)
predict_tfidf = tfidf_mnb.predict(tfidf_test.A)
accuracy_tfidf = accuracy_score(y_test,predict_tfidf)*100
accuracy_tfidf