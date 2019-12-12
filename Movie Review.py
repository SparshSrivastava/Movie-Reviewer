import nltk

# Importing the dataset
from nltk.corpus import movie_reviews

# Looking at various categories of the dataset
print(movie_reviews.categories())
print(movie_reviews.fileids())

# Looking at length of various categories
print(len(movie_reviews.fileids()))
print(len(movie_reviews.fileids('neg')))
print(len(movie_reviews.fileids('pos')))

# Extracting data from various folders
documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((movie_reviews.words(fileid), category))

print(documents[0:5])

# As the document consists of one category then other so need to shuffle it
import random
random.shuffle(documents)
print(documents[:5])

# Importing the stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')

# Adding punctuations to the list of stopwords
import string
punctuations = list(string.punctuation)
stop += punctuations

# Defining a function to  clean the document
from nltk.corpus import wordnet

def get_simple_pos(pos):
    if pos.startswith('J'):
        return wordnet.ADJ
    elif pos.startswith('V'):
        return wordnet.VERB
    elif pos.startswith('N'):
        return wordnet.NOUN
    elif pos.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

#print(pos_tag(['Apple']))

def clean_document(words):
    output_words = []
    for w in words:
        if w.lower() not in stop:
            pos = pos_tag([w])
            pos = get_simple_pos(pos[0][1])
            clean_word = lemmatizer.lemmatize(w, pos)
            output_words.append(clean_word.lower())
    return output_words

documents = [ (clean_document(w),category) for w, category in documents]

print(documents[0])

# Spllting the document into training and testing documents
training_document = documents[:1500]
testing_document = documents[1500:]

# Finding the frequency of all words present in the data
all_words = []

for doc in training_document:
    all_words += doc[0]
    
freq = nltk.FreqDist(all_words)

# Calculating the features out of these frequencies
features = freq.most_common(3000)
features = [f for f,cnt in features]

# Defining a function to get which features appears in the list
def get_feature_dict(words):
    current_dict = {}
    word_set = set(words)
    for w in features:
        current_dict[w] = w in word_set
    return current_dict

# Now getting our data into (dict of features, category) form
training_document = [ (get_feature_dict(w), category) for w, category in training_document]
testing_document = [ (get_feature_dict(w), category) for w, category in testing_document]

# Now applying the Naive Bayes Classifier onto it
from nltk.classify import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(training_document)

# Predicting the performance of the model
print(nltk.classify.accuracy(classifier, testing_document)) 

# Printing the most informative 15 most informative words 
print(classifier.most_informative_features(15))

# Using Count Vectorizer to convert this into X, Y Matrices
from sklearn.feature_extraction.text import CountVectorizer
cnt_vec = CountVectorizer(max_features = 2000, ngram_range=(1,3))

# Making the array Y
categories = [ category for doc, category in documents]

# Making the text_document consisting of text rather than words
text_document = [ " ".join(doc) for doc, category in documents]
print(text_document[0])

# Splitting the dataset into training and testing
from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(text_document, categories)

X_train = cnt_vec.fit_transform(X_train)
X_test = cnt_vec.transform(X_test)

# Applying SVC classifier 
from sklearn.svm import SVC
svc = SVC()

svc.fit(X_train, Y_train)

# Applying grid search to reach to best values of SVC
from sklearn.model_selection import GridSearchCV
grid = {"C": [1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
        "gamma": [1e-3, 5e-4, 1e-4, 5e-3]}
grid_search_classifier = GridSearchCV(svc, grid)

# Fitting onto the data
grid_search_classifier.fit(X_train, Y_train)
Y_pred_svm_grid = grid_search_classifier.predict(X_test)
Y_pred_svm = svc.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(Y_test, Y_pred_svm_grid))
print(confusion_matrix(Y_test, Y_pred_svm_grid))
print(grid_search_classifier.score(X_test, Y_test)) # Grid Search gives a better accuracy so far

print(classification_report(Y_test, Y_pred_svm))
print(confusion_matrix(Y_test, Y_pred_svm))
print(svc.score(X_test, Y_test))

# KNN classfier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

#Applying grid search
grid_knn = {'n_neighbors':[i for i in range(1,26,2)]}
knn_grid_search = GridSearchCV(knn, grid_knn)
knn_grid_search.fit(X_train, Y_train)

print(knn_grid_search.best_estimator_)

Y_pred_knn = knn_grid_search.predict(X_test)

print(classification_report(Y_test, Y_pred_knn))
print(confusion_matrix(Y_test, Y_pred_knn))
print(knn_grid_search.score(X_test, Y_test))

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier()

forest.fit(X_train, Y_train)
Y_pred_forest = forest.predict(X_test)

print(classification_report(Y_test, Y_pred_forest))
print(confusion_matrix(Y_test, Y_pred_forest))
print(forest.score(X_test, Y_test))

# Thus we can conclude that SVC classifier gives us the best results on this dataset giving us an accuracy of around 0.808