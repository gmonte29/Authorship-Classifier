import pandas as pd
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import requests
from bs4 import BeautifulSoup

#Dataset available at https://archive.ics.uci.edu/ml/machine-learning-databases/00454/
#pull training data from url and save to folder as 'authorship_data.csv'
trainData = pd.read_csv('authorship_data.csv',sep = ',',  encoding ='latin1')

#create tf-idf vector using training data
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(trainData['text'])

#Vector based on frequency, not needed for TF-IDF vector
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

#Vector based on TF-IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#train multinomial naive bayes classifier with training data
clf = MultinomialNB().fit(X_train_tfidf, trainData['author'])



'''
Find url for book you want to test and enter here
'''
bookText = requests.get('https://www.fadedpage.com/books/20210122/html.php').text
soup = BeautifulSoup(bookText, 'html.parser')

# Extract the text content of the book from the parsed HTML
bookContent = soup.get_text()

# Create TI-IDF matrix with bookContent and enter into classifier
X_new_counts = count_vect.transform([bookContent])
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)

# retrieve author, label mapping file
label_mapping = pd.read_excel(
    "Author_Label_Mapping.xlsx", header=None, names=["key", "value"]
)
dictionary = label_mapping.set_index("key")["value"].to_dict()

#prints the suggested Victorian Author
print(dictionary[predicted[0]])

