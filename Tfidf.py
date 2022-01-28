import nltk
paragraph="NLP is a subfield of computer science and artificial intelligence concerned with interactions between computers and human (natural) languages. It is used to apply machine learning algorithms to text and speech. we can use NLP to create systems like speech recognition, document summarization, machine translation, spam detection, named entity recognition, question answering, autocomplete, predictive typing and so on.Nowadays, most of us have smartphones that have speech recognition. These smartphones use NLP to understand what is said. Also, many people use laptops which operating system has a built-in speech recognition."

import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


wordnet=WordNetLemmatizer()
sentences=nltk.sent_tokenize(paragraph)
corpus=[]



for i in range(len(sentences)):
    review=re.sub('[^a-zA-Z]',' ',sentences[i])
    review=review.lower()
    review=review.split()
    
    review=[wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    
    review=' '.join(review)
    corpus.append(review)
    
    
    
from sklearn.feature_extraction.text import TfidfVectorizer
cv=TfidfVectorizer()
X=cv.fit_transform(corpus).toarray()