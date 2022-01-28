import nltk
paragraphs="Looking back on a childhood filled with events and memories, I find it rather difficult to pick one that leaves me with the fabled warm and fuzzy feelings As the daughter of an Air Force major, I had the pleasure of traveling across America in many moving trips. I have visited the monstrous trees of the Sequoia National Forest, stood on the edge of the Grand Canyon and have jumped on the beds at Caesar's Palace in Lake Tahoe."
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


ps=PorterStemmer()
wordnet=WordNetLemmatizer()

sentences=nltk.sent_tokenize(paragraphs)
corpus=[]


for i in range(len(sentences)):
    review=re.sub('[^a-zA-Z]',' ',sentences[i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=''.join(review)
    corpus.append(review)
    
    
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(corpus).toarray()
    
    
  



