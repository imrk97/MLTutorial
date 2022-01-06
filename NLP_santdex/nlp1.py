
# tokenization: dividing a text into tokens of sentance and words.
from nltk import word_tokenize, sent_tokenize
example_text = "The quick brown fox, jumps over the lazy dog. What are you gonna do? Sue me?"
sentances = sent_tokenize(example_text)
words = word_tokenize(example_text)


# stop words removal: removing words that are useless to NLP nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
'''
filtered_sentance = list()
for w in words:
    if w not in stop_words:
        filtered_sentance.append(w)
'''
filtered_sentance = [w for w in words if w not in stop_words]


#stemming:- getting the root of a word. e.g. searching->search.
from nltk.stem import PorterStemmer
ps = PorterStemmer()
stemmed_words = list(map(lambda w: ps.stem(w), filtered_sentance))