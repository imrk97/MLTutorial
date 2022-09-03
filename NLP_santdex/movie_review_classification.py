import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from nltk import word_tokenize
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    def classify(self, features):
        votes=[]
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidance(self, features):
        votes=[]
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes/len(votes)
        return conf

#documents = [(list(movie_reviews.words(fileid)),category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids()]

short_pos = open("positive.txt", "r").read()
short_neg = open("negative.txt", "r").read()

documents = []

for r in short_pos.split("\n"):
    documents.append((r, "pos"))
for r in short_neg.split("\n"):
    documents.append((r, "neg"))
    
random.shuffle(documents)
#print(documents[1])





all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())
for w in short_neg_words:
    all_words.append(w.lower())
    
all_words = nltk.FreqDist(all_words)
print(all_words)
word_features = list(all_words.keys())[:5000]


'''all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
#print(all_words.most_common(15))
#print(all_words["brad"])
word_features = list(all_words.keys())[:3000]
'''
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]
print(len(featuresets))
training_set = featuresets[:1000]
testing_set = featuresets[1000:]

#classifier = nltk.NaiveBayesClassifier.train(training_set)
#classifier_f = open("naivebayes.pickle", "rb")
#classifier = pickle.load(classifier_f)
#classifier_f.close()

#classifier.show_most_informative_features(15)
# classifier = nltk.NaiveBayesClassifier.train(training_set)
# save_classifier = open("pickels/naivebayes.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()


classifier_f = open("pickels/naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

#print("Naive Bayes Algo Accuracy: ", (nltk.classify.accuracy(classifier, testing_set))*100)
#classifier.show_most_informative_features(15)


MNB_classifier = SklearnClassifier(MultinomialNB())
classifier_f1 = open("pickels/multinomialNB.pickle", "rb")
MNB_classifier = pickle.load(classifier_f1)
# classifier_f1.close()
# MNB_classifier.train(training_set)

# MNB_classifier = SklearnClassifier(MultinomialNB())
# MNB_classifier.train(training_set)
# save_classifier = open("pickels/multinomialNB.pickle", "wb")
# pickle.dump(MNB_classifier, save_classifier)
# save_classifier.close()
#print("Multinomial Naive Bayes Algo Accuracy: ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

#GaussianNB_classifier = SklearnClassifier(GaussianNB())
#GaussianNB_classifier.train(training_set)
#print("GaussianNB Algo Accuracy: ", (nltk.classify.accuracy(GaussianNB_classifier, testing_set))*100)


# BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
# BernoulliNB_classifier.train(training_set)
# save_classifier = open("pickels/bernoulliNB.pickle", "wb")
# pickle.dump(BernoulliNB_classifier, save_classifier)
# save_classifier.close()
classifier_f2 = open("pickels/bernoulliNB.pickle", "rb")
BernoulliNB_classifier = pickle.load(classifier_f2)
classifier_f2.close()
#print("BernoulliNB Algo Accuracy: ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

# LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
# LogisticRegression_classifier.train(training_set)
# save_classifier = open("LogisticRegression.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()


# LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
# LogisticRegression_classifier.train(training_set)
# save_classifier = open("pickels/LogisticRegression.pickle", "wb")
# pickle.dump(LogisticRegression_classifier, save_classifier)
# save_classifier.close()
classifier_f7 = open("pickels/LogisticRegression.pickle", "rb")
LogisticRegression_classifier = pickle.load(classifier_f7)
classifier_f7.close()
#print("LogisticRegression Algo Accuracy: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

# SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
# SGDClassifier_classifier.train(training_set)
# save_classifier = open("pickels/SGDClassifier.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()



# SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
# SGDClassifier_classifier.train(training_set)
# save_classifier = open("pickels/SGDClassifier.pickle", "wb")
# pickle.dump(SGDClassifier_classifier, save_classifier)
# save_classifier.close()

classifier_f6 = open("pickels/SGDClassifier.pickle", "rb")
SGDClassifier_classifier = pickle.load(classifier_f6)
classifier_f6.close()
#print("SGDClassifier Algo Accuracy: ", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# save_classifier = open("pickels/SVC.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# save_classifier = open("pickels/SVC.pickle", "wb")
# pickle.dump(SVC_classifier, save_classifier)
# save_classifier.close()

classifier_f3 = open("pickels/SVC.pickle", "rb")
SVC_classifier = pickle.load(classifier_f3)
classifier_f3.close()
#print("SVC Naive Bayes Algo Accuracy: ", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

# LinearSVC_classifier = SklearnClassifier(LinearSVC())
# LinearSVC_classifier.train(training_set)
# save_classifier = open("LinearSVC.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()


# LinearSVC_classifier = SklearnClassifier(LinearSVC())
# LinearSVC_classifier.train(training_set)
# save_classifier = open("pickels/LinearSVC.pickle", "wb")
# pickle.dump(LinearSVC_classifier, save_classifier)
# save_classifier.close()
classifier_f4 = open("pickels/LinearSVC.pickle", "rb")
LinearSVC_classifier = pickle.load(classifier_f4)
classifier_f4.close()
#print("LinearSVC Algo Accuracy: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

# NuSVC_classifier = SklearnClassifier(NuSVC())
# NuSVC_classifier.train(training_set)
# save_classifier = open("NuSVC.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()



# NuSVC_classifier = SklearnClassifier(NuSVC())
# NuSVC_classifier.train(training_set)
# save_classifier = open("pickels/NuSVC.pickle", "wb")
# pickle.dump(NuSVC_classifier, save_classifier)
# save_classifier.close()
classifier_f5 = open("pickels/NuSVC.pickle", "rb")
NuSVC_classifier = pickle.load(classifier_f5)
classifier_f5.close()
#print("NuSVC Algo Accuracy: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

voted_classifier = VoteClassifier(NuSVC_classifier, LinearSVC_classifier, classifier, MNB_classifier, BernoulliNB_classifier, LogisticRegression_classifier, SVC_classifier, SGDClassifier_classifier)
#print("Best Classifier Algo Accuracy: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
#save_classifier = open("pickels/best.pickle", "wb")
#pickle.dump(voted_classifier, save_classifier)
#save_classifier.close()
#print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidance %: ", voted_classifier.confidance(testing_set[0][0])*100)
#print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidance %: ", voted_classifier.confidance(testing_set[1][0])*100)
#print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidance %: ", voted_classifier.confidance(testing_set[2][0])*100)
#print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidance %: ", voted_classifier.confidance(testing_set[3][0])*100)
#print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidance %: ", voted_classifier.confidance(testing_set[4][0])*100)

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidance(feats)


print(sentiment("Ankur is a nice boy."))
#classifier.show_most_informative_features(15)
#save_classifier = open("naivebayes.pickle", "wb")
#pickle.dump(classifier, save_classifier)
#save_classifier.close()
