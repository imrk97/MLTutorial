from nltk.corpus import wordnet


syns = wordnet.synsets("laugh")
#synset
print(syns[0].name())

#just the word
print(syns[0].lemmas()[0].name())

print(syns[0].definition())
print(syns[0].examples())

synonyms = list()
antonyms = list()

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
#print(set(synonyms), set(antonyms))

w1 = wordnet.synset("man.n.01")
w2 = wordnet.synset("motorcycle.n.01")

print(w1.wup_similarity(w2))
