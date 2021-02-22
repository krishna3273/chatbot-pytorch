import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer


stemmer=PorterStemmer()


def tokenise(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

#Using one hot encoding
def bag_of_words(tokenised_sentence,all_words):
    tokenised_sentence=[stem(w) for w in tokenised_sentence]
    # print(tokenised_sentence)
    v=[]
    for word in all_words:
        if word in tokenised_sentence:
            v.append(1)
        else:
            v.append(0)
    return np.array(v,dtype=np.float32)



# test= "Is anyone there?"
# print(test)
# tokenised_test=tokenise(test)
# print(tokenised_test)

