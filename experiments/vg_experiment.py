from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer

print "Reading data"
path = "../attributes_data/"
imgids = open(path+"imgids.txt").read().split('\n')
atts = open(path+"attributes.txt").read().split('\n')
objs = open(path+"objects.txt").read().split('\n')
objs = map(lambda x: x.replace(' ', '').replace('-', ''), objs)

print "Number of images:", len(set(imgids))

print "Computing features"
attribute_vectorizer = CountVectorizer(binary=True)
object_tokenizer = Tokenizer()

object_tokenizer.fit_on_texts(objs)
O = object_tokenizer.texts_to_sequences(objs)
A = attribute_vectorizer.fit_transform(atts)

print "Number of object classes:", len(object_tokenizer.word_counts)
print "Number of attribute classes:", len(attribute_vectorizer.vocabulary_)
