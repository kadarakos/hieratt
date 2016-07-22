from itertools import cycle, izip
from model import hieratt_network
import numpy as np
from PIL import Image
import pandas as pd
from nltk import word_tokenize
from collections import Counter
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam

def compose(sents):
    """ I'M HACKY
        Take a single sentence, map words
        to embeddings and take average of the embeddings """
    batch = []
    for sent in sents:
        sent = word_tokenize(sent.lower()) # use NLTK word tokenizer to tokenize sentence
        mat = np.zeros((len(sent), 300)) # initialize empty sentence representation
        for i,word in enumerate(sent):
            try:
                mat[i] = embeddings[word] # fill each row with the word embedding
            except:
                # ignore unknown words
                pass
                #print word, '|',
        mat = np.sum(mat, axis=0) * 0
        batch.append(mat)
    return np.array(batch)


def batch_data(data_set, batch_size=100):
    """Return (images, question vectors, answers) batches."""
    image_batch = np.zeros((batch_size,  3, target_size[0], target_size[1]))
    questions_batch = []
    answer_batch = []
    c = -1
    for j in data_set:
        if j[2] in common_answers:
            c+=1
            img = Image.open(vqa_path+j[1]).resize(target_size)
            img = np.array(img)[:, :, :3]
            img = img.astype('float32')
            img /= 255.
            image_batch[c] = img.transpose((2,0,1))
            questions_batch.append(j[0])
            answer_batch.append(ans2id[j[2]])

            if c == batch_size-1:
                questions_batch = compose(questions_batch)
                answer_batch = to_categorical(answer_batch,
                                              nb_classes=n_answers)             
                return image_batch, questions_batch, answer_batch



print "Compiling"
target_size = (200, 200)
n_answers = 1000
adam = Adam(lr=0.001)
model = hieratt_network(target_size[0], 300, 100, n_answers)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])


vqa_path = "/roaming/akadar/VQA/"
train_questions = open(vqa_path+"data/preprocessed/questions_id_train2015.txt").read().split('\n')[:1000]
train_images = open(vqa_path+"data/preprocessed/images_train2015.txt").read().split('\n')[:1000]
train_answers = open(vqa_path+"data/preprocessed/answers_train2015_modal.txt").read().split('\n')[:1000]
val_questions = open(vqa_path+"data/preprocessed/questions_val2015.txt").read().split('\n')[:1000]
val_images = open(vqa_path+"data/preprocessed/images_val2015_all.txt").read().split('\n')[:1000]
val_answers = open(vqa_path+"data/preprocessed/answers_val2015_modal.txt").read().split('\n')[:1000]
train_set = cycle(izip(train_questions, train_images, train_answers))
val_set = cycle(izip(val_questions, val_images, val_answers))

embed = "omar"
vectors = "paragram-phrase-XXL.txt" if embed == "paraphrase" else "deps.words"
print vectors
embeddings = pd.read_csv('/roaming/akadar/word_embeddings/'+vectors, 
                         sep=' ', header=None, index_col=None).T
embeddings.columns = embeddings.iloc[0]                  # first row as column names (for fast indexing) 
embeddings = embeddings.reindex(embeddings.index.drop(0))# removing original first row
print embeddings.shape


common_answers = set([x[0] for x in Counter(train_answers).most_common(n_answers)])
ans2id = {v: k for k, v in enumerate(common_answers)}
id2ans = {v: k for k, v in ans2id.iteritems()}

batchsize = 100
n_samples = len(train_images)
epochs = 50


print "Number of samples: ", n_samples

for i in range(epochs):
    seen = 0
    while seen < n_samples:
        X, q, y = batch_data(val_set, batch_size=batchsize)
	a, b =  model.train_on_batch([X, q], y)
        print "Epoch: ", i, "Loss: ", a, "Acc: ", b
        seen += batchsize
