import numpy as np
from model import hieratt_network
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import pickle
import sys
sys.path.append( "../deep-learning-models/")
from vgg19_hiearatt import VGG19_hieratt
from keras.layers import Dense
from keras.models import Model

MODEL = 'vgg'

if MODEL == 'vgg':
    base_model = VGG19_hieratt(include_top=True, 
                               query_in_size=10, 
                               query_embed_size=20)
    x = base_model.output
    predictions = Dense(5, activation='softmax', name='aclassifier')(x)
    mref_model = Model(input=base_model.input, output=predictions)

    for layer in mref_model.layers:
        if  not layer.name[0] == 'a':
            layer.trainable = False
    data = np.load('mref_vgg.npz')


else:
    mref_model = hieratt_network(100, 10, 20, 5)
    data = np.load('mref.npz')

X_train = data['train_data']
q_train = data['train_queries']
y_train = data['train_targets']
X_test = data['test_data']
q_test = data['test_queries']
y_test = data['test_targets']

# shift channel dimension of images
X_train = np.transpose(X_train, [0,3,1,2])
X_test = np.transpose(X_test, [0,3,1,2])
color_mapper = dict(zip(set(y_train), range(0, len(set(y_train)))))
y_train =  map(lambda x: color_mapper[x], y_train)
y_test = map(lambda x: color_mapper[x], y_test)
y_train, y_test = to_categorical(y_train), to_categorical(y_test)
queries_train = to_categorical(data['train_queries'])
queries_test = to_categorical(data['test_queries'])
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


print "Compiling"
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

mref_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print "Training"
hist = mref_model.fit([X_train, queries_train], y_train, batch_size=12, nb_epoch=50,
          verbose=1, validation_data=([X_test, queries_test], y_test), callbacks=[early_stopping])

pickle.dump(hist.history, open("history.pkl", 'w'))
print "Saving training history, architecture and weights matrices"
json_string = mref_model.to_json()

open('mref_model_architecture.json', 'w').write(json_string)
mref_model.save_weights('mref_model_weights.h5')

# NOTE TO MY SELF
# model = model_from_json(open('mref_model_architecture.json').read())
# model.load_weights('mref_model_weights.h5')
