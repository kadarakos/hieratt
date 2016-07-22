import numpy as np
from model import hieratt_network
from keras.utils.np_utils import to_categorical


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
mref_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print "Training"
mref_model.fit([X_train, queries_train], y_train, batch_size=128, nb_epoch=100,
          verbose=1, validation_data=([X_test, queries_test], y_test))
