from keras.models import Sequential, Model
import keras.backend as K
from keras.utils.np_utils import to_categorical
from keras.layers import Input, merge
from keras.layers.core import Flatten, Dense, Dropout, Reshape
from keras.layers.core import TimeDistributedDense, RepeatVector, Permute, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.optimizers import SGD


def repeat_1(x):
    """Wrap keras backend repeat."""
    return K.repeat_elements(x, 32, 2)

def sum_(x):
    """Wrap keras backend sum."""
    return K.sum(x, axis=1)

def hieratt_network(w, query_in_size, query_embed_size, nb_classes):
    """Try to reproduce Hierarchical attention networks.

    https://arxiv.org/abs/1606.02393
    """
    input_image = Input(shape=(3,w,w))  # w by w color image
    input_question = Input(shape=(query_in_size,))     # question vector
    w = w/2

    # Feature map 1
    f_1 = Convolution2D(32, 3, 3, activation='relu',
                        border_mode='same')(input_image)
    f_1 = Convolution2D(32, 3, 3,  activation='relu',
                        border_mode='same')(input_image)
    f_1 = MaxPooling2D((2,2), strides=(2,2))(f_1)
    # f_1 = Dropout(0.25)(f_1)
    num_features = f_1.shape[2] * 2

    # Attention 1
    # Create num_feature by num_channels "feature columns".
    f_1 = Reshape((32, w*w))(f_1)
    f_1 = Permute((2,1))(f_1)
    q_1   = Dense(query_embed_size, activation='relu')(input_question)  # Encode question
    # Add question embedding to each feature column
    q_1   = RepeatVector(w*w)(q_1)
    q_f   = merge([f_1, q_1], 'concat')
    # Estimate and apply attention per feature
    att_1 = TimeDistributedDense(1, activation="sigmoid")(q_f)
    att_1 = Lambda(repeat_1, output_shape=(w*w, 32))(att_1)
    att_1 = merge([f_1, att_1], 'mul')
    # Reshape to the original feature map from previous layer
    att_1 = Permute((2,1))(att_1)
    f_1_att = Reshape((32, w, w))(att_1)


    # Feature map 2
    f_2 = Convolution2D(32, 3, 3, activation='relu',
                        border_mode='same')(f_1_att)
    f_2 = Convolution2D(32, 3, 3,  activation='relu',
                        border_mode='same')(f_2)
    f_2 = MaxPooling2D((2,2), strides=(2,2))(f_2)
    # f_2 = Dropout(0.25)(f_2)

    w = w/2
    # Attention 2
    f_2 = Reshape((32, w*w))(f_2)
    f_2 = Permute((2,1))(f_2)
    q_2   = Dense(query_embed_size, activation='relu')(input_question)
    q_2   = RepeatVector(w*w)(q_2)
    q_f   = merge([f_2, q_2], 'concat')
    att_2 = TimeDistributedDense(1, activation="sigmoid")(q_f)
    att_2 = Lambda(repeat_1, output_shape=(w*w, 32))(att_2)
    att_2 = merge([f_2, att_2], 'mul')
    att_2 = Permute((2,1))(att_2)
    f_2_att = Reshape((32, w, w))(att_2)

    # Feature map 3
    f_3 = Convolution2D(32, 3, 3, activation='relu',
                        border_mode='same')(f_2_att)
    f_3 = Convolution2D(32, 3, 3,  activation='relu',
                        border_mode='same')(f_3)
    f_3 = MaxPooling2D((2,2), strides=(2,2))(f_3)
    # f_3 = Dropout(0.25)(f_3)

    w = w/2
    # Attention 3
    f_3 = Reshape((32, w*w))(f_3)
    f_3 = Permute((2,1))(f_3)
    q_3   = Dense(query_embed_size, activation='relu')(input_question)
    q_3   = RepeatVector(w*w)(q_3)
    q_f   = merge([f_3, q_3], 'concat')
    att_3 = TimeDistributedDense(1, activation="sigmoid")(q_f)
    att_3 = Lambda(repeat_1, output_shape=(w*w, 32))(att_3)
    att_3 = merge([f_3, att_3], 'mul')
    att_3 = Permute((2,1))(att_3)
    f_3_att = Reshape((32, w, w))(att_3)

     # Feature map 4
    f_4 = Convolution2D(32, 3, 3, activation='relu',
                        border_mode='same')(f_3_att)
    f_4 = Convolution2D(32, 3, 3,  activation='relu',
                        border_mode='same')(f_4)
    f_4 = MaxPooling2D((2,2), strides=(2,2))(f_4)
    # f_4 = Dropout(0.25)(f_4)

    w = w/2
    # Attention 4
    f_4 = Reshape((32, w*w))(f_4)
    f_4 = Permute((2,1))(f_4)
    q_4   = Dense(query_embed_size, activation="relu")(input_question)
    q_4   = RepeatVector(w*w)(q_4)
    q_f   = merge([f_4, q_4], 'concat')
    att_4 = TimeDistributedDense(1, activation="softmax")(q_f)
    att_4 = Lambda(repeat_1, output_shape=(w*w, 32))(att_4)
    att_4 = merge([f_4, att_4], 'mul')
    f_att = Lambda(sum_, output_shape=(32,))(att_4)

    probs = Dense(nb_classes, activation="softmax")(f_att)
    model = Model(input=[input_image, input_question], output=probs)
    return model


