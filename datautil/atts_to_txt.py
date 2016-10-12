"""This module prepares the Visual Genome data set for attribute prediction."""

import json
import os
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
import numpy as np
import pickle
from PIL import Image
import h5py
from collections import defaultdict


ATT_path = "/roaming/public_datasets/VisualGenome/attributes/"
IMG_path = "/roaming/public_datasets/VisualGenome/images/"
DATA = "attributes_v2.json"
img_path = [IMG_path+"VG_100K/", IMG_path+"VG_100K_2/"]
N = 500                    # Minimum frequency of query and target labels
target_size = (224, 224)   # Size of images

if DATA == "attributes_v2.json":
    print "v2"
    names = 'synsets'
    imgname = 'image_id'
else:
    print "v1"
    names = "object_names"
    imgname = "id"


def data_filter(sample):
    """Filter attributes and objects that are not in the most common."""
    new_sample = sample.copy()
    atts = filter(lambda x: x in common_att, sample['attributes'])
    objs = filter(lambda x: x in common_obj, sample[names])
    if atts and objs:
        new_sample['attributes'] = atts
        new_sample[names] = objs
        return new_sample


def load_image(id, target_size):
    """
    Load image from path and return HxWx3 numpy array.

    Skip gray-scale.
    """
    id = str(id)
    try:
        path = img_path[0] + str(id) + '.jpg'
        img = Image.open(path)
        if img.mode == "L":
            return
        img = img.resize((target_size[1], target_size[0]))
        img = np.asarray(img.getdata()).reshape((target_size[0],
                                                 target_size[1],
                                                 3))
    except:
        path = img_path[1] + str(id) + '.jpg'
        img = Image.open(path)
        if img.mode == "L":
            return
        img = img.resize((target_size[1], target_size[0]))
        img = np.asarray(img.getdata()).reshape((target_size[0],
                                                 target_size[1],
                                                 3))
    return np.array(img).astype("uint8")


def list_duplicates(seq):
    """Given the imgids list, return the ranges of the same elements."""
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items())


print "Loading data set"
data = json.load(open(ATT_path + DATA))

print "Computing attribute and object frequencies"
# Count the occurrences of attributes and objects in the data set
att_freq = {}
obj_freq = {}
for i in data:
    for j in i['attributes']:
        if 'attributes' in j:
            for att in j['attributes']:
                if att in att_freq:
                    att_freq[att] += 1
                else:
                    att_freq[att] = 1
            for obj in j[names]:
                if obj in obj_freq:
                    obj_freq[obj] += 1
                else:
                    obj_freq[obj] = 1

# Top most frequent classes
# num_att = 300
# num_obj = 500
# common_att = dict(sorted(att_freq.iteritems(), key=lambda x: x[1], reverse=True)[:num_att])
# common_obj = dict(sorted(obj_freq.iteritems(), key=lambda x: x[1], reverse=True)[:num_obj])

# Classes that appear at least N times
common_att = dict(filter(lambda x: x[1] >= N, att_freq.iteritems()))
common_obj = dict(filter(lambda x: x[1] >= N, obj_freq.iteritems()))

data_path = "/home/akadar/hieratt/attributes_data/"
if not os.path.exists(data_path):
    os.makedirs(data_path)

print "Writing samples to imigits.txt, objects.txt, attributes.txt"
imgid_file = open(data_path+"imgids.txt", "wb")
obj_file = open(data_path+"objects.txt", "wb")
att_file = open(data_path+"attributes.txt", "wb")

print len(data)
for i in range(0, len(data)):
    print i, '\r',
    sample = data[i]['attributes']
    imgid = data[i][imgname]
    for j in sample:
        # If object annotation has attributes
        if 'attributes' in j:
            filtered = data_filter(j)
            # If both object and attribute classes remain after filtering
            if filtered:
                att, obj = filtered['attributes'], filtered[names]
                # If there is only one object annotated
                # (not shit like "cat" and "ear)
                if len(obj) == 1:
                            imgid_file.write(str(imgid) + "\n")
                            obj_file.write(' '.join(obj) + "\n")
                            att_file.write(' '.join(att) + "\n")

imgid_file.close()
obj_file.close()
att_file.close()
path = '/home/akadar/hieratt/attributes_data/'
imgids = map(int, open(path+"imgids.txt").read().split('\n')[:-1])
atts = open(path+"attributes.txt").read().split('\n')[:-1]
objs = open(path+"objects.txt").read().split('\n')[:-1]


if names == "synsets":
    objs = map(lambda x: x.replace('.', '').replace('_', '').replace('-', ''),
               objs)
else:
    objs = map(lambda x: x.replace(' ', '').replace('-', ''), objs)

print len(objs), len(imgids), len(atts)
print "Computing indicator vectors for attirbutes (target)"
attribute_vectorizer = CountVectorizer(binary=True)
object_tokenizer = Tokenizer()
A = attribute_vectorizer.fit_transform(atts).todense()
print A.shape
print "Converting object labels to indices"
object_tokenizer.fit_on_texts(objs)
O = object_tokenizer.texts_to_sequences(objs)
print len(O)
print "Number of Classes that appear more than", N, "times"
print "---------------------------------------"
print "Object classes:", len(object_tokenizer.word_counts)
print "Attribute classes:", len(attribute_vectorizer.vocabulary_)
print
print "Writing attribute vectors to 'attributes.npy'"
np.save(path+"attributes", A)

print "Writing object indices to  'objects.pkl'"
pickle.dump(O, open(path+"objects.pkl", 'w'))

print "Writing object encoder to 'object_tokenizer'.pkl"
pickle.dump(object_tokenizer, open(path+"object_tokenizer.pkl", "w"))

print "Writing attribute vectorizer to attribute_vectorizer.pkl"

pickle.dump(attribute_vectorizer, open(path+"attribute_vectorizer.pkl", "w"))


imgids_unique = sorted(set(imgids))
duplicates = sorted(list_duplicates(imgids))

f = h5py.File(path+'attributes.h5', 'w')  # HDF5 data base for the whole set
# data set for the pre-processed images (num images x H x W x 3)
dset = f.create_dataset("images",
                        (len(imgids_unique), target_size[0], target_size[1], 3),
                        dtype="uint8")
# data set for the image ids (num images x 1)
imgidset = f.create_dataset("imgids", (len(imgids_unique), 1), dtype="int32")
# data set with pointers from images to attributes/objects (num images x 2)
pointers = f.create_dataset("pointers", (len(imgids_unique), 2), dtype="int32")
# data set for attribute indicator vectors
attributes = f.create_dataset("attributes", A.shape)
objects = f.create_dataset("objects", (len(O),))

print "Writing attribute indicator vectors to HDF5"
print attributes.shape
attributes[...] = A
print "Writing object indices to HDF5"
print objects.shape
objects[...] = [y for x in O for y in x]

print "Writing", len(imgids_unique), "images to hdf5 database with size"
print target_size
print dset.shape
print imgidset.shape
print pointers.shape
for i, j in enumerate(imgids_unique):
    print i, '\r'
    a = load_image(j, target_size)
    if a != None:
        dset[i] = a
        imgidset[i] = j
        pointers[i] = [min(duplicates[i][1]), max(duplicates[i][1])]
