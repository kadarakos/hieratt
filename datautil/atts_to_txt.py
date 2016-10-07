import json
import os

def data_filter(sample):
    """Filter attributes and objects that are not in the most common."""
    # print sample
    # print len(sample['attributes']), len(sample['names'])
    new_sample = sample.copy()
    atts = filter(lambda x: x in common_att, sample['attributes'])
    objs = filter(lambda x: x in common_obj, sample['object_names'])
    # print len(atts), len(objs)
    if atts and objs:
        new_sample['attributes'] = atts
        new_sample['object_names'] = objs
        return new_sample

print  "Loading data set"

VG_path = "/roaming/public_datasets/VisualGenome/attributes/"
DATA = "attributes_v1.json"
data = json.load(open(VG_path+DATA))

print "Computing attribute and object frequencies"

# Count the occurrences of attributes and objects in the data set
att_freq = {}
obj_freq = {}
for i in data:
    for j in i['attributes']:
        if j.has_key('attributes'):
            for att in j['attributes']:
                if att in att_freq:
                    att_freq[att] += 1
                else:
                    att_freq[att] = 1
            for obj in j['object_names']:
                if obj in obj_freq:
                    obj_freq[obj] += 1
                else:
                    obj_freq[obj] = 1


# The paper says We used 488 object classes and 274 attribute classes that appear more than 100 times.
# I can't replicate this, because there are more than 1000 obj and att classes that appear more than 100 times
# even with the old version, so i dont know what they mean
# I just choose top 300 attributes and 500 objects

# num_att = 300
# num_obj = 500
# common_att = dict(sorted(att_freq.iteritems(), key=lambda x: x[1], reverse=True)[:num_att])
# common_obj = dict(sorted(obj_freq.iteritems(), key=lambda x: x[1], reverse=True)[:num_obj])

common_att = dict(filter(lambda x: x[1] > 500, att_freq.iteritems()))
common_obj = dict(filter(lambda x: x[1] > 500, obj_freq.iteritems()))

data_path = "./attributes_data/"
if not os.path.exists(data_path):
    os.makedirs(data_path)

print "Creating data strings"
imgid_file = open(data_path+"imgids.txt", "wb")
obj_file = open(data_path+"objects.txt", "wb")
att_file = open(data_path+"attributes.txt", "wb")

print len(data)
for i in range(0, len(data)):
    print i, '\r',
    sample = data[i]['attributes']
    imgid = data[i]['id']
    for j in sample:
        if j.has_key('attributes'):
            filtered = data_filter(j)
            if filtered:
                att, obj = filtered['attributes'], filtered['object_names']
                imgid_file.write(str(imgid) + "\n")
                obj_file.write(' '.join(obj) + "\n")
                att_file.write(' '.join(att ) + "\n")

