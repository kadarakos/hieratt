"""Wabalabadubdub."""

from collections import defaultdict
from numpy import random


class AttributeIterator():
    """Yield batches of triplets (img, obj, att) from VG."""

    def __init__(self, database, batchsize=5, n_samples=1000,
                 shuffle=False, test_samples=100):
        """Initialize with the necessary components to yield."""
        self.n_samples = n_samples                  # Number of samples
        self.test_samples = test_samples            # Number of test samples
        self.batchsize = batchsize
        self.shuffle = shuffle                      # Shuffle before iter
        self.database = database                    # HDF5 instance
        self.images = self.database['images']       # image tensors
        self.imgids = self.database['imgids']       # ids from the original set
        self.pointers = self.database['pointers']   # point to obj, att from im
        self.objects = self.database['objects']     # object vectors
        self.attributes = self.database['attributes']   # attribute vetors
        self.indices = range(0, self.n_samples)
        if self.shuffle:
            shuffle(self.indices)
        self.val = self.indices[self.n_samples-test_samples:]    # val portion
        self.train = self.indices[:self.n_samples-test_samples]  # train portio
        self.end = len(self.train)
        self.reset_epoch()

    def reset_epoch(self):
        """Set state for a new epoch, cursor to 0, shuffle training set."""
        self.cursor = 0
        random.shuffle(self.train)

    def pick_pointer(self, i):
        """
        Desctruct minium and maximum and choose a random value in between.

        It looks this weird, because I vectorized it for speed.
        """
        minimum, maximum = self.pointers[i].T
        indices = minimum + (maximum - minimum)/random.randint(0, 10)
        return list(indices)

    def __iter__(self):
        """Initalize iterator."""
        return self

    def next(self):
        """Iterate over the training data.

        Return the next value till current is lower than high.
        """
        if self.cursor >= self.end:
            self.reset_epoch()
        bottom = self.cursor
        top = min(self.cursor + self.batchsize, self.end)
        self.cursor = top
        indices = sorted(self.train[bottom:top])
        imgs = self.images[indices]
        pointer = self.pick_pointer(indices)
        objs = self.objects[pointer]
        atts = self.attributes[pointer]
        return imgs, objs, atts

    def __str__(self):
        """Show items in the HDF5 file object."""
        return str(self.database.items())


def list_duplicates(seq):
    """Given the imgids list, return the ranges of the same elements."""
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items())
