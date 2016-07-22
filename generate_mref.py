from keras.datasets import mnist
import numpy as np
from scipy.misc import imresize
import cPickle as pickle
import argparse

def colorize(img, color):
    """Return colored version of MNIST image."""
    colored = np.zeros((img.shape[0],img.shape[1],3))
    if color == "red":
        colored[:,:,0] = img
    elif color == "green":
        colored[:,:,1] = img
    elif color == "blue":
        colored[:,:,2] = img
    elif color == "yellow":
        colored[:,:,0] = img
        colored[:,:,1] = img
    elif color == "white":
        colored[:,:,0] = img
        colored[:,:,1] = img
        colored[:,:,2] = img
    return colored

def modify(img, color, scale):
    """Color and resize MNIST image."""
    return  colorize(imresize(img, scale),color).astype("uint8")

def iter_MREF(data, labels, num_samples):
    # Initialize empty data set, queries and targets
    dataset = np.empty((num_samples, 100, 100, 3)).astype("uint8")
    queries = np.empty(num_samples).astype("int")
    targets = np.empty(num_samples).astype("str")
    colors = ["red", "green", "blue", "yellow", "white"]
    # Try as many times as we want samples
    for i in range(num_samples):
        print i, '\r',
        # Initialize empty canvas
        canvas = np.zeros((100,100, 3)).astype("uint8")
        used_labels = []
        used_colors = []
        nums = 0
        # Until we manage to come up with a picture with at least 5 numbers
        while nums < 5:
            # Try at most 9 times
            for j in range(9):
                # pick image, scale, and color
                ind = np.random.randint(0, len(data))
                label = labels[ind]
                # Do not reuse the same number
                if label in used_labels:
                    continue
                else:
                    # Pick a random image, color and scale
                    img = data[ind]
                    scale = np.random.uniform(0.5, 3.5, 1)[0]
                    color = np.random.choice(colors)
                    # Color the image white and rescale
                    candidate = modify(img, "white", scale)
                    s = candidate.shape[0]
                    # Pick random location for the image (top-left corner)
                    x, y = np.random.randint(0, 100-s, size=(2,))
                    candidate_loc = np.copy(canvas[y:y+s, x:x+s, :])
                    # If the image doesn't overlap with anything on the canvas
                    if np.all(candidate_loc * candidate == 0):
                        nums += 1
                        candidate = modify(img, color, scale)
                        canvas[y:y+s, x:x+s, :] += candidate
                        used_labels.append(label)
                        used_colors.append(color)
        # After an image is generated pick a target number and it's color as q
        pick = np.random.randint(0, len(used_labels))
        dataset[i] = canvas
        queries[i] = used_labels[pick]
        targets[i] = used_colors[pick]
    return dataset, queries, targets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("filename")
    parser.add_argument("--num_train", type=int, default=60000)
    parser.add_argument("--num_test", type=int, default=10000)
    args = parser.parse_args()

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    train_mref, train_queries, train_targets = iter_MREF(X_train,
                                                         y_train,
                                                         args.num_train)
    test_mref, test_queries, test_targets = iter_MREF(X_test,
                                                      y_test,
                                                      args.num_test)
    np.savez_compressed(args.filename, 
                train_data=train_mref, train_queries=train_queries, train_targets=train_targets,
                test_data=test_mref, test_queries=test_queries, test_targets=test_targets)
