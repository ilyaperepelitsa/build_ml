from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

import numpy as np

# we load the data with load_iris from sklearn
data = load_iris()
features = data["data"]
feature_names = data["feature_names"]
target = data["target"]
labels = data["target_names"]

for t, marker, c in zip(range(3), ">ox", "rgb"):
    # we plot each class on its own to get different colored markers
    plt.scatter(features[target == t, 0],
                features[target == t, 1],
                marker = marker,
                c = c)
# plt.show()
plength.labels

plength = features[:, 2]
# use numpy operations to get setosa features
# is_setosa = (labels == "setosa")
# This is the important step
max_setosa = plength[data["target_names"] == "setosa"].max()
