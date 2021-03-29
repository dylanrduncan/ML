from sklearn.datasets import load_digits

digits = load_digits()

# digits object is a 'bunch'-- basically a dictionary
# two parts to machine learning
# 1 - training
# 2 - testing
# two types as well- supervised & unsupervised

# print(digits.DESCR)  # constraints the dataset's description
"""
print(digits.data[13])  # numpy array that contains the 1797 samples

print(digits.data.shape)

print(digits.target[13])

print(digits.target.shape)

print(digits.images[13])

import matplotlib.pyplot as plt

figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(6, 4))
# python zip function bundles the 3 iterables and
# produces one iterable
for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target = item
    # displays multichannel (RGB) or single-channel (grayscale)
    # image data.
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)
plt.tight_layout()
plt.show()
"""
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    digits.data, digits.target, random_state=11
)  # random_state for reproducibility

print(data_train.shape)

print(target_train.shape)

print(data_test.shape)

print(target_test.shape)
# targets have no second value because there's only 1 column-- that of which is the target number

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

# load the training data into the model using the fit method
# Note: the KNeighborClassifier fit method does not do calculations it just loads
# the model
knn.fit(X=data_train, y=target_train)
# Returns an array containing the predicted class of each test image:
# creates an array of digits

predicted = knn.predict(X=data_test)

expected = target_test

print(predicted[:20])
print(expected[:20])