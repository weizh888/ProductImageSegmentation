import numpy as np
import pandas as pd
np.random.seed(1)

full_labels = pd.read_csv('data/labels.csv')

grouped = full_labels.groupby('filename')
grouped.apply(lambda x: len(x)).value_counts()

print(grouped.apply(lambda x: len(x)).value_counts())

gb = full_labels.groupby('filename')
grouped_list = [gb.get_group(x) for x in gb.groups]
print("The total number of samples is {}.".format(len(grouped_list)))

n_train_images = len(grouped_list) * 4/5
n_test_images = len(grouped_list) - n_train_images
print("The number of training samples is {}.".format(n_train_images))
print("The number of testing samples is {}.".format(n_test_images))

train_index = np.random.choice(len(grouped_list), size=n_train_images, replace=False)
test_index = np.setdiff1d(list(range(len(grouped_list))), train_index)

# take first 200 files
train = pd.concat([grouped_list[i] for i in train_index])
test = pd.concat([grouped_list[i] for i in test_index])

train.to_csv('data/train_labels.csv', index=None)
test.to_csv('data/test_labels.csv', index=None)