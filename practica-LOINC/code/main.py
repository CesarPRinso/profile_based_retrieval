import numpy as np
import csv

from data_preparation import build_learning_data_from
from r_train import r_train
from r_predict import r_predict
import pandas as pd
import pylab as pl

x = []
for t in csv.reader(open('text/glucose_in_blood.csv', 'r')):
    x.append(t)
x = np.asarray(x, dtype='object')

rank1 = pd.read_csv('text/glucose_in_blood.csv', encoding="ISO-8859-1")

# new_data_set = prepare_dataset(rank1, 'glucose in blood')  # data
learning_data = build_learning_data_from(rank1)
# learning_data = learning_data.to_numpy()
train = learning_data.iloc[:50, :]
test = learning_data.iloc[50:, :]
y = []
for t in csv.reader(open('text/y.csv', 'r')):
    y.append(t)
y = np.asarray(y, 'd')

# train
rsvm = r_train(train, y[:50, :])
# rank
r = r_predict(rsvm, test)


pl.scatter(r[:, 0], r[:, 1])
pl.plot([0, len(r)], [r[4, 1], r[4, 1]], 'k--', lw=2)
pl.xlabel('CANDIDATE_ID')
pl.ylabel('SCORE')
pl.show()
