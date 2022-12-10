import numpy as np
import csv

from data_preparation import build_learning_data_from
from r_train import r_train
from r_predict import r_predict
import pandas as pd
import pylab as pl

y = []
for t in csv.reader(open('text/y.csv', 'r')):
    y.append(t)
y = np.asarray(y, 'd')

with open("text/queries.txt") as archivo:
    for linea in archivo:
        query = linea.replace('\n', "").replace("_", " ")
        rank1 = pd.read_csv('loinc_dataset/' + linea.replace('\n', "") + '.csv', encoding="ISO-8859-1")
        learning_data = build_learning_data_from(rank1, query)
        learning_data.to_csv('out_' + linea.replace('\n', "") + '.csv')
        train = learning_data.iloc[:50, :]
        test = learning_data.iloc[50:, :]
        rsvm = r_train(train, y[:50, :])
        r = r_predict(rsvm, test)
        pl.scatter(r[:, 0], r[:, 1])
        pl.plot([0, len(r)], [r[4, 1], r[4, 1]], 'k--', lw=2)
        pl.xlabel('CANDIDATE_ID')
        pl.ylabel('SCORE')
        pl.show()
'''
rank1 = pd.read_csv('loinc_dataset/glucose_in_blood.csv', encoding="ISO-8859-1")

learning_data = build_learning_data_from(rank1, "glucose in blood")
train = learning_data.iloc[:50, :]
test = learning_data.iloc[50:, :]

# train
rsvm = r_train(train, y[:50, :])
# rank
r = r_predict(rsvm, test)

pl.scatter(r[:, 0], r[:, 1])
pl.plot([0, len(r)], [r[4, 1], r[4, 1]], 'k--', lw=2)
pl.xlabel('CANDIDATE_ID')
pl.ylabel('SCORE')
pl.show()
'''
