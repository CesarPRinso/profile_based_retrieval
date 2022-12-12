import numpy as np
import csv

from data_preparation import build_learning_data_from
#from r_train import r_train
#from r_predict import r_predict
import pandas as pd
import pylab as pl

def get_svm_format(x, qid):
    long_common_name = data['long_common_name'][x]
    component = data['component'][x]
    system = data['system'][x]
    clicks = data['sum_clicks'][x]
    return f"{clicks} qid:{qid} 1:{long_common_name} 2:{component} 3:{system}\n"

with open("input/queries.txt") as f:
    query_list = f.read().split("\n")

# For each of the queries
for qid in range(len(query_list)):
    query = query_list[qid]
    data = pd.read_csv(f"input/{query}.csv")
    query = query.replace("_"," ")

    #Run data through BM25
    data = build_learning_data_from(data, query)

    #generate learning data in correct format for SVMrank and save to .dat files
    data['svm_format'] = data['index'].apply(lambda x: get_svm_format(x-1, qid))
    
    #separate into training and testing
    train = data.copy()
    train = train.iloc[:50, :]
    test = data.copy()
    test = test.iloc[50:, :]
    
    print(f"generated appropiate data for query: \"{query}\"\n")

    #save SVMrank format files
    with open("result/intermediate/train.dat", "a") as f:
        f.writelines(train['svm_format'].to_string(index=False))
    
#generate the model with SVMrank

#obtain search results with SVMrank for each query



    
    
"""
for linea in archivo:
    linea = linea.replace('\n', "")
    #process queries
    query = linea.replace("_", " ")
    rank1 = pd.read_csv('loinc_dataset/' + linea + '.csv', encoding="ISO-8859-1")
    learning_data = build_learning_data_from(rank1, query)
       
    #save intermediate results 
    learning_data.to_csv('result/out_intermediate_' + linea + '.csv')
    #separate dataset into train and test
    train = learning_data.iloc[:50, :]
    test = learning_data.iloc[50:, :]

    #generate learning data in correct format for SVMrank
        
        
    # apply rsvm and save results
    rsvm = r_train(train, y[:50, :])
    r = r_predict(rsvm, test)
    print(r)
    #r.to_csv('result/out_' + linea + '.csv', encoding="ISO-8859-1")

        
    pl.scatter(r[:, 0], r[:, 1])
    pl.plot([0, len(r)], [r[4, 1], r[4, 1]], 'k--', lw=2)
    pl.xlabel('entry')
    pl.ylabel('score')
    pl.savefig('result/'+ linea + '.png')
    #pl.show()
    pl.close()
"""
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
