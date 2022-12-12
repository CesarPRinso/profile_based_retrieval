import csv
import os
import pandas as pd
from data_preparation import build_learning_data_from

def get_svm_format(x, qid):
    """
    Builds proper svm format for loinc codes with more than one click
    """
    
    clicks = data['sum_clicks'][x]
    if clicks == 0:
        return ""
    long_common_name = data['long_common_name'][x]
    component = data['component'][x]
    system = data['system'][x]
    return f"{clicks} qid:{qid} 1:{long_common_name} 2:{component} 3:{system}\n"

if os.path.exists("result/train.dat"):
    os.remove("result/train.dat")
if os.path.exists("result/test.dat"):
    os.remove("result/test.dat")

with open("input/queries.txt") as f:
    query_list = f.read().split("\n")

# For each of the queries
for qid in range(len(query_list)):
    # read input file
    query = query_list[qid]
    data = pd.read_csv(f"input/{query}.csv")
    query = query.replace("_"," ")

    # run data through BM25
    data = build_learning_data_from(data, query)

    # generate learning data in correct format for SVMrank 
    data['svm_format'] = data['index'].apply(lambda x: get_svm_format(x-1, qid))
    
    # separate into training and testing and save dat files
    train_dat = ""
    train = data.copy()
    train = train.iloc[:50, :]
    for i in range(train.shape[0]):
        train_dat += train['svm_format'][i]
    with open("result/train.dat", "a") as f:
        f.writelines(train_dat)
    
    test_dat = ""
    test = data.copy()
    test = test.iloc[50:, :]
    for i in range(test.shape[0]):
        test_dat += test['svm_format'][50+i]
    with open("result/test.dat", "a") as f:
        f.writelines(test_dat)

    print(f"generated appropiate data for query \"{query}\"")
    
#generate the model with SVMrank
command = "./aux_files/svm_rank_learn -c 3 result/train.dat result/model.dat > result/svm_learn_output.txt"
os.system(command)

#obtain search results with SVMrank for each query
command = "./aux_files/svm_rank_classify result/test.dat result/model.dat result/prediction_order.txt > result/svm_rank_output.txt"
os.system(command)

# transform SVMrank results back to a human readable list

