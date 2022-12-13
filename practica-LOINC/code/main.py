import os
import re
import csv
import pandas as pd
from data_preparation import build_learning_data_from

def get_svm_format(x, qid):
    """
    Builds proper svm format for loinc codes with more than one click
    """
    
    clicks = data['sum_clicks'][x]
    #if clicks == 0:
    #    return ""
    long_common_name = data['long_common_name'][x]
    component = data['component'][x]
    system = data['system'][x]
    return f"{clicks} qid:{qid} 1:{long_common_name} 2:{component} 3:{system}\n"

def sortSVM(lst):
    return lst[2]

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
    #print("[",query,"] BM25 rank:\n",data)

    # generate learning data in correct format for SVMrank 
    data['svm_format'] = data['index'].apply(lambda x: get_svm_format(x-1, qid))
    # drop duplicate rows using svm_format as primary key
    data = data.drop_duplicates(subset=['svm_format'])
    
    cut_train = int(data.shape[0]*0.8)
    # separate into training(80%) and testing(20%) and save dat files
    train_dat = ""
    train = data.copy().iloc[:cut_train, :]
    for i in train.index.values.tolist():
        train_dat += train['svm_format'][i]
    with open("result/train.dat", "a") as f:
        f.writelines(train_dat)
    
    test_dat = ""
    test = data.copy().iloc[cut_train:, :]
    for i in test.index.values.tolist():
        test_dat += test['svm_format'][i]
    with open("result/test.dat", "a") as f:
        f.writelines(test_dat)
    test.to_csv(f"result/data_test_q{qid}.csv")

    print(f"Generated appropiate data for query \"{query}\"")

#generate the model with SVMrank
command = f"./aux_files/svm_rank_learn -c {qid+1} result/train.dat result/model.dat > result/svm_learn_info.txt"
os.system(command)
print("Generated model")

#obtain search results with SVMrank for each query
command = "./aux_files/svm_rank_classify result/test.dat result/model.dat result/prediction_order.txt > result/svm_rank_info.txt"
os.system(command)
print("Generated prediction")

# transform SVMrank results back to a human readable list
with open("result/prediction_order.txt") as f:
    results = f.read().split("\n")[:-1]
with open("result/test.dat") as f:
    test_dat = f.read().split("\n")[:-1]

aux = []
for elem in test_dat:
    gr = re.match(r"^\d+ qid:(\d+) 1:([\d.-]+).*", elem)
    aux.append(["" + gr.group(1), "" + gr.group(2)]) #qid, long_common_name index

# qid, long_common_name index, results svm
all_together = [[aux[i][0], aux[i][1], results[i]] for i in range(len(results))]
all_together.sort(key=sortSVM)

human_results = ""

for [qid, lcn_index, result] in all_together:
    data = pd.read_csv(f"result/data_test_q{qid}.csv")
    data['this_one'] = data['long_common_name'].apply(lambda x: abs(float(x) - float(lcn_index)) < 1e-10)
    data = data.set_index('Unnamed: 0')

    index_used = data.this_one[data['this_one']].index
    # if there's more than one index used, choose the one with svm_format not nan
    for i in range(len(index_used)):
        if not pd.isna(data['svm_format'][index_used[i]]):
            index = index_used[i]
            break

    data_original = pd.read_csv(f"input/{query_list[int(qid)]}.csv")
        
    human_results += "query \"" + query_list[int(qid)].replace("_"," ") + "\": "+ data_original['loinc_num'][index] + "\t" + data_original['long_common_name'][index] + "\t" + data_original['component'][index] + "\t" + data_original['system'][index] + "\t" + data_original['property'][index] + "\n"

# save human readable prediction results
with open("result/human_readable_prediction_order.txt", "w") as f:
        f.writelines(human_results)
print("Generated human readable prediction\n End of execution.")
