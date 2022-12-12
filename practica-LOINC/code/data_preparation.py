import re
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import StandardScaler
import pandas as pd


def clean_string(string):
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def rank(df, query):
    # using BM25 for rank result of loinc, using the similarity of tokens

    names = df["long_common_name"].values.tolist()
    component = df["component"].values.tolist()
    system = df["system"].values.tolist()
    # property = df["property"].values.tolist()

    tokenized_name = [clean_string(doc).split(" ") for doc in names]
    tokenized_component = [clean_string(doc).split(" ") for doc in component]
    tokenized_system = [clean_string(doc).split(" ") for doc in system]
    # tokenized_property = [clean_string(doc).split(" ") for doc in property]

    bm25_name = BM25Okapi(tokenized_name)
    bm25_component = BM25Okapi(tokenized_component)
    bm25_system = BM25Okapi(tokenized_system)
    # bm25_property = BM25Okapi(tokenized_property)

    tokenized_query = query.split()
    names_scores = bm25_name.get_scores(tokenized_query)
    component_scores = bm25_component.get_scores(tokenized_query)
    system_scores = bm25_system.get_scores(tokenized_query)
    # property_scores = bm25_property.get_scores(tokenized_query)

    df["long_common_name"] = names_scores
    df["component"] = component_scores
    df["system"] = system_scores
    df = df.sort_values(by=['long_common_name','system', 'component'], ascending=False)
    #print("[",query,"] BM25 rank:\n",df[['loinc_num', 'long_common_name', 'component', 'system','sum_clicks']])
    return df


def build_learning_data_from(data_loinc, query):
    df_rank = pd.DataFrame(columns=['long_common_name', 'component', 'system'])
    dfprob = rank(data_loinc, query)
    learning_data = dfprob.loc[:, :]

    scaler = StandardScaler()
    df_rank['index'] = range(1, len(learning_data) + 1)
    df_rank['long_common_name'] = scaler.fit_transform(learning_data[['long_common_name']])
    df_rank['component'] = scaler.fit_transform(learning_data[['component']])
    df_rank['system'] = scaler.fit_transform(learning_data[['system']])
    df_rank['sum_clicks'] = learning_data['sum_clicks']
    #print("[",query,"] scaled rank:\n",df_rank)
    return df_rank
