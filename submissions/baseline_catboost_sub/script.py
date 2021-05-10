
from nltk.corpus import stopwords
import string
import pandas as pd
import numpy as np
import pickle
import nltk
import sys

sys.path.append("./razdel/")


from razdel import tokenize

nltk.data.path.append("nltk_models")

feature_names = ['receipt_id', 'receipt_dayofweek', 'receipt_time', 
                 'item_name', 'item_quantity', 'item_price', 'item_nds_rate', 
                 'mean_item_quantity', 'sum_item_quantity', 'std_item_quantity', 
                 'min_item_quantity', 'max_item_quantity', 'mean_item_price',
                 'sum_item_price', 'std_item_price', 'min_item_price',
                 'max_item_price', 'mean_item_nds_rate', 'sum_item_nds_rate', 
                 'std_item_nds_rate', 'min_item_nds_rate', 'max_item_nds_rate', 
                 'hours', 'mean_hours', 'sum_hours', 
                 'std_hours', 'min_hours', 'max_hours', 'median_dayofweek',
                 
                 'first',
                 'last', 'is_preservativ', 'is_paper', 'is_otkritka', 'is_born', 'is_a4', 'is_ubiley',                                         
                 'is_auto', 'is_list', 'is_kanc', 'is_dnevnik',
                
                'first_word_len', 'last_word_len',
                 'second_last_word', 'second_first_word',
                 'last_first_word'] 

cat_features = ['receipt_time', 'receipt_id', 'hours', 'min_hours', 'max_hours', 
                'median_dayofweek', 'first', 'last', 'second_last_word', 'second_first_word',
                 'last_first_word']
text_features = ['item_name']
RU_STOP_WORDS = set(stopwords.words('russian'))


def create_features_part_1(data):
    gr_item_name = data.groupby('item_name')
    data['mean_item_quantity'] = gr_item_name['item_quantity'].transform("mean")
    data['sum_item_quantity'] = gr_item_name['item_quantity'].transform("sum")
    data['std_item_quantity'] = gr_item_name['item_quantity'].transform("std")
    data['min_item_quantity'] = gr_item_name['item_quantity'].transform("min")
    data['max_item_quantity'] = gr_item_name['item_quantity'].transform("max")

    data['mean_item_price'] = gr_item_name['item_price'].transform("mean")
    data['sum_item_price'] = gr_item_name['item_price'].transform("sum")
    data['std_item_price'] = gr_item_name['item_price'].transform("std")
    data['min_item_price'] = gr_item_name['item_price'].transform("min")
    data['max_item_price'] = gr_item_name['item_price'].transform("max")

    data['mean_item_nds_rate'] = gr_item_name['item_nds_rate'].transform("mean")
    data['sum_item_nds_rate'] = gr_item_name['item_nds_rate'].transform("sum")
    data['std_item_nds_rate'] = gr_item_name['item_nds_rate'].transform("std")
    data['min_item_nds_rate'] = gr_item_name['item_nds_rate'].transform("min")
    data['max_item_nds_rate'] = gr_item_name['item_nds_rate'].transform("max")


    data['hours'] = data['receipt_time'].apply(lambda x: int(x.split(":")[0]))

    data['mean_hours'] = gr_item_name['hours'].transform("mean")
    data['sum_hours'] = gr_item_name['hours'].transform("sum")
    data['std_hours'] = gr_item_name['hours'].transform("std")
    data['min_hours'] = gr_item_name['hours'].transform("min")
    data['max_hours'] = gr_item_name['hours'].transform("max")


    data['median_dayofweek'] = gr_item_name['receipt_dayofweek'].transform('median')
    data['median_dayofweek'] = data['median_dayofweek'].astype(np.int8)
    return data

def create_features_part_2(data):
    data['first'] = data['item_name'].apply(lambda x: x.split()[0] if len(x.split()) > 0 else 'none')
    data['last'] = data['item_name'].apply(lambda x: x.split()[-1] if len(x.split()) > 0 else 'none')
    data['is_preservativ'] = data.item_name.apply(lambda x: int('презерватив' in x.lower()))
    data['is_auto'] = data.item_name.apply(lambda x: int('авто' in x.lower() or \
                                                            'щетк' in x.lower() or \
                                                             'сантиметр' in x.lower()))

    data['is_paper'] = data.item_name.apply(lambda x: int('бумага' in x.lower()))
    data['is_otkritka'] = data.item_name.apply(lambda x: int('открытка' in x.lower()))
    data['is_born'] = data.item_name.apply(lambda x: int('рождения' in x.lower()))
    data['is_a4'] = data.item_name.apply(lambda x: int('а4' in x.lower()))
    data['is_ubiley'] = data.item_name.apply(lambda x: int('юбиле' in x.lower()))

    #24
    data['is_list'] = data.item_name.apply(lambda x: int('лист' in x.lower()))
    data['is_kanc'] = data.item_name.apply(lambda x: int('канцелярский' in x.lower()))
    data['is_dnevnik'] = data.item_name.apply(lambda x: int('дневник' in x.lower()))
    
    data['second_last_word'] = data.item_name.apply(lambda x: x.split()[-2] if len(x.split()) > 1 else 'none')
    data['second_first_word'] = data.item_name.apply(lambda x: x.split()[1] if len(x.split()) > 1 else 'none')

    data['first_word_len'] = data['first'].apply(len)
    data['last_word_len'] = data['last'].apply(len)

    data['item_name_len'] = data.item_name.apply(lambda x: len(x.split()))

    data['last_first_word'] = data['last'] + data['first']
    return data


def fill_median(df):
    df['std_item_quantity'] = df['std_item_quantity'].fillna(np.nanmedian(df['std_item_quantity']))
    df['std_item_price'] = df['std_item_price'].fillna(np.nanmedian(df['std_item_price']))
    df['std_item_nds_rate'] = df['std_item_nds_rate'].fillna(np.nanmedian(df['std_item_nds_rate']))
    df['std_hours'] = df['std_hours'].fillna(np.nanmedian(df['std_hours']))
    return df



def pre_process(text):
    return " ".join([token.lower() for token in [_.text for _ in tokenize(text)]
                     if (token not in RU_STOP_WORDS) or (token not in string.punctuation)])

def main():
    test = pd.read_parquet('data/task1_test_for_user.parquet')

    with open("catboost.clf", 'rb') as fout:
        clf = pickle.loads(fout.read())
    
    test['item_name'] = test.item_name.apply(pre_process)
    
    test = create_features_part_1(test)
    test = create_features_part_2(test)
    test = fill_median(test)

    pred = clf.predict(test[feature_names])
    res = pd.DataFrame(pred, columns=['pred'])
    res['id'] = test['id'].values
    res[['id', 'pred']].to_csv('answers.csv', index=None)



if __name__ == "__main__":
    main()