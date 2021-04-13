from sklearn.model_selection import train_test_split
from sklearn.model_selection import Kfold
from catboost import CatBoostClassifier


def get_feature_names():
  return ['receipt_id', 'receipt_dayofweek', 'receipt_time', 
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
                 
                 'last_first_word',
                ] 

def get_cat_features_names():
  return ['receipt_time', 'receipt_id', 'hours', 'min_hours', 'max_hours', 
                'median_dayofweek', 'first', 'last', 'second_last_word', 'second_first_word',
                 'last_first_word']

def get_text_features_names():
  return ['item_name']

def get_catboost_model(verbose = 50, loss_function='MultiClass', eval_metric='TotalF1', task_type='GPU', iterations=1000,
                      learning_rate=0.2):
  return CatBoostClassifier(
    cat_features=get_cat_features_names(),
    text_features=get_text_features_names(),
    verbose=verbose,
    loss_function=loss_function,
    eval_metric=eval_metric,
    task_type=task_type,
    iterations=iterations,
    learning_rate=learning_rate,      
#     reg_lambda=0.0001,
        text_processing = {
        "tokenizers" : [{
            "tokenizer_id" : "Space",
            "separator_type" : "ByDelimiter",
            "delimiter" : " "
        }],

        "dictionaries" : [{
            "dictionary_id" : "BiGram",
            "token_level_type": "Letter",
            "max_dictionary_size" : "150000",
            "occurrence_lower_bound" : "1",
            "gram_order" : "2"
        },{
            "dictionary_id" : "Trigram",
            "max_dictionary_size" : "150000",
            "token_level_type": "Letter",
            "occurrence_lower_bound" : "1",
            "gram_order" : "3"
        },{
            "dictionary_id" : "Fourgram",
            "max_dictionary_size" : "150000",
            "token_level_type": "Letter",
            "occurrence_lower_bound" : "1",
            "gram_order" : "4"
        },{
            "dictionary_id" : "Fivegram",
            "max_dictionary_size" : "150000",
            "token_level_type": "Letter",
            "occurrence_lower_bound" : "1",
            "gram_order" : "5"
        },{
            "dictionary_id" : "Sixgram",
            "max_dictionary_size" : "150000",
            "token_level_type": "Letter",
            "occurrence_lower_bound" : "1",
            "gram_order" : "6"
        }
        ],

        "feature_processing" : {
            "default" : [
                    {
                    "dictionaries_names" : ["BiGram", "Trigram", "Fourgram", "Fivegram", "Sixgram"],
                    "feature_calcers" : ["BoW"],
                    "tokenizers_names" : ["Space"]
                },
                    {
                "dictionaries_names" : ["BiGram", "Trigram", "Fourgram", "Fivegram", "Sixgram"],
                "feature_calcers" : ["NaiveBayes"],
                "tokenizers_names" : ["Space"]
            },{
                "dictionaries_names" : [ "BiGram", "Trigram", "Fourgram", "Fivegram", "Sixgram"],
                "feature_calcers" : ["BM25"],
                "tokenizers_names" : ["Space"]
            },
            ],
        }
    }
  )

def eval_f1_score(pred, y_valid):
  print('F1_score: ')


def run_cv(model=None, data, k=5, target='category_id', feature_names=get_feature_names(), valid_data):
  kf = Kfold(n_splits=k,
   random_state=42,
    shuffle=True)


  y = data[target]
  X = data[feature_names]

  y_valid = valid_data[target]
  x_valid = valid_data[feature_names]


  if model == None:
    model = get_catboost_model()

  for i, (train_index, test_index) in enumerate(kf.split(data)):
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    X_train, X_valid = X.iloc[train_index, :], X.iloc[test_index, :]
    print("\nFold: ", i)

    fit_model = model.fit(X_train, y_train)

    pred = fit_model.predict(X_valid)
    eval_f1_score(pred, y_valid)
  
  y_test_pred /= k

  print('\nWeighted F1 Score for full training set:')





  

  
