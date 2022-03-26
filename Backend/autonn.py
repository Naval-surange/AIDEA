#not required


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import PowerTransformer
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import autokeras as ak

class autonn:
    def __init__(self, scoring_function = 'balanced_accuracy', n_iter = 50):
        self.scoring_function = scoring_function
        self.n_iter = n_iter

    def fit(self,X,y):
        X_train = X
        y_train = y

        categorical_values = []

        cat_subset = X_train.select_dtypes(include = ['object','category','bool'])

        for i in range(cat_subset.shape[1]):
            categorical_values.append(list(cat_subset.iloc[:,i].dropna().unique()))
            
        num_pipeline = Pipeline([
                        ('cleaner',SimpleImputer()),
                        ('scaler',PowerTransformer())
                        ])

        cat_pipeline = Pipeline([
                        ('cleaner',SimpleImputer(strategy = 'most_frequent')),
                        ('encoder',CatBoostEncoder(cols=categorical_values,random_state=101))
                    ])


        preprocessor = ColumnTransformer([
                        ('numerical', num_pipeline, make_column_selector(dtype_exclude=['object','category','bool'])),
                        ('categorical', cat_pipeline, make_column_selector(dtype_include=['object','category','bool']))
                        ])

        model_pipeline_steps = []
        model_pipeline_steps.append(('preprocessor',preprocessor))
        model_pipeline_steps.append(('feature_selector',SelectKBest(f_classif,k='all')))
        model_pipeline_steps.append(('estimator',ak.ImageClassifier()))
        model_pipeline = Pipeline(model_pipeline_steps)

        total_features = preprocessor.fit_transform(X_train).shape[1]
