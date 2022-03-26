import numpy as np
import pandas as pd
import time
import zipfile
import uuid
import re
import gc
import base64
import shutil
import autokeras as ak
import dill as pickle
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import cross_val_score, HalvingRandomSearchCV, HalvingGridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from xgboost.sklearn import XGBRegressor
import lightgbm as lgb      
import streamlit as st
# import shap

# parser = argparse.ArgumentParser()
# parser.add_argument('--savemodel', choices=['yes', 'no'])
# parser.add_argument('--method', choices=['ml', 'dl'],required=True)
# parser.add_argument('--savetype',choices=['tf','keras'],default='keras')
# args = parser.parse_args()

class autoregression:
    def __init__(self, scoring_function = 'neg_mean_squared_error', n_iter = 100, savemodel='yes',method='ml',
                savetype='tf',maxtrials=10, n_splits = 5, exp_name = 'AIDEA'):
        self.scoring_function = scoring_function
        self.n_iter = n_iter
        self.savemodel = savemodel
        self.method = method
        self.savetype = savetype
        self.maxtrials = maxtrials
        self.n_splits = n_splits
        self.class_imbalance = False
        self.best_model_name = "Best Model"
        self.exp_name = exp_name

    def fit(self,X,y):

        categorical_values = []

        cat_subset = X.select_dtypes(include = ['object','category','bool'])        

        for i in range(cat_subset.shape[1]):
            categorical_values.append(list(cat_subset.iloc[:,i].dropna().unique()))
        
        num_pipeline = Pipeline([
                        ('cleaner',SimpleImputer()),
                        ('scaler',PowerTransformer())
                        ])

        cat_pipeline = Pipeline([
                        ('encoder',CatBoostEncoder(cols=cat_subset.columns,random_state=101)),
                        ('cleaner',SimpleImputer(strategy = 'most_frequent')),
                    ])

        preprocessor = ColumnTransformer([
                        ('numerical', num_pipeline, make_column_selector(dtype_exclude=['object','category','bool'])),
                        ('categorical', cat_pipeline, make_column_selector(dtype_include=['object','category','bool'])),
                        ])
        

        if self.method=="ml":
            estimators = [
                LinearRegression(n_jobs = 4),
                Ridge(max_iter=1000,random_state=101),
                Lasso(max_iter=1000,random_state=101),
                RandomForestRegressor(random_state=101,n_jobs = 4),
                KNeighborsRegressor(n_jobs = 4),
                SVR(), 
                XGBRegressor(random_state=101,use_label_encoder=False,n_jobs = 4),
                lgb.sklearn.LGBMRegressor(random_state=101,n_jobs = 4),
                ExtraTreesRegressor(random_state=101,n_jobs = 4),
                GradientBoostingRegressor(max_depth=10,random_state=101),
                DecisionTreeRegressor(random_state=101,max_features='auto')
            ]


            model_pipeline_steps = []
            
            model_pipeline_steps.append(('preprocessor',preprocessor))
            model_pipeline_steps.append(('variance_threshold',VarianceThreshold(threshold=0)))
            model_pipeline_steps.append(('feature_selector',SelectKBest(f_regression,k='all')))
            model_pipeline_steps.append(('estimator',LinearRegression()))
            
            model_pipeline = Pipeline(model_pipeline_steps)
            
            try:
                total_features = preprocessor.fit_transform(X).shape[1]
            except:
                total_features = preprocessor.fit_transform(X,y).shape[1]
                

            optimization_grid = []

            # all estimators
            optimization_grid.append({
                'preprocessor__numerical__scaler':[RobustScaler(),StandardScaler(),MinMaxScaler(),PowerTransformer()],
                'preprocessor__numerical__cleaner__strategy':['mean','median'],
                'feature_selector__k': list(np.arange(1,total_features,1)) + ['all'],
                # 'feature_selector__n_features_to_select': list(np.arange(1,total_features+1,1)),
                'estimator':estimators
            })

            search = HalvingGridSearchCV(
                    model_pipeline,
                    optimization_grid,
                    factor=4,
                    scoring = self.scoring_function, 
                    n_jobs = 2,
                    verbose = 1,
                    cv = self.n_splits,
                    )

            search.fit(X, y)
            self.best_estimator = search.best_estimator_
            self.best_pipeline = search.best_params_
            
            feat = search.best_estimator_.named_steps['feature_selector'].get_support()
            try:
                selected_feat = list(X.columns[feat])
            except:
                X_dummy = pd.get_dummies(X)
                selected_feat = list(X_dummy.columns[feat])
            self.select_feat = selected_feat
            
        elif self.method=="dl":

            model_pipeline_steps = []
            model_pipeline_steps.append(('preprocessor',preprocessor))
            model_pipeline_steps.append(('feature_selector',SelectKBest(f_regression,k='all')))
            model_pipeline_steps.append(('estimator',ExtraTreesRegressor()))
            model_pipeline = Pipeline(model_pipeline_steps)

            try:
                total_features = preprocessor.fit_transform(X).shape[1]
            except:
                total_features = preprocessor.fit_transform(X,y).shape[1]


            optimization_grid = []

            optimization_grid.append({
                'preprocessor__numerical__scaler':[RobustScaler(),StandardScaler(),MinMaxScaler(),PowerTransformer()],
                'preprocessor__numerical__cleaner__strategy':['mean','median'],
                'feature_selector__k': list(np.arange(1,total_features,1)) + ['all'],
                'estimator':[ExtraTreesRegressor(random_state=101)],
            })

            grid_search = HalvingGridSearchCV(model_pipeline,
                       param_grid=optimization_grid,
                       scoring=self.scoring_function,
                       random_state=101,
                       factor=4,
                       n_jobs = 2,
                       verbose=2,
                       cv=3)

            grid_search.fit(X, y)
            feat = grid_search.best_estimator_.named_steps['feature_selector'].get_support()
            feat_names = list(X.columns)
            selected_feat = [feat_names[idx] for idx,feat in enumerate(feat) if feat==True]
            
            
            X_trans = grid_search.best_estimator_.named_steps['preprocessor'].transform(X)
            X_new = pd.DataFrame(X_trans,columns=list(X.columns))
            X_select = X_new[selected_feat]
            X_train, X_test, y_train, y_test = train_test_split(X_select, y, test_size=0.2,stratify=y)

            search = ak.StructuredDataRegressor(max_trials=self.maxtrials,seed=0,project_name=self.exp_name)
            search.fit(X_train, y_train,validation_data=(X_test,y_test))

            best_model = search.export_model()
            self.best_estimator = best_model
            self.best_pipeline = model_pipeline
            self.best_search = search
            self.select_feat = selected_feat


    def tune(self,X,y):

        if self.method=="ml":
            estimator = str(self.best_pipeline['estimator'].__class__.__name__)
            total_features = self.best_estimator.n_features_in_

            optimization_grid = []


            # Linear regression
            if estimator=="LinearRegression":
                optimization_grid.append({
                    'estimator__fit_intercept': [True,False],
                    'estimator__positive':[True,False]
                })

            # Ridge
            elif estimator=="Ridge":
                optimization_grid.append({
                    'estimator__fit_intercept': [True,False],
                    'estimator__alpha': np.arange(0.1,2,0.1),
                    'estimator__max_iter': np.arange(1000,10000,1000)
                })
            
            # Lasso
            elif estimator=="Lasso":
                optimization_grid.append({
                    'estimator__fit_intercept': [True,False],
                    'estimator__alpha': np.arange(0.1,2,0.1),
                    'estimator__max_iter': np.arange(1000,10000,1000),
                    'estimator__positive':[True,False],
                    'estimator__selection': ['cyclic', 'random']
                })
            
            # LightGBM
            elif estimator=="LGBMRegressor":
                optimization_grid.append({
                    'estimator__n_estimators':np.arange(100,500,50),
                    'estimator__learning_rate':np.arange(0.1,1,0.1),
                })
            
            # K-nearest neighbors
            elif estimator=="KNeighborsRegressor":
                optimization_grid.append({
                    'estimator__n_neighbors':np.arange(2,50),
                    'estimator__weights':['uniform','distance']
                })

            # Random Forest
            elif estimator=="RandomForestRegressor":
                optimization_grid.append({
                    'estimator__n_estimators':np.arange(100,500,50),
                    'estimator__criterion': ['gini','entropy']               
                })

            # Extra Tree
            elif estimator=="ExtraTreesRegressor":
                optimization_grid.append({
                    'estimator__n_estimators':np.arange(100,500,50),
                    'estimator__criterion': ['gini','entropy']
                })


            # Gradient boosting
            elif estimator=="GradientBoostingRegressor":
                optimization_grid.append({
                    'estimator__n_estimators':np.arange(100,500,50),
                    'estimator__learning_rate':np.arange(0.1,1,0.1),
                    'estimator__loss': ['ls', 'lad', 'huber', 'quantile'],
                    'estimator__criterion': ['friedman_mse', 'mse', 'mae'],
                })

            # XGBoost
            elif estimator=="XGBRegressor":
                optimization_grid.append({
                    'estimator__n_estimators':np.arange(100,500,50),
                    'estimator__learning_rate':np.arange(0.1,1,0.1),
                })

            # Decision tree
            elif estimator=="DecisionTreeRegressor":
                optimization_grid.append({
                    'estimator__criterion': ['gini','entropy'],
                })

            # SVR
            elif estimator=="SVR":
                optimization_grid.append({
                    'estimator__C': np.arange(0.1,1,0.1),
                    'estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'estimator_gamma': ['scale', 'auto']
                })
            
            search = HalvingRandomSearchCV(
                    self.best_estimator,
                    optimization_grid,
                    # n_iter=self.n_iter,
                    factor=4,
                    scoring = self.scoring_function,
                    random_state = 0, 
                    n_jobs = 2,
                    verbose = 1,
                    cv = self.n_splits,
                    )


            search.fit(X, y)
            self.best_estimator = search.best_estimator_
            self.best_pipeline = search.best_params_


    def predict(self,X,y = None):
            return self.best_estimator.predict(X)
    
    def evaluate(self,X,y=None):
        loss, acc = self.best_search.evaluate(X, y, verbose=0)
        return loss, acc

    def training(model,X,y,n_splits,type='regressor',method='ml'):
        """Trains the best model for given train data and label
        in:  Train data, Test data and label
        out: Trained model object and prediction array on test set
        """
        gc.collect()
        if method=='dl':
            model.fit(X,y)
            X_trans = model.best_pipeline['preprocessor'].transform(X)
            X_new = pd.DataFrame(X_trans,columns=list(X.columns))
            X_select = X_new[model.select_feat]
            y_pred = model.predict(X_select)
        
        else:
            with st.spinner(f"Fitting and Comparing Different Models..."):
                model.fit(X,y)
            with st.spinner("Tuning Best Model..."):
                try:
                    model.tune(X,y)
                except Exception as e:
                    st.info("Hyper-Parameter Tuning Failed.")

            y_pred = model.predict(X)

        return model,y_pred

    def download_model(model,method,exp_name):
        """Generates a link allowing the trained model to be downloaded
        in:  model
        out: href string
        """

        #dumping ML model
        if method=="ml":
            model_name = "./Trained_Models/model.pkl"
            zip_path = "./Trained_Models/model.zip"
            output_model = zipfile.ZipFile(zip_path, mode='w', compression=zipfile.ZIP_DEFLATED)
            with open(model_name, 'wb') as f:
                pickle.dump(model, f,protocol=pickle.HIGHEST_PROTOCOL)
            output_model.write(model_name)
            output_model.close()
        
        else:
            #exporting tf model
            model_name = f"./Trained_Models/{exp_name}/best_model"
            zip_path = "./Trained_Models/model.zip"
            shutil.make_archive("model", "zip", model_name)
            output_model = zipfile.ZipFile(zip_path, mode='a', compression=zipfile.ZIP_DEFLATED)
            output_model.close()
        
        #creating download link
        with open(zip_path, "rb") as f:
            bytes = f.read()
        encoder = base64.b64encode(bytes).decode()

        button_uuid = str(uuid.uuid4()).replace("-", "")
        button_id = re.sub("\d+", "", button_uuid)

        custom_css = f""" 
            <style>
                #{button_id} {{
                    background-color: rgb(14, 17, 23);
                    color: rgb(255, 255, 255);
                    padding: 0.25em 0.38em;
                    position: relative;
                    text-decoration: none;
                    border-radius: 4px;
                    border-width: 1px;
                    border-style: solid;
                    border-color: rgb(230, 234, 241);
                    border-image: initial;
                }} 
                #{button_id}:hover {{
                    border-color: rgb(246, 51, 102);
                    color: rgb(246, 51, 102);
                }}
                #{button_id}:active {{
                    box-shadow: none;
                    background-color: rgb(246, 51, 102);
                    color: white;
                    }}
            </style> """

        model_file = (
            custom_css
            + f'<a download="model.zip" id="{button_id}" href="data:file/output_model;base64,{encoder}">Download Trained Model</a><br></br>'
        )
        
        return model_file

    def print_metrics(self,model,X,y,y_pred,method):
        metrics = pd.DataFrame()
        if method=='ml':
            best_model_name = type(model.best_estimator._final_estimator).__name__
            self.best_model_name = best_model_name
            cross_X = model.best_estimator.named_steps['preprocessor'].transform(X)
            cross_val_acc_score = f"{np.mean((cross_val_score(model.best_estimator._final_estimator,X=cross_X,y=y,n_jobs=4,cv=5,scoring=self.scoring_function))):.3f}"
            
            metrics['Best Model'] = [best_model_name]
            metrics['Negative Mean Squared Error (Mean Cross Val. Score)'] = [cross_val_acc_score]

        
        r2 = r2_score(y,y_pred)
        rmse = mean_squared_error(y,y_pred,squared=False)
        mae = mean_absolute_error(y,y_pred)
        mape = mean_absolute_percentage_error(y,y_pred)
        metrics['R2 Score'] = [f"{r2:.3f}"]
        metrics['Root Mean Squared Error'] = [f"{rmse:.3f}"]
        metrics['Mean Absolute Error'] = [f"{mae:.3f}"]
        metrics['Mean Absolute Percentage Error'] = [f"{mape:.3f}"]

        metrics['Best Features'] = [model.select_feat]

        st.subheader("Results:")    
        styler = metrics.style.hide_index()
        st.write(styler.to_html(), unsafe_allow_html=True)
        if method=='ml':
            st.markdown(f"<small>ⓘ Scoring Function: Negative Mean Squared Error ({self.scoring_function})  \nⓘ Cross-Validation Folds: 5  \nⓘ Feature Selection Criteria: ANOVA F-test</small></font>",unsafe_allow_html=True)
        else:
            st.markdown(f"<small>ⓘ Feature Selection Criteria: ANOVA F-test</small></font>",unsafe_allow_html=True)

        if r2<0.9:
            st.markdown("<small><b>Tip to Remember:</b> A Model is as good a    s the data.</small>",unsafe_allow_html=True)

        with st.expander("Show Model Pipeline"):
            st.markdown(f"<b>Model Parameters:</b>  \n{model.best_estimator._final_estimator.get_params()}",unsafe_allow_html=True)
            st.markdown(f"<b>Pipeline Steps:</b>  \n{model.best_estimator.named_steps}",unsafe_allow_html=True)


    def run_experiment(method,X,y,n_splits,exp_name):

        #splitting X,y into train,test splits
        # if len(np.unique(y))>1:
        #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
        # else:
        #     st.warning("Target Variable contains single type of class.")
        #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if method=='ml':
                
            #Training ML classifier
            start_time = time.time()
            clf = autoregression(n_splits=n_splits,exp_name=exp_name)
            with st.spinner('Training your model, Wait for it... (This may take time depending on dataset size)'):
                try:
                    model, y_pred = clf.training(X,y,type='regressor',method='ml',n_splits=n_splits)
                    clf.print_metrics(model,X,y,y_pred,method='ml')
                    end_time = time.time()-start_time
                    st.success(f"Training Completed! [Time taken: {int(end_time/60)} minutes, {int(end_time%60)} seconds]")
                except Exception as e:
                    st.error(e)
            
            
            # #display results
            # st.write("Sample Inference on Data")                
            # test = pd.DataFrame(X.iloc[0]).transpose()
            # st.write(test)
            # st.text(f"Predicted Label: {le.inverse_transform(model.best_estimator.predict(test))[0]}")
            # st.text(f"Actual Label: {le.inverse_transform([y[0]])[0]}")
            return model
        
        else:
            #training DL model
            start_time = time.time()
            clf = autoregression(method='dl',exp_name=exp_name)
            with st.spinner('Training your model, Wait for it... (This may take time depending on dataset size)'):
                try:
                    
                    model,y_pred = clf.training(X,y,type='regressor',method='dl',n_splits=n_splits)
                    end_time = time.time()-start_time
                    st.success(f"Training Completed! [Time taken: {int(end_time/60)} minutes, {int(end_time%60)} seconds]")
                    # X_train = X_train[model.select_feat]
                    # X_test = X_test[model.select_feat]


                except Exception as e:
                    st.error(e)

            #plotting metrics and curves
            clf.print_metrics(model,X,y,y_pred,method='dl')

            #displaying results
            # label_map = dict(zip(le.transform(le.classes_),le.classes_))

            st.write("Sample Inference on Data")
            X_trans = model.best_pipeline['preprocessor'].transform(pd.DataFrame(X.iloc[0]).transpose())
            st.write(pd.DataFrame(X.iloc[0]).transpose())        
            test = pd.DataFrame(X_trans,columns=list(X.columns))
            test_pred = model.best_estimator.predict(test[model.select_feat])
            if model.multiclass==True:
                test_pred = list(map(lambda x: np.argmax(x),test_pred[0]))[0]
                
            else:
                test_pred = np.where(test_pred <= 0.5, 0, 1)[0][0]

            st.text(f"Predicted Label: {test_pred}")
            st.text(f"Actual Label: {y[0]}")
            return model

# d = load_breast_cancer()
# y = d['target']
# X = pd.DataFrame(d['data'],columns = d['feature_names'])

# if len(np.unique(y))>1:
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
# else:
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # building and training model
# model = autoclassifier()
# model.fit(X_train,y_train)
# y_pred = model.predict(X_test)
# y_pred = (y_pred > 0.5)

# # results
# model.best_estimator.summary()
# print(f"\nTest Dataset Accuracy: {model.evaluate(X_test,y_test)[1]*100:.2f} %")
# print(f"\nBalanced Accuracy: {balanced_accuracy_score(y_test,y_pred)*100:.2f} %")
# print(f"\nConfusion Matrix:\n {confusion_matrix(y_test,y_pred)}")
# print(f"\nClassification Report:\n {classification_report(y_test,y_pred)}")

# # finalize model
# if args.method=='ml':
#     best_parameters = model.best_pipeline
#     model = best_parameters
#     model.fit(X,y)
#     print("Model fitted with best parameters")

# # saving model
# if args.savemodel == 'yes':
#     if args.method=="ml":
#         model_location = os.getcwd()+"\model.pkl"
#         with open(model_location, 'wb') as file:
#             pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
#             print(f"Model saved to {model_location}")
#     else:
#         if args.savetype=='keras':
#             model_location = os.getcwd()+"\model.h5"
#             model.best_estimator.save(model_location)
#             print(f"Model saved to {model_location}")
#         else:
#             model_location = os.getcwd()+"\model"
#             model.best_estimator.save(model_location)
#             print(f"Model saved to {model_location}")
