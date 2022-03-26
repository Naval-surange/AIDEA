import numpy as np
import pandas as pd
import autokeras as ak
import dill as pickle
import zipfile
import uuid
import gc
import re
import base64
import shutil
import time
import scikitplot as skplt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import cross_val_score, HalvingRandomSearchCV, HalvingGridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score
import lightgbm as lgb
import streamlit as st


# parser = argparse.ArgumentParser()
# parser.add_argument('--savemodel', choices=['yes', 'no'])
# parser.add_argument('--method', choices=['ml', 'dl'],required=True)
# parser.add_argument('--savetype',choices=['tf','keras'],default='keras')
# args = parser.parse_args()

           
class autoclassifier:
    def __init__(self, scoring_function = 'f1', n_iter = 100, savemodel='yes',method='ml',multiclass=False,
                savetype='tf',maxtrials=10,class_weight=None, n_splits = 5, exp_name = 'AIDEA'):
        self.scoring_function = scoring_function
        self.n_iter = n_iter
        self.savemodel = savemodel
        self.method = method
        self.savetype = savetype
        self.maxtrials = maxtrials
        self.class_weight = class_weight
        self.n_splits = n_splits
        self.class_imbalance = False
        self.best_model_name = "Best Model"
        self.exp_name = exp_name
        self.multiclass = multiclass

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
                        ('encoder',CatBoostEncoder(cols=cat_subset,random_state=101)),
                        ('cleaner',SimpleImputer(strategy = 'most_frequent'))
                        # ('encoder',OneHotEncoder(sparse = False, categories=categorical_values))
                    ])


        preprocessor = ColumnTransformer([
                        ('numerical', num_pipeline, make_column_selector(dtype_exclude=['object','category','bool'])),
                        ('categorical', cat_pipeline, make_column_selector(dtype_include=['object','category','bool']))
                        ])
        

        if self.method=="ml":
            multi_class = False
            if len(np.unique(y))>2:
                multi_class = True
                if self.class_imbalance==True:
                    self.scoring_function = 'f1_micro'
                else:
                    self.scoring_function = 'f1_macro'

            estimators = [
                LogisticRegression(class_weight="balanced",max_iter=1000,random_state=101,n_jobs = 4), 
                RandomForestClassifier(random_state=101,class_weight="balanced",n_jobs = 4),
                KNeighborsClassifier(n_neighbors=5, n_jobs = 4),
                SVC(kernel='linear',probability=True,random_state = 101,class_weight="balanced"), 
                GaussianNB(),
                # CatBoostClassifier(auto_class_weights='Balanced',random_state=101,iterations=500),
                XGBClassifier(random_state=101,use_label_encoder=False,n_jobs = 4),
                lgb.sklearn.LGBMClassifier(random_state=101,class_weight="balanced",n_jobs = 4),
                ExtraTreesClassifier(random_state=101,class_weight="balanced",n_jobs = 4),
                GradientBoostingClassifier(max_depth=20,random_state=101),
                DecisionTreeClassifier(random_state=101,class_weight="balanced",max_features='auto')
            ]


            model_pipeline_steps = []
            model_pipeline_steps.append(('preprocessor',preprocessor))
            model_pipeline_steps.append(('variance_threshold',VarianceThreshold(threshold=0)))
            
            # model_pipeline_steps.append(('feature_selector',RFE(estimator=ExtraTreesClassifier(random_state=101))))
            model_pipeline_steps.append(('feature_selector',SelectKBest(f_classif,k='all')))
            model_pipeline_steps.append(('estimator',LogisticRegression(class_weight="balanced",random_state=101)))
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
                    # n_iter=self.n_iter,
                    factor=4,
                    scoring = self.scoring_function, 
                    # random_state = 0, 
                    n_jobs = 2,
                    verbose = 1,
                    cv = self.n_splits,
                    )

            search.fit(X, y)
            time.sleep(1)
            self.best_estimator = search.best_estimator_
            self.best_pipeline = search.best_params_
            
            feat = search.best_estimator_.named_steps['feature_selector'].get_support()
            time.sleep(1)
            selected_feat = list(X.columns[feat])
            self.select_feat = selected_feat
            self.multiclass = multi_class
            
        elif self.method=="dl":
            multi_class = False
            if len(np.unique(y))>2:
                multi_class = True

            model_pipeline_steps = []
            model_pipeline_steps.append(('preprocessor',preprocessor))
            model_pipeline_steps.append(('feature_selector',SelectKBest(f_classif,k='all')))
            # model_pipeline_steps.append(('feature_selector',RFE(estimator=ExtraTreesClassifier(random_state=101))))
            model_pipeline_steps.append(('estimator',ExtraTreesClassifier(class_weight='balanced')))
            model_pipeline = Pipeline(model_pipeline_steps)

            try:
                total_features = preprocessor.fit_transform(X).shape[1]
            except:
                total_features = preprocessor.fit_transform(X,y).shape[1]


            optimization_grid = []

            optimization_grid.append({
                'preprocessor__numerical__scaler':[RobustScaler(),StandardScaler(),MinMaxScaler(),PowerTransformer()],
                'preprocessor__numerical__cleaner__strategy':['mean','median'],
                # 'feature_selector__n_features_to_select': list(np.arange(1,total_features+1,1)),
                'feature_selector__k': list(np.arange(1,total_features,1)) + ['all'],
                'estimator':[ExtraTreesClassifier(random_state=101,class_weight='balanced')],
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
            class_weight_dict = dict(enumerate(self.class_weight))
            X_train, X_test, y_train, y_test = train_test_split(X_select, y, test_size=0.2,stratify=y)

            search = ak.StructuredDataClassifier(max_trials=self.maxtrials,seed=0,project_name=self.exp_name)
            search.fit(X_train, y_train,class_weight=class_weight_dict,validation_data=(X_test,y_test))

            best_model = search.export_model()
            self.best_estimator = best_model
            self.best_pipeline = model_pipeline
            self.best_search = search
            self.multiclass = multi_class
            self.select_feat = selected_feat


    def tune(self,X,y):

        if self.method=="ml":
            estimator = str(self.best_pipeline['estimator'].__class__.__name__)
            total_features = self.best_estimator.n_features_in_

            optimization_grid = []


            # Logistic regression
            if estimator=="LogisticRegression":
                optimization_grid.append({
                    'estimator__C': np.arange(0.1,1,0.1),
                    'estimator__max_iter': np.arange(1000,10000,500),
                    'estimator__fit_intercept': [True,False]
                })
                # optimization_grid.append({
                #     'estimator__C': Real(0.1,1),
                #     'estimator__max_iter': Integer(100,20000),
                # })

            # Naive Bayes
            elif estimator=="GaussianNB":
                pass

            # LightGBM
            elif estimator=="LGBMClassifier":
                optimization_grid.append({
                    'estimator__n_estimators':np.arange(100,500,50),
                    'estimator__learning_rate':np.arange(0.1,1,0.1),
                })
                
            
            # K-nearest neighbors
            elif estimator=="KNeighborsClassifier":
                max_class = np.max(np.bincount(y))
                optimization_grid.append({
                    'n_neighbors':np.arange(1,max_class,2),
                    'estimator__weights':['uniform','distance']
                })
                # optimization_grid.append({
                #     'estimator__weights':Categorical(categories=['uniform','distance']),
                #     'estimator__n_neighbors':Integer(1,50)
                # })

            # Random Forest
            elif estimator=="RandomForestClassifier":
                optimization_grid.append({
                    'estimator__n_estimators':np.arange(100,500,50),
                    'estimator__criterion': ['gini','entropy']               
                })

            # Extra Tree
            elif estimator=="ExtraTreesClassifier":
                optimization_grid.append({
                    'estimator__n_estimators':np.arange(100,500,50),
                    'estimator__criterion': ['gini','entropy']
                })

            # Gradient boosting
            elif estimator=="GradientBoostingClassifier":
                optimization_grid.append({
                    'estimator__n_estimators':np.arange(100,500,50),
                    'estimator__learning_rate':np.arange(0.1,1,0.1),
                    'estimator__loss': ['deviance','exponential'],
                    'estimator__criterion': ['friedman_mse', 'mse', 'mae'],
                })

            # XGBoost
            elif estimator=="XGBClassifier":
                optimization_grid.append({
                    'estimator__n_estimators':np.arange(100,500,50),
                    'estimator__learning_rate':np.arange(0.1,1,0.1),
                })

            # Decision tree
            elif estimator=="DecisionTreeClassifier":
                optimization_grid.append({
                    'estimator__criterion': ['gini','entropy'],
                })

            # Linear SVM
            elif estimator=="SVC":
                optimization_grid.append({
                    'estimator__C': np.arange(0.1,1,0.1),
                    'estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
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

    def predict_proba(self,X,y = None):
        return self.best_estimator.predict_proba(X)
    
    def evaluate(self,X,y=None):
        loss, acc = self.best_search.evaluate(X, y, verbose=0)
        return loss, acc


    def dl_proba(self,y_proba,multi_class=False):
        if multi_class==False:
            new_proba = []
            for prob in y_proba:
                if prob[0]>0.5:
                    new_proba.append([prob[0],1-prob[0]])
                else:
                    new_proba.append([1-prob[0],prob[0]])
            return new_proba
        else:
            return y_proba

    def training(model,X,y,n_splits,type='classifier',method='ml'):
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
            y_proba = model.predict(X_select)
        
        else:
            with st.spinner(f"Fitting and Comparing Different Models..."):
                model.fit(X,y)
            with st.spinner("Tuning Best Model..."):
                try:
                    model.tune(X,y)
                except Exception as e:
                    st.info("Hyper-Parameter Tuning Failed.")

            y_proba = model.predict_proba(X)
            y_pred = model.predict(X)

        return model,y_pred,y_proba

    def download_model(model,decoder,method):
        """Generates a link allowing the trained model to be downloaded
        in:  model
        out: href string
        """
        decoder_name = "./Trained_Models/decoder.pkl"
        #dumping label decoder
        with open(decoder_name, 'wb') as f:
            pickle.dump(decoder, f,protocol=pickle.HIGHEST_PROTOCOL)

        #dumping ML model
        if method=="ml":
            model_name = "./Trained_Models/model.pkl"
            zip_path = "./Trained_Models/model.zip"
            output_model = zipfile.ZipFile(zip_path, mode='w', compression=zipfile.ZIP_DEFLATED)
            with open(model_name, 'wb') as f:
                pickle.dump(model, f,protocol=pickle.HIGHEST_PROTOCOL)
            output_model.write(model_name)
            output_model.write(decoder_name)
            output_model.close()
        
        else:
            #exporting tf model
            model_name = "./Trained_Models/structured_data_classifier/best_model"
            zip_path = "./Trained_Models/model.zip"
            shutil.make_archive("model", "zip", model_name)
            output_model = zipfile.ZipFile(zip_path, mode='a', compression=zipfile.ZIP_DEFLATED)
            output_model.write(decoder_name)
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
            # st.markdown(f"Best Model: <font color='gold'>{best_model_name}</font>",unsafe_allow_html=True)
            cross_X = model.best_estimator.named_steps['preprocessor'].transform(X)
            # st.markdown(f"Mean Cross Validation Accuracy: <font color=#32CD32>{round(np.mean((cross_val_score(model.best_estimator._final_estimator,X=cross_X,y=y,n_jobs=-1,cv=5,scoring=self.scoring_function)))*100,2)}</font> %<font color='grey'><small>  \nⓘ Scoring Function: Balanced Accuracy</small></font>",unsafe_allow_html=True)
            cross_val_acc_score = round(np.mean((cross_val_score(model.best_estimator._final_estimator,X=cross_X,y=y,n_jobs = 4,cv=5,scoring=self.scoring_function)))*100,2)
            metrics['Best Model'] = [best_model_name]
            metrics['F1 Score (Mean Cross Val. Score)'] = [f"{cross_val_acc_score:.2f} %"]
        
        acc = round(accuracy_score(y,y_pred)*100,2)
        bal_acc = round(balanced_accuracy_score(y,y_pred)*100,2)
        if self.multiclass==True:
            roc_auc = round(roc_auc_score(y,model.predict_proba(X),multi_class='ovr')*100,2)
        else:
            roc_auc = round(roc_auc_score(y,model.predict_proba(X)[:, 1])*100,2)
        metrics['Accuracy'] = [f"{acc:.2f} %"]
        metrics['Balanced Accuracy'] = [f"{bal_acc:.2f} %"]
        metrics['ROC AUC'] = [f"{roc_auc:.2f} %"]

        metrics['Best Features'] = [model.select_feat]

        st.subheader("Results:")
        styler = metrics.style.hide_index()
        st.write(styler.to_html(), unsafe_allow_html=True)
        if method=='ml':
            st.markdown(f"<small>ⓘ Scoring Function: F1 Score ({self.scoring_function})  \nⓘ Cross-Validation Folds: 5  \nⓘ Feature Selection Criteria: ANOVA F-test</small></font>",unsafe_allow_html=True)
        else:
            st.markdown(f"<small>ⓘ Feature Selection Criteria: ANOVA F-test</small></font>",unsafe_allow_html=True)

        if abs(bal_acc-acc)>5:
            with st.expander("Why significant difference between balanced accuracy and regular accuracy?"):
                st.markdown('''Because of class imbalance model might get overfitted towards majority class.
                            But here class weight balancing is already applied so that model won't miss out on minority class (See ROC & PRC Curve in XAI Dashboard).
                              \nProblem with accuracy:
                            It hides the detail you need to better understand the performance of your classification model.
                            You can go through the below examples that will help you understand the problem:
                              \n1. Multi-class target variable: When your data has 3 or more classes you may get a higher classification
                              accuracy, but you don't know if that is because all classes are being predicted equally well or whether
                              one or two classes are being neglected by the model.
                              \n2. Imbalanced dataset: When you have imbalanced data (does not have an even number of classes).
                              You may achieve higher accuracy (let's say 95% or more) but is not a good score if 95 records for
                              every 100 belong to one class as you can achieve this score by always predicting the most common class value.
                                \n\n<a href='https://www.analyticsvidhya.com/blog/2020/12/accuracy-and-its-shortcomings-precision-recall-to-the-rescue/'>Source</a>''',
                              unsafe_allow_html=True)
        
        if int(cross_val_acc_score)<90:
            st.markdown("<small><b>Tip to Remember:</b> A Model is as good as the data.</small>",unsafe_allow_html=True)

        with st.expander("Show Model Pipeline"):
            st.markdown(f"<b>Model Parameters:</b>  \n{model.best_estimator._final_estimator.get_params()}",unsafe_allow_html=True)
            st.markdown(f"<b>Pipeline Steps:</b>  \n{model.best_estimator.named_steps}",unsafe_allow_html=True)

        st.subheader("Confusion Matrix:")
        st.text(confusion_matrix(y,y_pred))
        st.subheader("Classification Report:")
        st.write(pd.DataFrame(classification_report(y,y_pred,output_dict=True)).transpose())

    def plot_curves(self,y,y_proba,class_imbalance):
        if class_imbalance==False:
            with st.expander("Show ROC Curve (Focus on macro average, since data is balanced)"):
                fig = skplt.metrics.plot_roc(y, y_proba, title=self.best_model_name)
                st.pyplot(fig.figure)
        else:
            with st.expander("Show Precision Recall Curve (Focus on micro average, since data is imbalanced)"):
                fig = skplt.metrics.plot_precision_recall(y, y_proba, title=self.best_model_name)
                st.pyplot(fig.figure)


    def run_experiment(method,X,y,le,class_imbalance,class_weight,n_splits,exp_name):

        #splitting X,y into train,test splits
        # if len(np.unique(y))>1:
        #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
        # else:
        #     st.warning("Target Variable contains single type of class.")
        #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if method=='ml':
                
            #Training ML classifier
            start_time = time.time()
            clf = autoclassifier(n_splits=n_splits,exp_name=exp_name)
            with st.spinner('Training your model, Wait for it... (This may take time depending on dataset size)'):
                try:
                    model, y_pred, y_proba = clf.training(X,y,type="classifier",method='ml',n_splits=n_splits)
                    clf.print_metrics(model,X,y,y_pred,method='ml')
                    end_time = time.time()-start_time
                    st.success(f"Training Completed! [Time taken: {int(end_time/60)} minutes, {int(end_time%60)} seconds]")
                except Exception as e:
                    st.exception(e)

            return model
        
        else:
            #training DL model
            start_time = time.time()
            clf = autoclassifier(class_weight=class_weight,method='dl',exp_name=exp_name)
            with st.spinner('Training your model, Wait for it... (This may take time depending on dataset size)'):
                try:
                    
                    model,y_pred,y_proba = clf.training(X,y,type="classifier",method='dl',n_splits=n_splits)
                    end_time = time.time()-start_time
                    st.success(f"Training Completed! [Time taken: {int(end_time/60)} minutes, {int(end_time%60)} seconds]")

                    if model.multiclass==True:
                        y_proba = clf.dl_proba(y_pred,multi_class=True)
                        y_pred = list(map(lambda x: int(np.argmax(x)),y_pred))
                    else:
                        y_proba = clf.dl_proba(y_pred,multi_class=False)
                        y_pred = np.where(y_pred <= 0.5, 0, 1)

                except Exception as e:
                    st.exception(e)

            #plotting metrics and curves
            clf.print_metrics(model,X,y,y_pred,method='dl')
            clf.plot_curves(y,y_proba,class_imbalance=class_imbalance)

            st.write("Sample Inference on Data")
            X_trans = model.best_pipeline['preprocessor'].transform(pd.DataFrame(X.iloc[0]).transpose())
            st.write(pd.DataFrame(X.iloc[0]).transpose())        
            test = pd.DataFrame(X_trans,columns=list(X.columns))
            test_pred = model.best_estimator.predict(test[model.select_feat])
            if model.multiclass==True:
                test_pred = list(map(lambda x: np.argmax(x),test_pred[0]))[0]
                
            else:
                test_pred = np.where(test_pred <= 0.5, 0, 1)[0][0]

            st.text(f"Predicted Label: {le.inverse_transform([test_pred])[0]}")
            st.text(f"Actual Label: {le.inverse_transform([y[0]])[0]}")
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
