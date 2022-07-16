import pandas_profiling as pp
from streamlit_pandas_profiling import st_profile_report
import streamlit as st
import base64
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from Backend.autoregression import autoregression
from load_data import dataloader
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, fetch_california_housing
import streamlit.components.v1 as components
from explainerdashboard import RegressionExplainer, ExplainerDashboard


def app():
    # st.set_page_config(layout="wide",page_title='AIDEA', page_icon="💡")
    hide_streamlit_style = """
                    <style>
                    #MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    </style>
                    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    best_model_name = "Best Model"
    df = None
    test_df = None
    numerical_var = None
    cat_var = None

    col1, col2, col3 = st.columns((1,1,2))
    
    with col1:
        exp_name = st.sidebar.text_input("Experiment Name",value="AIDEA Experiment")
    
    with col2:
        input_data = st.sidebar.radio("Input Dataset",options=['Select Existing','Upload File','Input URL'])

    # with col4:
    #     monitor = st.selectbox("Monitor Model Training",["--Select--","Telegram","WhatsApp","Email"])

    with col3:
        if input_data=="Input URL":
            url = st.sidebar.text_input("Enter File URL",help="Enter absolute URL of dataset file. For e.g., https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
            if url!="":
                df = dataloader.load_data(method='url',file_name=url)

        elif input_data=="Select Existing":
            existing_data = st.sidebar.selectbox("Select from Existing Datasets:",options=["--Select--","Servo","Diabetes"])
            if existing_data=="Servo":
                df_colnames = ["motor", "screw", "pgain", "vgain", "class"]
                df = pd.read_csv("./datasets/servo.data",sep=",", names=df_colnames)

            elif existing_data=="Diabetes":
                df = load_diabetes(as_frame=True).frame


        elif input_data=="Upload File":
            uploaded_file = st.sidebar.file_uploader("Upload Train Dataset File (Required)",type=['csv','tsv','xls','xlsx'])
            uploaded_file2 = st.sidebar.file_uploader("Upload Test Dataset File (Optional)",type=['csv','tsv','xls','xlsx'])
            if uploaded_file is not None:
                df = dataloader.load_data(method='upload',file_name=uploaded_file)
            if uploaded_file2 is not None:
                test_df = dataloader.load_data(method='upload',file_name=uploaded_file2)

    def find_id_columns(data, target):
    # some times we have id column in the data set, we will try to find it and then will drop it if found
        len_samples = len(data)
        id_columns = []
        for i in data.select_dtypes(include=["object", "int64", "float64", "float32"]).columns:
            col = data[i]
            if i != target:
                if sum(col.isnull()) == 0:
                    try:
                        col = col.astype("int64")
                    except:
                        continue
                    if col.nunique() == len_samples:
                        # we extract column and sort it
                        features = col.sort_values()
                        # now we subtract i+1-th value from i-th (calculating increments)
                        increments = features.diff()[1:]
                        # if all increments are 1 (with float tolerance), then the column is ID column
                        if sum(np.abs(increments - 1) < 1e-7) == len_samples - 1:
                            id_columns.append(i)
        
        if len(id_columns)>1:
            return id_columns
        elif len(id_columns)==0:
            return None
        else:
            return id_columns[0]

    if df is not None:
        df.drop_duplicates(inplace=True)
        # df.dropna(axis=0,inplace=True)
        # df = df[~df.isin([np.inf, -np.inf]).any(1)]
        df.replace([np.inf, -np.inf], np.NaN, inplace=True)
        df.index.name = 'index'

        st.write("Dataset Sample:")
        st.write(df.head(2))
        st.text(f"Dataset Shape: {df.shape[0]} Rows, {df.shape[1]} columns")

        if st.button("View EDA Report"):
            with st.spinner("Creating EDA Report..."):
                report = pp.ProfileReport(df, title="EDA Report",explorative=True,dark_mode=False)
                with st.expander("Toggle EDA Visualization"):
                    st_profile_report(report,height=500)

        c1, c2, c3 = st.columns((1,1,2))
        with c1:
            method = st.selectbox("Select Mode",["Machine Learning","Deep Learning"])
        with c2:
            target_var = st.selectbox("Select Target Column",options=["--Select--"]+df.columns.values.tolist())
        # with c3:
        #     index_var = st.selectbox("Select ID/Index Column",options=["--Select--","--NA--"]+df.columns.values.tolist(),help="Select --NA-- if there is no separate ID/Index column")
        with c3:
            drop_var = st.multiselect("Select Irrelevant Columns to drop (if any)",options=["--NA--"]+df.columns.values.tolist())
        
        if target_var!="--Select--":

            index_var = find_id_columns(df,target_var)

            df = df.loc[:,~df.columns.duplicated()]

            for i in df.select_dtypes(include=["float64","float32"]).columns:
                count_float = np.nansum([False if r.is_integer() else True for r in df[i]])
                if (count_float == 0) & (df[i].nunique() <= 20):
                    df[i] = df[i].astype("object")

            numerical_var = df.select_dtypes(include=["int32", "int64", "float64", "float32"]).columns.values.tolist()
            
            if index_var in numerical_var:
                numerical_var.remove(index_var)
            if target_var in numerical_var:
                numerical_var.remove(target_var)
            for var in drop_var:
                if var in numerical_var:
                    numerical_var.remove(var)

            if len(numerical_var)>1:
                for i in numerical_var:
                    if df[i].dtypes == "float64":
                        df[i] = df[i].astype("float32")
            elif len(numerical_var)==1:
                if df[numerical_var].dtypes == "float64":
                        df[numerical_var] = df[numerical_var].astype("float32")

            if df[target_var].dtypes == "float64":
                df[target_var] = df[target_var].astype("float32")

            

            cat_var = df.select_dtypes(include=["object","category","bool"]).columns.values.tolist()
            df[cat_var] = df[cat_var].astype("category")

            if index_var in cat_var:
                cat_var.remove(index_var)
            if target_var in cat_var:
                cat_var.remove(target_var)
            for var in drop_var:
                if var in cat_var:
                    cat_var.remove(var)

            if cat_var==[]:
                cat_var = None

            if numerical_var==[]:
                numerical_var = None

            st.markdown(f"Inferred Columns:<small><ol> <li><b>ID:</b> {index_var}</li>  <li><b>Numerical:</b> {numerical_var}</li>  <li><b>Categorical:</b> {cat_var}</li>  <li><b>Target (Label):</b> {target_var}</li></ol></small>",unsafe_allow_html=True)
            
            df = shuffle(df)
            df.reset_index(inplace=True, drop=True)

            X = df.drop([target_var,index_var],axis=1,errors='ignore')
            X = X.drop(drop_var,axis=1,errors='ignore')

            if test_df is not None:
                test_X = test_df.drop(drop_var,axis=1,errors='ignore')
            
            y = df[target_var]

            if index_var!=None:
                exp_idx = index_var
            else:
                exp_idx = None
            
            n_splits = 5
            if st.button("Run Experiment"):
                if exp_name!="":
                    if method=="Machine Learning":
                        model = autoregression.run_experiment(X=X,y=y,method='ml',n_splits=n_splits,exp_name=exp_name)
                        
                        st.subheader("XAI Dashboard")
                        with st.spinner('Creating Explainability Dashboard...'):
                            explainer = RegressionExplainer(model, X, y, 
                                        # labels=list(df[target_var].unique()),
                                        index_name = exp_idx,
                                        target = target_var, # defaults to y.name
                                        n_jobs=4,
                                        precision='float32',
                                        # pos_label='1'
                                        )

                            db = ExplainerDashboard(explainer, 
                                title = exp_name, # defaults to "Model Explainer"
                                shap_interaction = False, # you can switch off tabs with bools
                                whatif = False,
                                contributions = False,
                                hide_poweredby = True
                                )

                            components.html(db.to_html(),scrolling=True,height=800)
                            st.text("\n")

                        with st.spinner('Exporting Model...'):
                            try:
                                model_file = autoregression.download_model(model,method="ml",exp_name=exp_name)
                                st.markdown(model_file, unsafe_allow_html=True)
                            except:
                                st.info("Please clear the cache, train model again and then try again to export the model.")
                            
                    else:
                        model = autoregression.run_experiment(X=X,y=y,method='dl',n_splits=n_splits,exp_name=exp_name)

                        with st.spinner("Exporting Model..."):
                            try:
                                model_file = autoregression.download_model(model,method="dl")
                                st.markdown(model_file, unsafe_allow_html=True)
                            except:
                                st.info("Please clear the cache, train model again and then try again to export the model.")
                else:
                    st.error("Experiment Name is Required.")

                if test_df is not None:
                    test_X = test_X.drop([index_var],axis=1)
                    test_pred = model.predict(test_X)
                    test_df[target_var] = test_pred
                    test_df = test_df[[index_var,'Prediction']]
                    csv_file = test_df.to_csv(index=False).encode()
                    b64 = base64.b64encode(csv_file).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="submission.csv" target="_blank">Download submission.csv File</a>'
                    st.markdown(href, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<small><center><i>Note: Trained Models should not be used in production using current version of AIDEA.</i></center></small>",unsafe_allow_html=True)
    st.markdown("<small><center><b>Made with ❤️ by Naval Surange</b></center></small>",unsafe_allow_html=True)