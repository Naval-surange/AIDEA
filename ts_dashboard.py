import streamlit.components.v1 as components
import streamlit as st
import numpy as np
import time
import pandas as pd
from autotimeseries import autotimeseries
from load_data import dataloader

def app():
    df = None
    test_df = None

    hide_streamlit_style = """
                    <style>
                    footer {visibility: hidden;}
                    </style>
                    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


    freq_map = {"Day":'D',"Month":'M',"Year":'Y',"Hour":'H'}
    plot_map = {"Time Series":"ts","Diagnostics":"diagnostics","Residuals":"residuals","Insample":"insample","Forecast":"forecast"}
    season_period_map = {'--Select--':None,'Hourly':'H','Daily':'D','Weekly':'W','Monthly':'M','Quarterly':'Q','Yearly':'Y'}

    
    col1, col2, col3 = st.columns((1,1,2))
    
    with col1:
        exp_name = st.sidebar.text_input("Experiment Name",value="AIDEA Experiment")
    
    with col2:
        input_data = st.sidebar.radio("Input Dataset",options=['Select Existing','Upload File','Input URL'])

    with col3:
        if input_data=="Input URL":
            url = st.sidebar.text_input("Enter File URL",help="Enter absolute URL of dataset file. For e.g., https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
            if url!="":
                df = dataloader.load_data(method='url',file_name=url)

        elif input_data=="Select Existing":
            existing_data = st.sidebar.selectbox("Select from Existing Datasets:",options=["--Select--","Temperature"])
            if existing_data=="Temperature":
                df = pd.read_csv("./datasets/temp.csv")


        elif input_data=="Upload File":
            uploaded_file = st.sidebar.file_uploader("Upload Train Dataset File (Required)",type=['csv','tsv','xls','xlsx'])
            uploaded_file2 = st.sidebar.file_uploader("Upload Test Dataset File (Optional)",type=['csv','tsv','xls','xlsx'])
            if uploaded_file is not None:
                df = dataloader.load_data(method='upload',file_name=uploaded_file)
            if uploaded_file2 is not None:
                test_df = dataloader.load_data(method='upload',file_name=uploaded_file2)

    if df is not None:
        df.drop_duplicates(inplace=True)
        df.dropna(axis=0,inplace=True)

        c1, c2, c3, c4, c5 = st.columns((1,1,1,1,2))
        with c1:
            date_col = st.selectbox("Select Date Column",options=["--Select--"]+df.columns.values.tolist())
        with c2:
            target_var = st.selectbox("Select Target Column",options=["--Select--"]+df.columns.values.tolist())
        with c3:
            season_period = st.selectbox("Select Seasonal Period",options=['--Select--','Hourly','Daily','Weekly','Monthly','Quarterly','Yearly'])
        with c4:
            freq = st.selectbox("Select Frequency of Dataset",["Day","Month","Year","Hour"])
        with c5:
            drop_var = st.multiselect("Select Irrelevant Columns to drop (if any)",options=["--NA--"]+df.columns.values.tolist())

        st.dataframe(df.head(2))
        st.text(f"Dataset Shape: {df.shape[0]} Rows, {df.shape[1]} columns")
        
        if target_var!="--Select--" and season_period!='--Select--':
            try:
                df[date_col] = pd.to_datetime(df[date_col])
            except:
                df[date_col] = df[date_col].apply(lambda x: pd.Period(x, freq='ms'))
            df.rename(columns={date_col: 'date'}, inplace=True)
            df.set_index("date", inplace = True)
            df.sort_values(by='date',inplace=True)
            df = df.asfreq(freq_map[freq])

            # X = df.drop([target_var,index_var],axis=1,errors='ignore')
            X = df.drop(drop_var,axis=1,errors='ignore')
            if test_df is not None:
                test_X = test_df.drop(drop_var,axis=1,errors='ignore')
            
            y = df[target_var]
            if st.button("Run Experiment"):
                if exp_name!="":
                    start_time = time.time()
                    model = autotimeseries(data=X,seasonal_period=season_period_map[season_period])
                    with st.spinner("Training your model, Wait for it... (This may take time depending on dataset size)"):
                        tuned_model = model.setup_experiment()
                        end_time = time.time()-start_time
                    st.success(f"Training Completed! [Time taken: {int(end_time/60)} minutes, {int(end_time%60)} seconds]")
                    st.write("Model Summary:") 
                    try:
                        st.write(tuned_model.summary())
                    except:
                        st.write(tuned_model)

                    # st.text("\n")
                    # st.text(model.experiment.predict_model(tuned_model))

                    with st.expander("Show Plots"):
                        for plot in ["Time Series","Diagnostics","Residuals","Insample","Forecast"]:
                            try:
                                model.forecast_plot(type=plot_map[plot])
                            except Exception as e:
                                st.error(e)

    st.markdown("---")
    st.markdown("<small><center><i>Note: Trained Models should not be used in production using current version of AIDEA.</i></center></small>",unsafe_allow_html=True)
    st.markdown("<small><center><b>Made with ❤ by <a href='https://www.linkedin.com/in/aaryanverma' target='_blank'>Aaryan Verma</a></b></center></small>",unsafe_allow_html=True)
                
                    
