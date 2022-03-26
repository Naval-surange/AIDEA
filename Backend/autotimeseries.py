from pycaret.time_series import *
from pycaret.internal.pycaret_experiment import TimeSeriesExperiment
import os
import pandas as pd
import matplotlib.pyplot as plt
import gc
import streamlit as st


class autotimeseries:
    def __init__(self, n_fold=3, session_id=42, seasonal_period=None,n_select=1,data=None,fh = 12):
        self.n_fold = n_fold
        self.session_id = session_id
        self.seasonal_period = seasonal_period
        self.n_select = n_select
        self.data = data
        self.experiment = None
        self.tuned_model = None
        self.forecast = None
        self.fh = fh
        
    def setup_experiment(self):
        gc.collect()
        experiment = TimeSeriesExperiment()
        experiment = experiment.setup(data=self.data, seasonal_period=self.seasonal_period, fold=self.n_fold, session_id=self.session_id,n_jobs=4)
        
        with st.spinner("Fitting and Comparing Different Models..."):
            try:
                best_model = experiment.compare_models(n_select = self.n_select)
            except Exception as e:
                print(e)
            try:
                with st.spinner("Tuning Best model..."):
                    tuned_model = experiment.tune_model(best_model,fold=self.n_fold)
            except:
                tuned_model = best_model
        
        self.experiment = experiment
        self.tuned_model = tuned_model
        
        return self.tuned_model

    def forecast_plot(self,type):
        if type == "insample":
            self.experiment.plot_model(self.tuned_model, plot = 'insample', display_format='streamlit',
                                    return_fig=True, fig_kwargs = {'fig_template': 'ggplot2'})
        
        elif type == "forecast":
            self.experiment.plot_model(self.tuned_model, plot = 'forecast', data_kwargs = {'fh' : self.fh}, display_format='streamlit',
                                    return_fig=True, fig_kwargs = {'fig_template': 'ggplot2'})

        elif type == "diagnostics":
            self.experiment.plot_model(self.tuned_model, plot = 'diagnostics', display_format='streamlit',
                                    return_fig=True, fig_kwargs = {'fig_size' : [1000, 700],'fig_template': 'ggplot2'})

        elif type == "ts":
            self.experiment.plot_model(self.tuned_model, plot = 'ts', display_format='streamlit',
                                    return_fig=True, fig_kwargs = {'fig_template': 'ggplot2'})

        elif type == "residuals":
            self.experiment.plot_model(self.tuned_model, plot = 'residuals', display_format='streamlit',
                                    return_fig=True, fig_kwargs = {'fig_template': 'ggplot2'})

    def confidence_plot(self):
        start_index = 0
        end_index = len(self.data)-1
        confidence_interval = None
        try:
            confidence_interval = self.tuned_model.get_prediction(start_index,end_index).conf_int()
        except:
            confidence_interval = self.tuned_model.prediction_intervals()
        
        if confidence_interval!=None:
            lower_limit = confidence_interval.filter(regex='lower')
            upper_limit = confidence_interval.filter(regex='upper')

            #plot prediction
            ax, fig = plt.subplots(figsize =(16,12))
            ax.plot(self.data.index, self.data, label='observed')
            #plot your mean prediction
            ax.plot(self.forecast.index,self.forecast, color='r', label='forecast')
            # shade the area between your condifidence limits
            ax.fill_between(self.forecast.index, lower_limit, upper_limit, color='pink')
            #set labels, legends and show plot
            ax.legend()
            st.pyplot(fig)

    



# parser = argparse.ArgumentParser()
# parser.add_argument('--savemodel', choices=['yes', 'no'])
# parser.add_argument('--forecast', required=True)
# args = parser.parse_args()

# startTime = datetime.now()

# class autotimeseries:
#     def __init__(self, forecast_length=10, drop_data_older_than_periods=200,ensemble='simple',frequency='infer',
#                 date_col='Date',value_col='Value',max_generations=5,num_validations=2):
#         self.forecast_length = forecast_length
#         self.drop_data_older_than_periods = drop_data_older_than_periods
#         self.ensemble = ensemble
#         self.frequency = frequency
#         self.date_col = date_col
#         self.value_col = value_col
#         self.max_generations = max_generations
#         self.num_validations = num_validations

#     def fit(self,df,date_col,val_col,id_col=None):

#         if df[date_col].dtypes=='object':
#             df[date_col] = pd.to_datetime(df[date_col])

#         model = AutoTS(self.forecast_length,self.frequency,self.ensemble,
#                 self.drop_data_older_than_periods,self.max_generations,
#                 self.num_validations)

#         model.fit(df,date_col,val_col)

#         self.model = model


# d = pd.read_csv("./datasets/AAPL.csv")
# y = d['Close']
# X = d['Date']

# forecast_days = int(args.forecast)
# print(forecast_days)
# model = AutoTS(forecast_length=forecast_days, frequency='infer', 
#                ensemble='simple', drop_data_older_than_periods=200,
#                max_generations=10,num_validations=2)
    
# model.fit(d, date_col='Date', value_col='Close', id_col=None)

# y_pred = model.predict(forecast_days).forecast

# best_parameters = model.best_model_params

# print("\nActual Values:\n",d[['Date','Close']].tail(forecast_days))
# print("\nPrediction:\n",y_pred)
# print(f"\nBest Model: {best_parameters}")

# print("Time Taken:",datetime.now() - startTime)

# # saving model
# if args.savemodel == 'yes':
#     model_location = os.getcwd()+"\model.pkl"
#     with open(model_location, 'wb') as file:
#         pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
#         print(f"Model saved to {model_location}")