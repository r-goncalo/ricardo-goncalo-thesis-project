from automl.component import InputSignature, Component, requires_input_proccess
from automl.loggers.logger_component import LoggerSchema

from logger.Log import LogClass, openLog

import wandb

import torch

import pandas
from typing import Dict

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import numpy as np

RESULTS_FILENAME = 'results.csv'

class ResultLogger(LoggerSchema):

    # TODO: empty parameters_signature should be able to be removed
    # TODO: verify order at which keys are verified
    parameters_signature = {
            "filename" : InputSignature(default_value=RESULTS_FILENAME),
            "keys" : InputSignature(possible_types=[list], mandatory=False),
            "save_on_log" : InputSignature(default_value=True)
        } 
    
    # INITIALIZATION --------------------------------------------------------

    def proccess_input(self): #this is the best method to have initialization done right after
        
        super().proccess_input()
        
        self.filename = self.input["filename"]
        
        self.initialize_dataframe()
        
        self.save_on_log = self.input["save_on_log"]
    
    
    
    def initialize_dataframe(self):
        
        try:
        
            dataframe_on_folder = self.lg.loadDataframe(filename=self.filename)

            self.dataframe = dataframe_on_folder
            self.columns = self.dataframe.columns
            self.lg.writeLine(f"Results dataframe with filename {self.filename} already existed with columns {self.columns}")
        
        except:
            
            self.columns = self.input["keys"]
            self.dataframe = pandas.DataFrame(columns=self.columns)
            self.save_dataframe()
        


                
        
    # USAGE -------------------------------------------------------------------
 
 
    @requires_input_proccess           
    def log_results(self, results : Dict[str, list]):
        
        results_df = pandas.DataFrame(results, columns=self.columns)
        
        self.dataframe = pandas.concat((self.dataframe, results_df), ignore_index=True) 
        
        if self.save_on_log:
            self.save_dataframe()
              
        
    
    @requires_input_proccess    
    def save_dataframe(self):
        
        self.lg.saveDataframe(self.dataframe, filename=self.filename)
        

    # GRAPHS ------------------------------------------------------------------------------------------------------------------

    @requires_input_proccess
    def plot_bar_graph(self, x_axis : str, y_axis : list, title : str = '', save_path: str = None, to_show=True, y_label=''):
        """
        Plots a graph using the dataframe stored in ResultLogger.

        :param x_axis: The column key for the X-axis.
        :param y_axis: A list of column keys for the Y-axis.
        :param save_path: Optional path to save the plot as an image.
        """
        if self.dataframe.empty:
            raise ValueError("Dataframe is empty. Log results before plotting.")

        if x_axis not in self.dataframe.columns:
            raise KeyError(f"Column '{x_axis}' not found in dataframe. Available columns: " + str(self.dataframe.columns))


        if isinstance(y_axis, str):
            y_axis = [y_axis]
            
        for i in range(len(y_axis)):

            if y_axis[i] not in self.dataframe.columns:
                raise KeyError(f"Column '{y_axis[i]}' not found in dataframe.")
            
        categories = [str(value) for value in self.dataframe[x_axis]]


        for column_name in y_axis:
            plt.bar(self.dataframe[x_axis], self.dataframe[column_name])

        plt.xlabel(x_axis)
        plt.ylabel(y_label)
        
        if title != '':
            plt.title(title)
                
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(self.lg.logDir + '\\' + save_path)

        if to_show:
            plt.show()


    @requires_input_proccess
    def plot_graph(self, x_axis : str, y_axis : list, title : str = '', save_path: str = None, to_show=True, y_label=''):
        """
        Plots a graph using the dataframe stored in ResultLogger.

        :param x_axis: The column key for the X-axis.
        :param y_axis: A list of column keys for the Y-axis.
        :param save_path: Optional path to save the plot as an image.
        """
        if self.dataframe.empty:
            raise ValueError("Dataframe is empty. Log results before plotting.")

        if x_axis not in self.dataframe.columns:
            raise KeyError(f"Column '{x_axis}' not found in dataframe. Available columns: " + str(self.dataframe.columns))


        if isinstance(y_axis, str):
            y_axis = [y_axis]
            
        for i in range(len(y_axis)):
            
            if isinstance(y_axis[i], str):
                y_axis[i] = (y_axis[i], y_axis[i])

            if y_axis[i][0] not in self.dataframe.columns:
                raise KeyError(f"Column '{y_axis[i][0] }' not found in dataframe.")


        #plt.figure(figsize=(10, 6))
        for (column_name, name_to_plot) in y_axis:
            plt.plot(self.dataframe[x_axis], self.dataframe[column_name], marker='o', label=name_to_plot)

        plt.xlabel(x_axis)
        plt.ylabel(y_label)
        
        if title != '':
            plt.title(title)
                
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(self.lg.logDir + '\\' + save_path)

        if to_show:
            plt.show()
            

    def plot_confidence_interval(self, x_axis : str, y_column : str, y_values_label : str = 'mean', show_std=True, aggregate_number = 5, title : str = '', save_path: str = None, to_show=True, y_label=''):

        if self.dataframe.empty:
            raise ValueError("Dataframe is empty. Log results before plotting.")

        if x_axis not in self.dataframe.columns:
            raise KeyError(f"Column '{x_axis}' not found in dataframe. Available columns: " + str(self.dataframe.columns))

        if y_column not in self.dataframe.columns:
            raise KeyError(f"Column '{y_column}' not found in dataframe. Available columns: " + str(self.dataframe.columns))

        values = self.dataframe[y_column]

        aggregated_values = [ [ values[u - aggregate_number + i] for i in range(0, aggregate_number * 2) ] for u in range(aggregate_number, len(values) - aggregate_number) ]
        x_values = self.dataframe[x_axis][aggregate_number:(len(self.dataframe[x_axis]) - aggregate_number)]

        print(f"aggregated values: {aggregated_values}")

        mean_values = np.mean(aggregated_values, axis=1)
        
        if show_std:
            std_values = np.std(aggregated_values, axis=1)

        plt.plot(x_values, mean_values, label=y_values_label)
        
        if show_std:
            plt.fill_between(x_values, mean_values - std_values, mean_values + std_values, alpha=0.3)

        plt.xlabel(x_axis)
        plt.ylabel(y_label)
        
        if title != '':
            plt.title(title)
                
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(self.lg.logDir + '\\' + save_path)

        if to_show:
            plt.show()

    
    requires_input_proccess
    def plot_linear_regression(self, x_axis: str, y_axis: list, title: str = '', save_path: str = None, to_show=True, y_label=''):
       """
       Plots a graph with linear regression lines for the given columns in the dataframe.

       :param x_axis: The column key for the X-axis.
       :param y_axis: A list of column keys for the Y-axis.
       :param title: Optional title for the plot.
       :param save_path: Optional path to save the plot as an image.
       :param to_show: Whether to display the plot.
       :param y_label: Label for the Y-axis.
       """
       if self.dataframe.empty:
           raise ValueError("Dataframe is empty. Log results before plotting.")

       if x_axis not in self.dataframe.columns:
           raise KeyError(f"Column '{x_axis}' not found in dataframe. Available columns: " + str(self.dataframe.columns))

       if isinstance(y_axis, str):
           y_axis = [y_axis]
               
       for i in range(len(y_axis)):
           if isinstance(y_axis[i], str):
               y_axis[i] = (y_axis[i], y_axis[i])

           if y_axis[i][0] not in self.dataframe.columns:
               raise KeyError(f"Column '{y_axis[i][0]}' not found in dataframe.")
       
       for (column_name, name_to_plot) in y_axis:
           # Extract X and Y data
           X = self.dataframe[[x_axis]].values  # X values (reshaped for regression)
           Y = self.dataframe[column_name].values  # Y values

           # Fit the linear regression model
           model = LinearRegression()
           model.fit(X, Y)

           # Predict Y values from the linear model
           Y_pred = model.predict(X)

           # Plot the regression line
           plt.plot(self.dataframe[x_axis], Y_pred, label=f'{name_to_plot} Regression Line')

       plt.xlabel(x_axis)
       plt.ylabel(y_label)

       if title != '':
           plt.title(title)

       plt.legend()
       plt.grid(True)

       if save_path:
           plt.savefig(self.lg.logDir + '\\' + save_path)

       if to_show:
           plt.show()



    # RETURN RESULTS ------------------------------------------------------------------------------------------------
    
        
    def get_last_results(self):
                        
        return self.dataframe.tail(1).to_dict(orient="records")[0]
    
    
    def get_n_last_results(self, n_results):
        
        return self.dataframe.tail(n_results).to_dict(orient="records")[0]
    

    def get_avg_n_last_results(self, n_results, column):
        
        return self.dataframe[column].tail(n_results).mean()
    
    
    def get_std_n_last_results(self, n_results, column):
        
        return self.dataframe[column].tail(n_results).std()
    
    
    def get_avg_and_std_n_last_results(self, n_results, column):
        
        return self.get_avg_n_last_results(n_results, column), self.get_std_n_last_results(n_results, column)
    
    def get_sorted_dataframe(self, column, ascending):
        
        return self.dataframe.sort_values(by=column, ascending=ascending)
    
    def get_n_last_ordered_results(self, n, column, ascending = True):
        
        '''Returns the best n results'''
        
        ordered_dataframe = self.get_sorted_dataframe(column=column, ascending=ascending)
        
        return ordered_dataframe.tail(n).to_records()
        
        
        
        
        
    
    ## WANDB --------------------------------
    #
    #def initialize_wandb(self):
    #    
    #    self.lg.writeLine("Initializing wandb...")
    #    
    #    self.wandb_run = wandb.init(project="rl_pipeline", entity="rl_pipeline", mode="offline", dir = self.lg.logDir)
    #    
    #def log_to_wandb(self, toLog : dict):
    #    
    #    self.wandb_run.log(toLog)
    #    
    #def close_wandb(self):
    #
    #    self.lg.writeLine("Closing wandb...")        
    #    self.wandb_run.finish()    