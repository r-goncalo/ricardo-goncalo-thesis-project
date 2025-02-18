from automl.component import InputSignature, Schema, requires_input_proccess
from automl.loggers.logger_component import LoggerSchema

from logger.Log import LogClass, openLog

import wandb

import torch

import pandas
from typing import Dict

import matplotlib.pyplot as plt


class ResultLogger(LoggerSchema):

    # TODO: empty parameters_signature should be able to be removed
    # TODO: verify order at which keys are verified
    parameters_signature = {
            "keys" : InputSignature(possible_types=[list]),
            "save_on_log" : InputSignature(default_value=True)
        } 
    
    # INITIALIZATION --------------------------------------------------------

    def proccess_input(self): #this is the best method to have initialization done right after
        
        super().proccess_input()
        
        self.columns = self.input["keys"]
                
        self.dataframe = pandas.DataFrame(columns=self.columns)
        
        self.save_on_log = self.input["save_on_log"]
                
        
    # USAGE -------------------------------------------------------------------
 
 
    @requires_input_proccess           
    def log_results(self, results : Dict[str, list]):
        
        results_df = pandas.DataFrame(results, columns=self.columns)
        
        self.dataframe = pandas.concat((self.dataframe, results_df), ignore_index=True) 
        
        if self.save_on_log:
            self.save_dataframe()
              
        
    
    @requires_input_proccess    
    def save_dataframe(self):
        
        self.lg.saveDataframe(self.dataframe, filename="results.csv")
        
    @requires_input_proccess
    def plot_graph(self, x_axis : str, y_axis : list, title : str = '', save_path: str = None):
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

        for y in y_axis:
            if y not in self.dataframe.columns:
                raise KeyError(f"Column '{y}' not found in dataframe.")

        #plt.figure(figsize=(10, 6))
        y_label = ''
        for y in y_axis:
            plt.plot(self.dataframe[x_axis], self.dataframe[y], marker='o', label=y)
            y_label += y + ' '

        plt.xlabel(x_axis)
        plt.ylabel(y_label)
        
        if title != '':
            plt.title(title)
                
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)

        plt.show()
        
    def get_last_results(self):
                        
        return self.dataframe.tail(1)
        
        
    
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