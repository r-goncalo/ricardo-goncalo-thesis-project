from ..component import InputSignature, Schema, requires_input_proccess
from ..logger_component import LoggerSchema

from logger.Log import LogClass, openLog

import wandb

import torch

import pandas

class ResultLogger(LoggerSchema):

    # TODO: empty parameters_signature should be able to be removed
    # TODO: verify order at which keys are verified
    parameters_signature = {
            "keys" : InputSignature(possible_types=[list])
        } 
    
    # INITIALIZATION --------------------------------------------------------

    def proccess_input(self): #this is the best method to have initialization done right after
        
        super().proccess_input()
        
        self.columns = self.input["keys"]
        
        self.dataframe = pandas.DataFrame(columns=self.columns)
        
    # USAGE -------------------------------------------------------------------
 
 
    @requires_input_proccess           
    def log_results(self, results):
        
        results_df = pandas.DataFrame(results, columns=self.columns)
        
        self.dataframe = pandas.concat((self.dataframe, results_df))        
    
    @requires_input_proccess    
    def save_dataframe(self):
        
        self.lg.saveDataframe(self.dataframe, filename="results.csv")
        
    
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