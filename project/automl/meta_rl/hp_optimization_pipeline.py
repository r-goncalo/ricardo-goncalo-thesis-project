from automl.component import InputSignature, Schema, requires_input_proccess
from automl.loggers.logger_component import LoggerSchema
from automl.loggers.result_logger import ResultLogger
from automl.rl.rl_pipeline import RLPipelineComponent

from automl.utils.json_component_utils import component_from_dict,  dict_from_json_string

import optuna

from automl.meta_rl.hyperparameter_suggestion import HyperparameterSuggestion

from automl.utils.random_utils import generate_and_setup_a_seed

class HyperparameterOptimizationPipeline(LoggerSchema):
    
    parameters_signature = {
        
                        "sampler" : InputSignature(default_value="TreeParzen"),
                        "seed" : InputSignature(generator= lambda self : generate_and_setup_a_seed()),
        
                        "configuration_dict" : InputSignature(mandatory=False),
                        "configuration_string" : InputSignature(mandatory=False),
                        "base_component_configuration_path" : InputSignature(mandatory=False),

                        "database_study_name" : InputSignature(default_value='experiment'),
                        
                        "direction" : InputSignature(default_value='maximize'),
                                                
                        "hyperparameters_range_list" : InputSignature(),
                        "n_trials" : InputSignature(),
                        
                        "steps" : InputSignature(default_value=1),
                        "pruner" : InputSignature(mandatory=False)
                                                    
                       }
    

    # INITIALIZATION -----------------------------------------------------------------------------

    def proccess_input(self): # this is the best method to have initialization done right after
        
        super().proccess_input()
        
        self.initialize_config_dict()
        self.initialize_sampler()
        self.initialize_database()
        self.initialize_pruning_strategy()
        
        self.n_steps = self.input["steps"]
        
        self.hyperparameters_range_list : list[HyperparameterSuggestion] = self.input["hyperparameters_range_list"]
        
        parameter_names = [hyperparameter_specification.name for hyperparameter_specification in self.hyperparameters_range_list]
        
        self.lg.writeLine(f"Hyperparameter names: {parameter_names}")

        self.results_logger : ResultLogger = self.initialize_child_component(ResultLogger, { "logger_object" : self.lg,
            "keys" : ['experiment', *parameter_names, "result"]})
        
        self.suggested_values = { parameter_name : 0 for parameter_name in parameter_names}
        
        self.n_trials = self.input["n_trials"]
        
        self.tried_configurations = 0
        
        
        
    def initialize_config_dict(self):
        
        if "base_component_configuration_path" in self.input.keys():
            self.load_configuration_str_from_path()
  
        elif "configuration_dict" in self.input.keys():
            self.config_dict = self.input["configuration_dict"]
            
        elif "configuration_string" in self.input.keys():
            self.config_dict = dict_from_json_string(self.input["configuration_string"])
            
        else:
            raise Exception("No configuration defined")
    
    def initialize_database(self):
        self.database_path = self.lg.logDir + "\\study_results.db"  # Path to the SQLite database file
        self.storage = f"sqlite:///{self.database_path}"  # This will save the study results to the file
    
    
    def initialize_sampler(self):
        
        if self.input["sampler"] == "TreeParzen":
            
            self.sampler : optuna.samplers.BaseSampler = optuna.samplers.TPESampler(seed=self.input["seed"])
        
        else:
            raise NotImplementedError(f"Did not implement for sampler '{self.input['sampler']}'") 
        
    def initialize_pruning_strategy(self):
        
        if 'pruner' in self.input.keys():
                        
            if isinstance(self.input["pruner"], optuna.pruners.BasePruner):
            
                self.pruning_strategy = self.input["pruner"]
            
            
            elif self.input["pruner"] == "Median":
            
                self.pruning_strategy = optuna.pruners.MedianPruner()
        
            else:
                raise NotImplementedError(f"Pruner '{self.input['pruner']}' is not implemented")
        
        else:
            self.lg.writeLine("Won't use prunning strategy")
        
    def load_configuration_str_from_path(self):
        
        self.rl_pipeline_config_path : str = self.input["base_component_configuration_path"]
        
        fd = open(self.rl_pipeline_config_path, 'r') 
        self.config_str = fd.read()
        fd.close()
        

    # OPTIMIZATION -------------------------------------------------------------------------

    def create_component_to_test(self):
        
        self.lg.writeLine("Creating component to test")

        rl_pipeline : RLPipelineComponent = component_from_dict(self.config_dict)
        
        name = 'configuration_' + str(self.tried_configurations + 1)        
                
        configuration_logger = self.lg.openChildLog(logName=name)
        
        rl_pipeline.pass_input({"logger_object" : configuration_logger})
        
        self.lg.writeLine(f"Created component with name {name}")
        
        return rl_pipeline


    def generate_configuration(self, trial : optuna.trial, base_component : Schema):
        
        for hyperparameter_suggestion in self.hyperparameters_range_list:
            
            suggested_value = hyperparameter_suggestion.make_suggestion(source_component=base_component, trial=trial)
            
            self.suggested_values[hyperparameter_suggestion.name] = [suggested_value]
            
            self.lg.writeLine(f"{hyperparameter_suggestion.name}: {suggested_value}")
                
                
                
    def objective(self, trial : optuna.Trial):
        
        self.component_to_test = self.create_component_to_test()

        self.lg.writeLine("Starting new training with hyperparameter cofiguration")
        
        self.generate_configuration(trial, self.component_to_test)
        
        self.episodes_per_test = int(self.component_to_test.input["num_episodes"] / self.n_steps)

        self.lg.writeLine(f"Number of episodes that will be done per step: {self.episodes_per_test}")
        
        for step in range(self.n_steps):
                
            self.component_to_test.train(self.episodes_per_test)
            
            results_logger : ResultLogger = self.component_to_test.get_results_logger() 

            avg_result, std_result = results_logger.get_avg_and_std_n_last_results(10, 'total_reward')

            result = avg_result - (std_result / 4)
            
            trial.report(result, step)
            
            if trial.should_prune():
                self.lg.writeLine("Prunning current experiment...")
                raise optuna.TrialPruned()
                    
        results_logger : ResultLogger = self.component_to_test.get_results_logger() 

        avg_result, std_result = results_logger.get_avg_and_std_n_last_results(10, 'total_reward')

        result = avg_result - (std_result / 4)
        
        
        return result
    
    def after_trial(self, study : optuna.Study, trial : optuna.trial.FrozenTrial):
        
        self.tried_configurations += 1
        
        result = trial.value
        
        results_to_log = {'experiment' : self.tried_configurations, **self.suggested_values, "result" : [result]}
        
        self.results_logger.log_results(results_to_log)
        
        
        self.component_to_test.save_configuration()
        
        del self.component_to_test
    
    
    # EXPOSED METHODS -------------------------------------------------------------------------------------------------
                    
                    
    @requires_input_proccess
    def run(self):
        
        study = optuna.create_study(sampler=self.sampler, 
                                    storage=self.storage, 
                                    study_name=self.input["database_study_name"], 
                                    load_if_exists=True,
                                    direction=self.input['direction'])


        study.optimize( lambda trial : self.objective(trial), 
                       n_trials=self.n_trials,
                       callbacks=[self.after_trial])

        self.lg.writeLine(f"Best parameters: {study.best_params}") 
        
        
        
        
