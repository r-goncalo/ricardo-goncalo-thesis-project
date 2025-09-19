import gc
import os
from typing import Union
from unittest import result
from automl.component import InputSignature, Component, requires_input_proccess
from automl.basic_components.artifact_management import ArtifactComponent
from automl.basic_components.exec_component import ExecComponent
from automl.core.advanced_input_management import ComponentInputSignature
from automl.basic_components.evaluator_component import ComponentWithEvaluator, EvaluatorComponent
from automl.loggers.component_with_results import ComponentWithResults
from automl.loggers.result_logger import ResultLogger
from automl.rl.evaluators.rl_std_avg_evaluator import LastValuesAvgStdEvaluator
from automl.rl.rl_pipeline import RLPipelineComponent

from automl.loggers.logger_component import LoggerSchema, ComponentWithLogging

from automl.utils.files_utils import write_text_to_file
from automl.utils.json_component_utils import gen_component_from_dict,  dict_from_json_string, json_string_of_component_dict, gen_component_from

import optuna

from automl.meta_rl.hyperparameter_suggestion import HyperparameterSuggestion

from automl.utils.random_utils import generate_and_setup_a_seed

from automl.basic_components.state_management import StatefulComponent, StatefulComponentLoader
import torch

from automl.basic_components.state_management import save_state
 
import copy

from automl.consts import CONFIGURATION_FILE_NAME

TO_OPTIMIZE_CONFIG_FILE = f"to_optimize_{CONFIGURATION_FILE_NAME}"

MEMORY_REPORT_FILE = "memory_report.txt"

Component_to_opt_type = Union[ExecComponent, StatefulComponent]


class HyperparameterOptimizationPipeline(ExecComponent, ComponentWithLogging, ComponentWithResults, StatefulComponent):
    
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
                        
                        "steps" : InputSignature(default_value=1, description="The number of times to run the component to evaluate, re-evaluating it at the end of each to know if it is pruned"),
                        "pruner" : InputSignature(mandatory=False),
                        "pruner_input" : InputSignature(mandatory=False),
                        
                        "evaluator_component" : ComponentInputSignature(
                            default_component_definition=(LastValuesAvgStdEvaluator, {}),
                            description="The evaluator component to be used for evaluating the components to optimize in their training process"
                            )
                                                    
                       }
            

    # PARTIAL INITIALIZATION -----------------------------------------------------------------------------
    
    
    def setup_files(self):
        
        '''Set ups partially an HP experiment for manual change before usage'''  
        
        print(f"Setting up files with input: \n{self.input}")
        
        self._setup_hp_to_optimize_config_file()
        self._setup_hp_config_file()
        
    
    def _setup_hp_to_optimize_config_file(self):
        
        # make sure values for artifact directory generation are set
        self._setup_default_value_if_no_value("artifact_relative_directory")
        self._setup_default_value_if_no_value("base_directory")
        self._setup_default_value_if_no_value("create_new_directory")
        
        
        self._initialize_config_dict() # initializes self.config_dict from the input
        
        self_artifact_directory = self.get_artifact_directory() #
        
        json_str_of_exp = json_string_of_component_dict(self.config_dict, ignore_defaults=True, save_exposed_values=False)
        
        configuration_path = os.path.join(self_artifact_directory, TO_OPTIMIZE_CONFIG_FILE)

        write_text_to_file(self_artifact_directory, TO_OPTIMIZE_CONFIG_FILE, json_str_of_exp, create_new=True)
            
        # remove input not supposed to be used
        self.remove_input("configuration_dict")
        self.remove_input("configuration_string")
        self.pass_input({"base_component_configuration_path" : configuration_path})
            
    
    def _setup_hp_config_file(self):
        
        self.save_configuration(save_exposed_values=True) 
        
        
                
    
    # INITIALIZATION -----------------------------------------------------------------------------


    def proccess_input_internal(self): # this is the best method to have initialization done right after
                
        super().proccess_input_internal()
                
        self._initialize_config_dict()
        self.initialize_sampler()
        self.initialize_database()
        self.initialize_pruning_strategy()
        
        self.n_steps = self.input["steps"]
        
        self.hyperparameters_range_list : list[HyperparameterSuggestion] = self.input["hyperparameters_range_list"]
        
        parameter_names = [hyperparameter_specification.name for hyperparameter_specification in self.hyperparameters_range_list]
        
        self.lg.writeLine(f"Hyperparameter names: {parameter_names}")
        
        #self.results_lg : ResultLogger = self.initialize_child_component(ResultLogger, { "artifact_relative_directory" : "",
        #    "results_columns" : ['experiment', *parameter_names, "result"]})
        
        self.add_to_columns_of_results_logger(["experiment", "step", *parameter_names, "result"])
                
        self.__suggested_values_by_trials = {}  
        
        self.n_trials = self.input["n_trials"]
                
        self.trial_loaders : dict[str, StatefulComponentLoader] = {}
        
        self.evaluator_component : EvaluatorComponent = ComponentInputSignature.get_component_from_input(self, "evaluator_component")
                
        
    def _initialize_config_dict(self):
        
        if "base_component_configuration_path" in self.input.keys():
            self.load_configuration_dict_from_path()
  
        elif "configuration_dict" in self.input.keys():
            
            if not isinstance(self.input["configuration_dict"], dict):
                raise Exception("Configuration input passed is not a dictionary")
            
            self.config_dict = self.input["configuration_dict"]
            
        elif "configuration_string" in self.input.keys():
            self.config_dict = dict_from_json_string(self.input["configuration_string"])
            
        else:
            raise Exception("No configuration defined")
        
        
            
    def initialize_database(self):
        
        self.database_path = self.get_artifact_directory() + "\\study_results.db"  # Path to the SQLite database file
        
        self.lg.writeLine(f"Trying to initialize database in path: {self.database_path}")
        
        self.storage = f"sqlite:///{self.database_path}"  # This will save the study results to the file
    
    
    
    def initialize_sampler(self):
        
        if self.input["sampler"] == "TreeParzen":
            
            self.sampler : optuna.samplers.BaseSampler = optuna.samplers.TPESampler(seed=self.input["seed"])
        
        else:
            raise NotImplementedError(f"Did not implement for sampler '{self.input['sampler']}'") 
    
    
    
        
    def initialize_sampler_from_class(self, sampler_class : type[optuna.samplers.BaseSampler]):
        
        self.sampler : sampler_class(seed=self.input["seed"])
    
    
        
    def initialize_pruning_strategy(self):
        
        
        if 'pruner' in self.input.keys():
            
            passed_pruner = self.input["pruner"]

                        
            if isinstance(passed_pruner, optuna.pruners.BasePruner):
                
                self.lg.writeLine(f"Passed instanced pruner of type: {type(passed_pruner)}")
                self.pruning_strategy = passed_pruner
            
            elif isinstance(passed_pruner, str):
            
                self._initialize_pruner_from_string(passed_pruner)
                self.lg.writeLine(f"Passed pruner string: {passed_pruner}")
        
            else:
                raise NotImplementedError(f"Pruner type {type(passed_pruner)} is not implemented")
        
        else:
            self.lg.writeLine("Won't use prunning strategy, none passed")
            
            
        
    def _initialize_pruner_from_string(self, passed_pruner_str : str):
        
        pruner_input = {}
        
        if "pruner_input" in self.input.keys():
            pruner_input = {**pruner_input, **self.input["pruner_input"]}
        
        if passed_pruner_str == "Median":
            
                self.pruning_strategy = optuna.pruners.MedianPruner(**pruner_input)
                
        elif passed_pruner_str == "PercentilePruner":
            
                self.pruning_strategy = optuna.pruners.PercentilePruner(**pruner_input)
        
        else:
            raise NotImplementedError(f"Pruner '{passed_pruner_str}' is not implemented")
    
    
        
    def load_configuration_dict_from_path(self):
        
        self.rl_pipeline_config_path : str = self.input["base_component_configuration_path"]
        
        fd = open(self.rl_pipeline_config_path, 'r') 
        self.config_str = fd.read()
        fd.close()
        
        self.config_dict = dict_from_json_string(self.config_str)        
        

    # OPTIMIZATION -------------------------------------------------------------------------
    
    
    def create_component_to_optimize_configuration(self, trial : optuna.Trial) -> dict:
        
        '''Creates the configuration dictionary and making the hyperparameter suggestions'''
                
        config_of_opt = copy.deepcopy(self.config_dict)
        
        self.setup_trial_component_with_suggestion(trial, config_of_opt)
        
        return config_of_opt
        
        
        
    
    def create_component_to_optimize(self, trial : optuna.Trial) -> Component_to_opt_type:
        
        '''Creates the actual component to optimize, using the configuration dictionary and making the hyperparameter suggestions'''
        
        self.lg.writeLine(f"Creating component to test for trial {trial.number}...")
        
        config_of_opt = self.create_component_to_optimize_configuration(trial)
                
        component_to_opt : Component_to_opt_type = gen_component_from_dict(config_of_opt)
                
        name = self.gen_trial_name(trial)  
        
        component_to_opt.pass_input({"name" : name, "base_directory" : self.get_artifact_directory()})  
        
        self.lg.writeLine(f"Created component with name {name}")
        
        return component_to_opt
        
        
    
    def gen_trial_name(self, trial : optuna.Trial) -> str:
        '''Generates a name for the trial based on its number'''
        
        return f"configuration_{trial.number}"



    def create_component_to_test(self, trial : optuna.Trial) -> Component_to_opt_type:
        
        '''Creates the component to optimize and and saver / loader for it, returning the component to optimize itself'''
        
        self.lg.writeLine(f"Creating component to test for trial {trial.number} with current memory info\n: {torch.cuda.memory_summary() if torch.cuda.is_available() else 'No CUDA available'}", file=MEMORY_REPORT_FILE)

        
        component_to_opt = self.create_component_to_optimize(trial)
        
        component_to_opt.pass_input({"base_directory" : self.get_artifact_directory(), "artifact_relative_directory" : component_to_opt.name, "create_new_directory" : True})
        
        component_saver_loader = StatefulComponentLoader()
        component_saver_loader.define_component_to_save_load(component_to_opt)
        
        self.trial_loaders[trial.number] = component_saver_loader
                
        return component_to_opt
    
    
    def load_component_to_test(self, trial : optuna.Trial) -> Component_to_opt_type:

        self.lg.writeLine(f"\nLoading component to test for trial {trial.number} with current memory info\n: {torch.cuda.memory_summary() if torch.cuda.is_available() else 'No CUDA available'}", file=MEMORY_REPORT_FILE)
        
        component_saver_loader = self.trial_loaders[trial.number]
        component_to_opt = component_saver_loader.load_component()
        
        return component_to_opt    
    
    
    def create_or_load_component_to_test(self, trial : optuna.Trial) -> Component_to_opt_type:
        
        if trial.number in self.trial_loaders.keys():
            component_to_opt = self.load_component_to_test(trial)
            
        else:
            component_to_opt = self.create_component_to_test(trial)
        
        return component_to_opt


    def unload_component_to_test(self, trial : optuna.Trial):

        self.lg.writeLine(f"\nUnloading component to test for trial {trial.number} with current memory info\n: {torch.cuda.memory_summary() if torch.cuda.is_available() else 'No CUDA available'}", file=MEMORY_REPORT_FILE)
        
        component_saver_loader = self.trial_loaders[trial.number]
        component_saver_loader.unload_component()

    def setup_trial_component_with_suggestion(self, trial : optuna.trial, base_component : Union[Component, dict]):
        
        '''Generated the configuration for the trial, making suggestions for the hyperparameters'''
        
        self.lg.writeLine("Generating configuration for trial " + str(trial.number))

        self.__suggested_values_by_trials[trial] = {}
        
        for hyperparameter_suggestion in self.hyperparameters_range_list:
            
            suggested_value = hyperparameter_suggestion.make_suggestion(source_component=base_component, trial=trial)
            
            self.__suggested_values_by_trials[trial][hyperparameter_suggestion.name] = [suggested_value]
            
            self.lg.writeLine(f"{hyperparameter_suggestion.name}: {suggested_value}")
            
            
                
                
    def evaluate_component(self, component_to_test : Component_to_opt_type) -> float:
        
        if self.evaluator_component is None: # TODO: Implement this, right now, the evaluator component is mandatory
            
            if not isinstance(component_to_test, ComponentWithResults):
                raise Exception("Component to test is not a ComponentWithResults, cannot get evaluation")
            
            component_to_test_results : ResultLogger = component_to_test.get_results_logger()
            
            result = component_to_test_results.get_last_results()["result"]
            
        else:
            
            result = self.evaluator_component.evaluate(component_to_test)
            
        return result
    
    def try_save_stat_of_trial(self, component_to_test : Component_to_opt_type, trial = optuna.Trial):
    
        try:                 
            save_state(component_to_test, save_definition=True)
        except Exception as e:
            self.on_exception_saving_trial(e, component_to_test, trial)

                
    def objective(self, trial : optuna.Trial):
        
        '''Responsible for running the optimization trial and evaluating the component to test'''
        
        self.lg.writeLine("\nStarting new training with hyperparameter cofiguration")
                
        component_to_test  = self.create_or_load_component_to_test(trial)

        
        for step in range(self.n_steps):
                        
            try:

                try:
                    component_to_test.run()

                except Exception as e:

                    self.lg.writeLine(f"EXCEPTION TESTING COMPONENT IN TRIAL {trial.number}")

                    try:
                        self.try_save_stat_of_trial(component_to_test, trial)
                    
                    except Exception as saving_exception:
                        self.lg.writeLine(f"EXCEPTION TRYING SAVING TRIAL AFTER ORIGINAL EXCEPTION")
                        raise e

                self.try_save_stat_of_trial(component_to_test, trial)
                

                evaluation_results = self.evaluate_component(component_to_test)
                
                self.lg.writeLine(f"Evaluation results for trial {trial.number} at step {step}: \n{evaluation_results}")

                result = evaluation_results["result"]

                trial.report(result, step)
                
                results_to_log = {'experiment' : trial.number, "step" : step, **self.__suggested_values_by_trials[trial], "result" : [result]}
                
                print(f"Logging results: {results_to_log}")

                self.log_results(results_to_log)                

                if trial.should_prune():
                    self.lg.writeLine("Prunning current experiment dues to pruner...")
                    trial.set_user_attr("prune_reason", "pruner")
                    raise optuna.TrialPruned()
                
            except Exception as e:
                
                self.lg.writeLine(f"Error in trial {trial.number}, prunning it")
                trial.set_user_attr("prune_reason", "error")
                raise optuna.TrialPruned("error")
                            
        
        return evaluation_results["result"]
    
    
    
    def after_trial(self, study : optuna.Study, trial : optuna.trial.FrozenTrial):
        
        '''Called when a trial is over'''        
                        
        self.unload_component_to_test(trial)


    def onException(self, exception : Exception):

        import traceback
        
        super().onException(exception)
        
        error_message = str(exception)
        full_traceback = traceback.format_exc()

        self.lg.writeLine("Error message:", file="error_report.txt")
        self.lg.writeLine(error_message, file="error_report.txt")

        self.lg.writeLine("\nFull traceback:")
        self.lg.writeLine(full_traceback, file="error_report.txt")

        raise exception
                    
    
    # EXPOSED METHODS -------------------------------------------------------------------------------------------------
                    
                    
    @requires_input_proccess
    def algorithm(self):        

        study = optuna.create_study(sampler=self.sampler, 
                                    storage=self.storage, 
                                    study_name=self.input["database_study_name"], 
                                    load_if_exists=True,
                                    direction=self.input['direction'])

        study.optimize( lambda trial : self.objective(trial), 
                       n_trials=self.n_trials,
                       callbacks=[self.after_trial])

        self.lg.writeLine(f"Best parameters: {study.best_params}") 
        
        
    # TRIAL ERROR HANDLING -------------------------------------------


    
    def on_exception_saving_trial(self, exception : Exception, component_to_test : Component_to_opt_type, trial : optuna.Trial):

        import traceback

        self.lg.writeLine(f"ERROR: CAN'T SAVE TRIAL {trial.number}, storing error report in it")
        
        error_message = str(exception)
        full_traceback = traceback.format_exc()

        error_report_specific_path = "on_save_error_report.txt"
        error_report_path = os.path.join(component_to_test.get_artifact_directory(), error_report_specific_path)

        self.lg.writeLine("Error message:", file=os.path.joinerror_report_path)
        self.lg.writeLine(error_message, file=error_report_path)

        self.lg.writeLine("\nFull traceback:")
        self.lg.writeLine(full_traceback, file=error_report_path)

        raise exception
        
        
