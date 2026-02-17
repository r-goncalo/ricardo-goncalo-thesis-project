import os
from typing import Union
from automl.component import InputSignature, Component, requires_input_proccess
from automl.basic_components.exec_component import ExecComponent, State
from automl.core.advanced_input_management import ComponentInputSignature
from automl.basic_components.evaluator_component import EvaluatorComponent
from automl.hp_opt.hp_suggestion.hyperparameter_suggestion import HyperparameterSuggestion
from automl.hp_opt.optuna.custom_pruners import MixturePruner
from automl.loggers.component_with_results import ComponentWithResults
from automl.loggers.result_logger import ResultLogger
from automl.rl.evaluators.rl_std_avg_evaluator import LastValuesAvgStdEvaluator

from automl.loggers.logger_component import ComponentWithLogging

from automl.utils.files_utils import write_text_to_file, read_text_from_file
from automl.utils.json_utils.json_component_utils import gen_component_from_dict,  dict_from_json_string, json_string_of_component_dict, value_from_json_string

import optuna

import shutil

from automl.basic_components.state_management import StatefulComponent
from automl.basic_components.seeded_component import SeededComponent
from automl.utils.configuration_component_utils import save_configuration

from automl.basic_components.state_management import save_state
 
import copy

from automl.consts import CONFIGURATION_FILE_NAME
from automl.core.exceptions import common_exception_handling

from automl.core.debug.debug_utils import substitute_classes_by_debug_classes

TO_OPTIMIZE_CONFIG_FILE = f"to_optimize_{CONFIGURATION_FILE_NAME}"

HYPERPARAMETER_PATH = "hyperparameters.json"

Component_to_opt_type = Union[ExecComponent, StatefulComponent, ComponentWithLogging]

OPTUNA_STUDY_PATH = 'study_results.db'

BASE_CONFIGURATION_NAME = 'configuration'

class HyperparameterOptimizationPipeline(ExecComponent, ComponentWithLogging, ComponentWithResults, StatefulComponent, SeededComponent):
    
    '''
    An hyperparameter optimization pipeline
    '''


    parameters_signature = {
        
                        "sampler" : InputSignature(default_value="TreeParzen"),
                        "sampler_input" : InputSignature(mandatory=False),
        
                        "configuration_dict" : InputSignature(mandatory=False, possible_types=[dict, str]),
                        "base_component_configuration_path" : InputSignature(mandatory=False),

                        "database_study_name" : InputSignature(default_value='experiment'),
                        
                        "direction" : InputSignature(default_value='maximize'),
                                                
                        "hyperparameters_range_list" : InputSignature(mandatory=False),
                        "n_trials" : InputSignature(),
                        
                        "steps" : InputSignature(default_value=1, description="The number of times to run the component to evaluate, re-evaluating it at the end of each to know if it is pruned"),
                        "pruner" : InputSignature(mandatory=False),
                        "pruner_input" : InputSignature(mandatory=False),
                        
                        "evaluator_component" : ComponentInputSignature(
                            default_component_definition=(LastValuesAvgStdEvaluator, {}),
                            description="The evaluator component to be used for evaluating the components to optimize in their training process"
                            ),

                        "start_with_given_values" : InputSignature(default_value=True),

                        "continue_after_error" : InputSignature(default_value=True, ignore_at_serialization=True, description="If trials should continue after an error or not"),

                        "do_initial_evaluation" : InputSignature(default_value=False),

                        "debug_classes" : InputSignature(mandatory=False)
                                                    
                       }
            

    # PARTIAL INITIALIZATION -----------------------------------------------------------------------------
    
    
    def setup_files(self):
        
        '''Set ups partially an HP experiment for manual change before usage'''  
        
        print(f"Setting up files with input: \n{self.input}")
        
        self._setup_hp_to_optimize_config_file()
        self._setup_hp_config_file()
        
    
    def _setup_hp_to_optimize_config_file(self):

        '''To be called outside of the processing input loop, to setup the necessaru files for instantiating the config to be optimized'''
        
        # make sure values for artifact directory generation are set
        self.setup_default_value_if_no_value("artifact_relative_directory")
        self.setup_default_value_if_no_value("base_directory")
        self.setup_default_value_if_no_value("create_new_directory")

        self.setup_default_value_if_no_value("debug_classes")
        
        
        self._initialize_config_dict() # initializes self.config_dict from the input
        
        self_artifact_directory = self.get_artifact_directory() #
        
        json_str_of_exp = json_string_of_component_dict(self.config_dict, ignore_defaults=True, save_exposed_values=False, respect_ignore_order=False)
        
        configuration_path = os.path.join(self_artifact_directory, TO_OPTIMIZE_CONFIG_FILE)

        write_text_to_file(self_artifact_directory, TO_OPTIMIZE_CONFIG_FILE, json_str_of_exp, create_new=True)
            
        # remove input not supposed to be used
        self.remove_input("configuration_dict")
        self.pass_input({"base_component_configuration_path" : configuration_path})
            
    
    def _setup_hp_config_file(self):
        
        self.save_configuration(save_exposed_values=True, respect_ignore_order=False) 
        
        
                
    
    # INITIALIZATION -----------------------------------------------------------------------------


    def _proccess_input_internal(self): # this is the best method to have initialization done right after
                
        super()._proccess_input_internal()
                
        # LOAD VALUES
        self.start_with_given_values = self.get_input_value("start_with_given_values")
        self.study_name=self.get_input_value("database_study_name")
        self.n_steps = self.get_input_value("steps")
        
        self.n_trials = self.get_input_value("n_trials")
        self.evaluator_component : EvaluatorComponent = self.get_input_value("evaluator_component", look_in_attribute_with_name="evaluator_component")
        
        self.continue_after_error = self.get_input_value("continue_after_error")

        self.do_initial_evaluation = self.get_input_value("do_initial_evaluation")
        
        self.direction = self.get_input_value("direction")

        # MAKE NECESSARY INITIALIZATIONS
        self._initialize_hyperparameter_range_list()
        self._initialize_config_dict()
        self._initialize_sampler()
        self._initialize_database()
        self._initialize_pruning_strategy()
        self._initialize_study()

        # SETUP AUX VALUES
        
        parameter_names = [hyperparameter_specification.name for hyperparameter_specification in self.hyperparameters_range_list]
        
        self.lg.writeLine(f"Hyperparameter names: {parameter_names}")
        
        self._setup_results_logger(parameter_names)
                
        self._suggested_values_by_trials = {}  


    def _initialize_hyperparameter_range_list(self):

        self.hyperparameters_range_list : list[HyperparameterSuggestion] = self.get_input_value("hyperparameters_range_list")

        hyperparameters_path = os.path.join(self.get_artifact_directory(), HYPERPARAMETER_PATH)

        if self.hyperparameters_range_list is None:
    
            if not os.path.exists(hyperparameters_path):
                raise Exception(f"Did not define hyperparameters and path {hyperparameters_path} does not exist")
            
            self.lg.writeLine(f"Getting hyperparameters from path: {hyperparameters_path}, as they were not passed to input")

            json_str_of_hyperparameters = read_text_from_file(hyperparameters_path)

            self.hyperparameters_range_list = value_from_json_string(self, json_str_of_hyperparameters)

        
        else:
            if os.path.exists(hyperparameters_path):
                self.lg.writeLine(f"Hyperparameters in path and input, using the ones in input...")

        

    def _setup_results_logger(self, parameter_names):
        self.add_to_columns_of_results_logger(["experiment", "step", *parameter_names, "result"])
                

    def _setup_configuration_dict_with_debug_classes(self):
        '''Substitutes the configuration dict by passed debug classes'''

        self.config_dict = substitute_classes_by_debug_classes(self.config_dict, self.debug_classes)
    
    # initialize the base configuration to create components to test
    def _initialize_config_dict(self):
        
        if "base_component_configuration_path" in self.input.keys():

            self.rl_pipeline_config_path : str = self.get_input_value("base_component_configuration_path")
            self.load_configuration_dict_from_path(self.rl_pipeline_config_path)

            self.input.pop("base_component_configuration_path")

            shutil.copy(self.rl_pipeline_config_path, os.path.join(self.get_artifact_directory(), TO_OPTIMIZE_CONFIG_FILE))
        
  
        elif "configuration_dict" in self.input.keys():

            self.config_dict = self.get_input_value("configuration_dict")

            if isinstance(self.config_dict, str):
                self.config_dict = dict_from_json_string(self.config_dict)
        
        else:

            hp_component_path = self.get_artifact_directory()
            possible_configuration_file = os.path.join(hp_component_path, TO_OPTIMIZE_CONFIG_FILE)

            if os.path.exists(possible_configuration_file):

                self.rl_pipeline_config_path : str = possible_configuration_file
                self.load_configuration_dict_from_path(self.rl_pipeline_config_path)

            else:
                raise Exception("No configuration defined")
            
        self.debug_classes = self.get_input_value("debug_classes")
            
        if self.debug_classes is not None and len(self.debug_classes) > 0:
            self.lg.writeLine(f"There are debug classes to use: {self.debug_classes}")
            self._setup_configuration_dict_with_debug_classes()
            
        
        
    
    #it constructs the path for the database        
    def _initialize_database(self):
        
        self.database_path = os.path.join(self.get_artifact_directory(), OPTUNA_STUDY_PATH)  # Path to the SQLite database file
        
        self.lg.writeLine(f"Trying to initialize database in path: {self.database_path}")
        
        self.storage = f"sqlite:///{self.database_path}"  # This will save the study results to the file



    def get_database(self):

        return self.storage


    
    def _initialize_sampler(self):

        self.sampler = self.get_input_value("sampler")

        if isinstance(self.sampler, str):
            self._initialize_sampler_from_str(self.sampler)

        elif isinstance(self.sampler, type):
            self._initialize_sampler_from_class(self.sampler)

        else:
            raise Exception("Non valid type for sampler")
        
        
    
    def _initialize_sampler_from_str(self, sampler_str):

        self.lg.writeLine(f"Initializing sampler with string {sampler_str}")

        sampler_input = self.get_input_value("sampler_input")
        sampler_input = {} if sampler_input is None else sampler_input

        if sampler_input is not None:
            self.lg.writeLine(f"Sampler input: {sampler_input}")
        
        if sampler_str == "TreeParzen":
            
            self.sampler : optuna.samplers.BaseSampler = optuna.samplers.TPESampler(seed=self.seed, **sampler_input)

        elif sampler_str == "Random":
            self.sampler : optuna.samplers.BaseSampler = optuna.samplers.RandomSampler(seed=self.seed, **sampler_input)
        
        else:
            raise NotImplementedError(f"Non valid string for sampler '{sampler_str}'") 
    
    
        
    def _initialize_sampler_from_class(self, sampler_class : type[optuna.samplers.BaseSampler]):

        self.lg.writeLine(f"Initializing sampler with class {sampler_class}")

        sampler_input = self.get_input_value("sampler_input")
        sampler_input = {} if sampler_input is None else sampler_input

        if sampler_input is not None:
            self.lg.writeLine(f"Sampler input: {sampler_input}")

        try:
            self.sampler = sampler_class(seed=self.seed, **sampler_input)

        except Exception as e:

            raise Exception(f"Could not instatiate sampler from class {sampler_class}") from e
    
    
    def _initialize_pruning_strategy(self):
        
        passed_pruner = self.get_input_value("pruner")

        if passed_pruner is not  None:

            pruner_input = self.get_input_value("pruner_input")

            if pruner_input is not None:
                self.lg.writeLine(f"Pruner input passed: {pruner_input}")

            else:
                pruner_input = {}
                self.lg.writeLine(f"No pruner input passed")

            self.pruning_strategy = self._return_pruning_strategy(passed_pruner, pruner_input)
        
        else:
            self.lg.writeLine(f"We won't use a pruning strategy, none passed")

        
    def _return_pruning_strategy(self, passed_pruner, pruner_input):
                
            
        self.lg.writeLine("Prunning strategy was defined")
                    
        if isinstance(passed_pruner, optuna.pruners.BasePruner):
            
            self.lg.writeLine(f"Passed instanced pruner of type: {type(passed_pruner)}")
            pruning_strategy = passed_pruner
        
        elif isinstance(passed_pruner, str):

            self.lg.writeLine(f"Passed pruner string: {passed_pruner}")
        
            pruning_strategy = self._initialize_pruner_from_string(passed_pruner, pruner_input)
            

        else:
            raise NotImplementedError(f"Pruner type {type(passed_pruner)} is not implemented")
        
        return pruning_strategy
            
            
        
    def _initialize_pruner_from_string(self, passed_pruner_str : str, pruner_input):

        self.lg.writeLine(f"Initializing pruner from string {passed_pruner_str} and input {pruner_input}")
        
        
        if passed_pruner_str == "Median":
                pruning_strategy = optuna.pruners.MedianPruner(**pruner_input)
                
        elif passed_pruner_str == "PercentilePruner":
                pruning_strategy = optuna.pruners.PercentilePruner(**pruner_input)

        elif passed_pruner_str == "HyperbandPruner":
                pruning_strategy = optuna.pruners.HyperbandPruner(**pruner_input)

        elif passed_pruner_str == 'MixturePruner':
            
            pruners_for_mixture = pruner_input["pruners"]
            instanced_pruners_for_mixture = []


            for pruner_definition in pruners_for_mixture:
                
                pruner_for_mixture = pruner_definition[0]
                pruner_for_mixture_input = pruner_definition[1]

                instanced_pruners_for_mixture.append(
                    self._return_pruning_strategy(pruner_for_mixture, pruner_for_mixture_input)
                )

            pruning_strategy = MixturePruner([
                    instanced_pruners_for_mixture        
                ])


        
        else:
            raise NotImplementedError(f"Pruner '{passed_pruner_str}' is not implemented")
        

        return pruning_strategy
    
    
        
    def load_configuration_dict_from_path(self, path):

        '''Loads the configuration dict that will be optimized from a path'''
        
        with open(path, 'r') as fd:
            fd = open(path, 'r') 
            self.config_str = fd.read()
            fd.close()

        self.config_dict = dict_from_json_string(self.config_str)



    # OPTIMIZATION -------------------------------------------------------------------------
    
    
    def _create_component_to_optimize_configuration(self, trial : optuna.Trial) -> dict:
        
        '''Creates the configuration dictionary and making the hyperparameter suggestions'''
                
        config_of_opt = copy.deepcopy(self.config_dict)
        
        self._setup_trial_component_with_suggestion(trial, config_of_opt)
        
        return config_of_opt
        

    def _setup_component_to_optimize_after_creation(self, trial : optuna.Trial, component_to_test : Component_to_opt_type):

        component_to_test.pass_input({"times_to_run" :  self.n_steps }) # it is responsibility of the component being optimized to deal with any cut in computation it should made from times_to_run
        
    
    def _create_component_to_optimize(self, trial : optuna.Trial) -> Component_to_opt_type:
        
        '''Creates the actual component to optimize, using the configuration dictionary and making the hyperparameter suggestions'''
        
        self.lg.writeLine(f"Creating component to test for trial {trial.number}...\n")
        
        config_of_opt = self._create_component_to_optimize_configuration(trial)
                
        component_to_opt : Component_to_opt_type = gen_component_from_dict(config_of_opt)
                
        name = self.gen_trial_name(trial)  
        
        component_to_opt.pass_input({"name" : name, 
                                     "artifact_relative_directory" : name, 
                                     "base_directory" : self.get_artifact_directory(), 
                                     "create_new_directory" : False,
                                     "save_state_on_run_end" : True,
                                     "save_dataframes_on_run_end" : True})  
        
        self._setup_component_to_optimize_after_creation(trial, component_to_opt)
        
        self.lg.writeLine(f"Created component with name {name}\n")
        
        return component_to_opt
        
            
    def gen_trial_name(self, trial : optuna.Trial) -> str:
        '''Generates a name for the trial based on its number'''
        
        return f"{BASE_CONFIGURATION_NAME}_{trial.number}"


    def get_component_to_test(self, trial : optuna.Trial) -> Component_to_opt_type:
        '''Gets the component to test, loading it or creating it'''
        raise NotImplementedError()

    def get_component_to_test_path(self, trial : optuna.Trial) -> str:
        raise NotImplementedError()


    # SETUP TRIALS AND SUGGESTIONS -------------------------------------------------------------------------


    def _queue_trial_with_suggestion(self, suggestion_dict):

        self.lg.writeLine(f"Queued trial with values: {suggestion_dict}")

        self.study.enqueue_trial(suggestion_dict)


    def _queue_trial_with_initial_suggestion(self):

        self.lg.writeLine(f"Queueing trial with initial suggestion...")

        initial_suggestion = {}

        for hp_suggestion in self.hyperparameters_range_list:

            initial_optuna_values : dict = hp_suggestion.try_get_suggested_optuna_values(self.config_dict)

            if initial_optuna_values != None:
                initial_suggestion = {**initial_suggestion, **initial_optuna_values}

            else:
                self.lg.writeLine(f"Couldn't initialize hyperparameter suggestion {hp_suggestion.name} with given value")

        self.lg.writeLine(f"Initial (optuna coded) values retrieved from configuration: {initial_suggestion.keys()}")

        if initial_suggestion != {}:
            self._queue_trial_with_suggestion(initial_suggestion)



    def _setup_trial_component_with_suggestion(self, trial : optuna.Trial, base_component : Union[Component, dict]):
        
        '''Generated the configuration for the trial, making suggestions for the hyperparameters'''
        
        self.lg.writeLine("Generating configuration for trial " + str(trial.number))

        if trial.number in self._suggested_values_by_trials.keys():
            raise Exception(f"Trial {trial.number} already had value(s) in suggested values: {self._suggested_values_by_trials[trial.number]}")

        self._suggested_values_by_trials[trial.number] = {}
        
        for hyperparameter_suggestion in self.hyperparameters_range_list:

            # if the trial was created with a specification of the value
            if hyperparameter_suggestion.already_has_suggestion_in_trial(trial):

                suggested_value = trial.params[hyperparameter_suggestion.name]

            # if we should use the hp_suggestion object to suggest a value to the trial
            else:

                suggested_value = hyperparameter_suggestion.make_suggestion(trial=trial)
                self.lg.writeLine(f"For hyperparameter {hyperparameter_suggestion.name}, value {suggested_value} was sampled, using the sampler {self.sampler}")
            
            #save suggestion value in our internal dict
            self._suggested_values_by_trials[trial.number][hyperparameter_suggestion.name] = suggested_value

            hyperparameter_suggestion.set_suggested_value(suggested_value, base_component) # set suggested value in component
            
            self.lg.writeLine(f"{hyperparameter_suggestion.name}: {suggested_value}")
            


    #  POST PROCESSING OF RUN TRIAL --------------------------------------------------------------------------------- 

    def _evaluate_component(self, component_to_test : Component_to_opt_type) -> float:

        '''Evaluates the component that has already run and returns the evaluation result'''
        
        if self.evaluator_component is None: # TODO: Implement this, right now, the evaluator component is mandatory
            
            if not isinstance(component_to_test, ComponentWithResults):
                raise Exception("Component to test is not a ComponentWithResults, cannot get evaluation")
            
            component_to_test_results : ResultLogger = component_to_test.get_results_logger()
            
            result = component_to_test_results.get_last_results()
            
        else:
            
            result = self.evaluator_component.evaluate(component_to_test)
            
        return result          
    

    
    def _try_evaluate_component(self, component_to_test_path, trial : optuna.Trial, component_to_test = None) -> float:

        '''Tries evaluate the component that has already run and return its result, and deals with an exception that could appear'''
        
        try:
            if component_to_test is None:
                component_to_test = self.get_component_to_test(trial)      
            
            return self._evaluate_component(component_to_test)
        
        except Exception as e:
            self.on_exception_evaluating_trial(e, component_to_test_path, trial)
            raise e
        

    
    def _try_run_component(self, component_to_test_path, trial : optuna.Trial):

        '''Tries running the component, raising and dealing with an exception if it appears'''
        try:
            component_to_test = self.get_component_to_test(trial)      
            return component_to_test.run()

        except Exception as e:
            self.on_exception_running_trial(e, component_to_test_path, trial)
            raise e
    


    def _try_save_state_of_trial(self, component_to_test_path, trial : optuna.Trial):
    
        '''Tries saving the state of the component, raising and dealing with an exception if it appears'''

        try:
            self.lg.writeLine(f"Trying to save state of trial {trial.number}")
            component_to_test = self.get_component_to_test(trial)                 
            return save_state(component_to_test, save_definition=True)

        except Exception as e:
            self.on_exception_saving_trial(e, component_to_test_path, trial)
            raise e
        

    # RUNNING A TRIAL -----------------------------------------------------------------------

    def log_results_of_trial(self, trial : optuna.Trial, step : int, evaluation_results):

        result = evaluation_results["result"]

        results_to_log = {'experiment' : trial.number, "step" : step, **self._suggested_values_by_trials[trial.number], "result" : result}

        for key, value in results_to_log.items():
            results_to_log[key] = [value]

        self.log_results(results_to_log)  


    def _run_single_step_of_objective(self, trial : optuna.Trial, step):


            component_to_test_path = self.get_component_to_test_path(trial) # we get the path at tehe begining so errors don't take away or capability to do so

            self._try_run_component(component_to_test_path, trial) # we try to run the trial, this can raise an exception

            self.lg.writeLine(f"Evaluating trial {trial.number}...")

            evaluation_results = self._try_evaluate_component(component_to_test_path, trial) # and then, if all succeded, we try to evaluate it, which can also raise an exception

            self.lg.writeLine(f"Evaluation results for trial {trial.number} at step {step}: {evaluation_results}")

            return evaluation_results                

    def _try_run_single_step_of_objective(self, trial : optuna.Trial, step):

            self.lg.writeLine()
            self.lg.writeLine(f"Starting step {step} of {self.n_steps} total steps for trial {trial.number}")

            component_to_test_path = self.get_component_to_test_path(trial)
             
            try:

                evaluation_results = self._run_single_step_of_objective(trial, step)

                try:

                    result = evaluation_results["result"]
                    trial.report(result, step) # we report the results to optuna

                    self.log_results_of_trial(trial, step, evaluation_results)

                except Exception as e:
                    self.on_general_exception_trial(e, component_to_test_path, trial)

                    self.lg.writeLine(f"Ended step {step}") 

                if trial.should_prune(): # we verify this after reporting the result
                    self.lg.writeLine("Prunning current experiment due to pruner...")
                    trial.set_user_attr("prune_reason", "pruner")
                    raise optuna.TrialPruned()
                    

                return evaluation_results

            except Exception as e:

                if isinstance(e, optuna.TrialPruned):
                    raise e # don't consume exception, let it pass, so optuna can deal with it
                                
                # if we reached here, it means an exception other than the trial being pruned was got
                self.lg.writeLine(f"Error in trial {trial.number} at step {step}, prunning it...")
                trial.set_user_attr("prune_reason", "error")


                if self.continue_after_error: # if we are to continue after an error, we count the trial simply as pruned, and let optuna deal with it
                    self.lg.writeLine(f"Exception will make the trial be ignored and continue, exception was: {e}")
                    raise optuna.TrialPruned("error")
                
                else: # if not, we propagate the exception
                    self.lg.writeLine("As <continue_after_error> was set to False, we end the Hyperparameter Optimization process and propagate the error to the caller")
                    raise e
                
            except KeyboardInterrupt as e:
            
                self.lg.writeLine(f"User interrupted experiment in trial {trial.number} at step {step}, prunning it...")
                trial.set_user_attr("prune_reason", "user_interrupt")

                raise e


    def _pre_proccess_component_before_objective(self, trial : optuna.Trial):
        pass

    
    def _run_optimization(self, trial: optuna.Trial):

        if self.do_initial_evaluation:

            self.lg.writeLine(f"Doing initial evaluation of trial {trial.number} before initiating training...")
            
            component_to_test_path = self.get_component_to_test_path(trial)
            evaluation_results = self._try_evaluate_component(component_to_test_path, trial) 
            
            self.lg.writeLine(f"Results were: {evaluation_results}\n")
            
            self.log_results_of_trial(trial, 0, evaluation_results)

        for step in range(1, self.n_steps + 1):

            evaluation_results = self._try_run_single_step_of_objective(trial, step)


        return evaluation_results["result"]   # this is the value the objective will optimize
            

    def objective(self, trial : optuna.Trial):
        
        '''Responsible for running the optimization trial and evaluating the component to test'''

        self._pre_proccess_component_before_objective(trial)

        return self._run_optimization(trial)
    
    def _call_objective(self):

        self.study.optimize( lambda trial : self.objective(trial), 
                       n_trials=self.n_trials,
                       callbacks=[self.after_trial])
        
    def after_trial(self, study : optuna.Study, trial : optuna.trial.FrozenTrial):
        
        '''
        Called when a trial is over
        It is passed to optuna in the callbacks when the objective is defined
        '''        
                        
        self._suggested_values_by_trials.pop(trial.number, None)

        pass

    
    # INTERNAL EXCEPTION HANDLING --------------------------------------------------------


    def _deal_with_exception(self, exception : Exception):
        
        super()._deal_with_exception(exception)
        
        common_exception_handling(self.lg, exception, 'error_report.txt')
    
    
                    
    # STUDY SETUP ------------------------------------------------------------------------------

    @requires_input_proccess
    def get_study(self):
        return self.study

    def _initialize_study(self):

        if not hasattr(self, "study"):

            try:
                # Try loading existing study
                self.study = optuna.load_study(
                    sampler=self.sampler,
                    storage=self.storage,
                    study_name=self.study_name,
                )
                self.lg.writeLine(f"Loaded existing study '{self.study_name}'")

                try:
                    trials_in_study = self.study.trials

                    trials_in_study_lines = []

                    for trial in trials_in_study:
                        trials_in_study_lines.append(
                            f"        Trial {trial.number}: intermediate values: {[f"step {step}: {value}" for step, value in trial.intermediate_values.items()]}, with result {trial.value}, and state {trial.state}"
                            )

                    self.lg.writeLine(f"Existing study had {len(self.study.trials)} trials:")
                    self.lg.writeLine('\n'.join(trials_in_study_lines), use_time_stamp=False)

                    if State.equals_value(self.values["running_state"], State.ERROR) or State.equals_value(self.values["running_state"], State.INTERRUPTED):
                        self.lg.writeLine(f"Noticed that current running state is {self.values['running_state']}, trials should be resumed")

                        number_of_completed_trials = sum(1 for trial in trials_in_study if trial.state == optuna.trial.TrialState.COMPLETE)

                        self.lg.writeLine(f"Only completed {number_of_completed_trials} trials")

                        new_number_of_trials = self.n_trials - number_of_completed_trials

                        self.lg.writeLine(f"Number of trials that were not done from the registered {self.n_trials}: {new_number_of_trials}")

                        self.n_trials = new_number_of_trials
                    

                except Exception as e:
                    self.lg.writeLine(f"Could not read trials in optuna study due to exception: {e}")
                

            except KeyError:
                # If not found, create a new study
                self.study = optuna.create_study(
                    sampler=self.sampler,
                    storage=self.storage,
                    study_name=self.study_name,
                    direction=self.direction ,
                )

                self.lg.writeLine(f"Created new study '{self.study_name}'")

                if self.start_with_given_values:
                    self.lg.writeLine("Starting the study with given values")
                    self._queue_trial_with_initial_suggestion()

        
        else:
            self.lg.writeLine("Running HP optimization with already loaded study")
    
    # EXPOSED METHODS -------------------------------------------------------------------------------------------------
                    
                    
    @requires_input_proccess
    def _algorithm(self): 

        self.lg.writeLine() 

        self.lg.writeLine(f"OPTIMIZING WITH {self.n_trials} TRIALS ------------------------------------------\n")      

        self._call_objective()
        
        self.lg.writeLine(f"OPTIMIZATION WITH {self.n_trials} TRIALS OVER --------------------------------------------------------------------")

        try:
            self.lg.writeLine(f"Best parameters: {self.study.best_params}, used in trial {self.study.best_trial.number}, with best result {self.study.best_value}" )
            
        
        except Exception as e:
            self.lg.writeLine(f"Error getting best parameters: {e}")


        
    # STATE MANAGEMENT ----------------------------------------------------------

        
        
    # TRIAL ERROR HANDLING -------------------------------------------


    
    def on_exception_saving_trial(self, exception : Exception, component_to_test_path, trial : optuna.Trial):

        self.lg.writeLine(f"ERROR: CAN'T SAVE TRIAL {trial.number}")

        component_to_test_path = os.path.abspath(component_to_test_path)

        error_report_specific_path = "on_save_error_report.txt"
        error_report_path = os.path.join(component_to_test_path, error_report_specific_path)

        self.lg.writeLine(f"Storing error report in configuration, path {error_report_path}\nError: {exception}")

        common_exception_handling(self.lg, exception, error_report_path)

        raise exception



    def on_exception_evaluating_trial(self, exception : Exception, component_to_test_path, trial : optuna.Trial):

        self.lg.writeLine(f"ERROR: CAN'T EVALUATE TRIAL {trial.number}")

        component_to_test_path = os.path.abspath(component_to_test_path)

        error_report_specific_path = "on_evaluate_error_report.txt"
        error_report_path = os.path.join(component_to_test_path, error_report_specific_path)

        self.lg.writeLine(f"Storing error report in configuration, path {error_report_path}\nError: {exception}")

        common_exception_handling(self.lg, exception, error_report_path)

        raise exception
    


    def on_exception_running_trial(self, exception : Exception, component_to_test_path, trial : optuna.Trial):

        self.lg.writeLine(f"ERROR RUNNING TRIAL {trial.number}")

        error_report_specific_path = "on_run_error_report.txt"
        
        component_to_test_path = os.path.abspath(component_to_test_path)

        error_report_path = os.path.join(component_to_test_path, error_report_specific_path)

        self.lg.writeLine(f"Storing error report in configuration, path {error_report_path}\nError: {exception}")

        common_exception_handling(self.lg, exception, error_report_path)

        raise exception
        

        
    def on_general_exception_trial(self, exception : Exception, component_to_test_path, trial : optuna.Trial):

        self.lg.writeLine(f"ERROR IN TRIAL {trial.number}")

        error_report_specific_path = "general_exception.txt"
        
        component_to_test_path = os.path.abspath(component_to_test_path)

        error_report_path = os.path.join(component_to_test_path, error_report_specific_path)

        self.lg.writeLine(f"Storing error report in configuration, path {error_report_path}\nError: {exception}")

        common_exception_handling(self.lg, exception, error_report_path)

        raise exception
