
import os

import pandas
from automl.component import Component, requires_input_proccess
from automl.core.advanced_input_management import ComponentInputSignature
from automl.rl.environment.environment_sampler import EnvironmentSampler
from automl.utils.json_utils.json_component_utils import gen_component_from
from automl.loggers.logger_component import ComponentWithLogging
from automl.rl.evaluators.rl_component_evaluator import RLPipelineEvaluator
from automl.rl.rl_pipeline import RLPipelineComponent
from automl.loggers.result_logger import ResultLogger, aggregate_results_logger

from automl.core.input_management import InputSignature

from automl.rl.rl_player.rl_player import RLPlayer
from automl.rl.evaluators.rl_std_avg_evaluator import LastValuesAvgStdEvaluator
from automl.utils.files_utils import loadDataframe, saveDataframe
from automl.utils.configuration_component_utils import save_configuration
from automl.rl.agent.agent_components import AgentSchema

class EvaluatorWithPlayer(RLPipelineEvaluator):
    
    '''
    An evaluator specific for RL pipelines, which evaluates the last results it has and uses them to compute a result, penalizing high variance and using the mean as the base value
    
    This is meant to be used not as a final evaluation of a component, but as an intermediary evaluator at training time.
    
    '''
    
    parameters_signature = {
        "base_evaluator" : ComponentInputSignature(default_component_definition=(LastValuesAvgStdEvaluator, {})),
        "rl_player_definition" : InputSignature(default_value=(RLPlayer, {})),
        "number_of_episodes" : InputSignature(default_value=5),
        "number_of_evaluations" : InputSignature(default_value=1),
        "environment" : ComponentInputSignature(mandatory=False),
    }
    
    exposed_values = {
        
        "last_evaluation" : {},
        "number_of_evaluations" : 0
    
    }
    

    def _proccess_input_internal(self):
        
        super()._proccess_input_internal()
        
        self.base_evaluator : RLPipelineEvaluator = self.get_input_value("base_evaluator")

        self.number_of_episodes = self.get_input_value("number_of_episodes")
        self.number_of_evaluations = self.get_input_value("number_of_evaluations")
        self.rl_player_definition = self.get_input_value("rl_player_definition")
        
        self._setup_environment()


    def _setup_environment(self):
        
        self.env = None
        
        if "environment" in self.input.keys():
            self.env = self.get_input_value("environment")
            
            
    # EVALUATION -------------------------------------------------------------------------------
    
    def get_metrics_strings(self) -> list[str]:
        return [*super().get_metrics_strings(), *self.base_evaluator.get_metrics_strings()]
    
    def _evaluate(self, component_to_evaluate : RLPipelineComponent):
        
        if isinstance(component_to_evaluate, tuple):
            return self._evaluate_from_tuple(component_to_evaluate)
    
        else: # Is of type RLPipelineComponent or something with same attributes
            return self._evaluate_from_component(component_to_evaluate)
        
    
    
    def _evaluate_from_component(self, component_to_evaluate : RLPipelineComponent):
        
        if self.env is not None:
            env = self.env
        else:
            env = component_to_evaluate.env


        agents = component_to_evaluate.agents
        device = component_to_evaluate.device
        evaluations_directory = os.path.join(component_to_evaluate.get_artifact_directory(), "evaluations")


        return self._evaluate_agents(agents, device, evaluations_directory, env)
        
        
    def _evaluate_from_tuple(self, tuple):
        
        (agents, device, evaluations_directory, env) = tuple

        return self._evaluate_agents(agents, device, evaluations_directory, env)

    def __generalize_get_environment(self, env, rl_player : RLPlayer = None):
        
        '''Generalizes the process of generating, setting or simply using a passed environment'''

        if env is not None:
            env = env # we use the environment passed
            
        elif self.env is not None: 
            env = self.env # we use the environment stored in the component
        
        else:
            return None # no environment was passed or stored, we return None

        env = gen_component_from(env) # in case the environment passed isn't an instance but a definition of an environment

        if isinstance(env, EnvironmentSampler): # if a sampler was passed
            
            env = env.sample()
            
            env.pass_input({
                "base_directory" : rl_player.get_artifact_directory()
            })
        
        return env
    

    def _load_evaluation_result_df(self, evaluations_directory):

        try:
            return loadDataframe(evaluations_directory, "evaluations.csv")
        
        except:
            return None
    
    def _save_evaluation_result(self, result, evaluations_directory):

        current_daframe = self._load_evaluation_result_df(evaluations_directory)

        if current_daframe == None: # TODO: add column named "evaluation" which is the number of the row
            saveDataframe(pandas.DataFrame([{"evaluation" : 0, **result}]), evaluations_directory, "evaluations.csv")

        else:
            new_row = pandas.DataFrame([{"evaluation" : len(current_daframe), **result}])

            updated = pandas.concat([current_daframe, new_row], ignore_index=True)
            saveDataframe(updated, evaluations_directory, "evaluations.csv")


        
    def _evaluate_agents(self, agents, device, evaluations_directory, env=None):

        '''Evaluate agents using the RL player and the base evaluator'''
        
        
        results_loggers_of_new_plays = []
                
        # compute new evaluations
        for i in range(self.number_of_evaluations): # evaluate plays and store their paths 

            rl_player_of_run : RLPlayer = self._run_play_to_evaluate(agents, device, evaluations_directory, env) 

            results_loggers_of_new_plays.append(rl_player_of_run.get_results_logger())

        results_logger_of_new_plays = aggregate_results_logger(results_loggers_of_new_plays, evaluations_directory, new_results_filename=f"evaluation_results_{number_of_evals_done}.csv")
                
        evaluation_to_return = self.base_evaluator.evaluate(results_logger_of_new_plays)

        self._save_evaluation_result(evaluation_to_return, evaluations_directory)


        return evaluation_to_return
    

    def _run_play_to_evaluate(self, agents : dict[str, AgentSchema], device, evaluations_directory, env):

        '''
            Plays a session with the RL player, returning it after
            The objective is to then evaluate the results with the evaluator
        '''

        rl_player_will_be_generated = not isinstance(self.rl_player_definition, Component)
                
        rl_player : RLPlayer = gen_component_from(self.rl_player_definition)

        rl_player.pass_input({
            "agents" : agents,
            "num_episodes" : self.number_of_episodes,
            "base_directory" : evaluations_directory,
            "artifact_relative_directory" : "evaluation",
            "create_new_directory" : True # if there were other play made, create a new one
        })
        
        env = self.__generalize_get_environment(env, rl_player)
        
        if env is not None:
            rl_player.pass_input({"environment" : env})            
        
        rl_player.run()

        if rl_player_will_be_generated: # if the player will be generated, might as well save the configuration for later consultation of it
            save_configuration(rl_player, rl_player.get_artifact_directory(), save_exposed_values=True, ignore_defaults=False)
        
        return rl_player