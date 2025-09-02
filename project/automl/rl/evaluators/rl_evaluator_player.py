
import os
from automl.component import Component, requires_input_proccess
from automl.core.advanced_input_management import ComponentInputSignature
from automl.rl.environment.environment_sampler import EnvironmentSampler
from automl.utils.json_component_utils import gen_component_from
from automl.loggers.logger_component import ComponentWithLogging
from automl.rl.evaluators.rl_component_evaluator import RLPipelineEvaluator
from automl.rl.rl_pipeline import RLPipelineComponent
from automl.loggers.result_logger import ResultLogger, aggregate_results_logger

from automl.core.input_management import InputSignature

from automl.rl.rl_player.rl_player import RLPlayer
from automl.rl.evaluators.rl_std_avg_evaluator import LastValuesAvgStdEvaluator

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
        
        "last_evaluation" : {}
    
    }
    

    def proccess_input_internal(self):
        
        super().proccess_input_internal()
        
        self.base_evaluator : RLPipelineEvaluator = ComponentInputSignature.get_component_from_input(self, "base_evaluator")
        self.number_of_episodes = self.input["number_of_episodes"]
        self.number_of_evaluations = self.input["number_of_evaluations"]
        
        self._setup_environment()


    def _setup_environment(self):
        
        self.env = None
        
        if "environment" in self.input.keys():
            self.env = ComponentInputSignature.get_component_from_input(self, "environment")
            
            
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


        return self._evaluate_agents(component_to_evaluate.agents, component_to_evaluate.device, evaluations_directory, env)
        
        
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

        env = gen_component_from(env)

        if isinstance(env, EnvironmentSampler): # if a sampler was passed
            
            env = env.sample()
            
            env.pass_input({
                "base_directory" : rl_player.get_artifact_directory()
            })
        
        return env
       

        
    def _evaluate_agents(self, agents, device, evaluations_directory, env=None):
        
        path_of_players = []
        environment_names = []
                
        for i in range(self.number_of_evaluations): # evaluate plays and store their paths 
            path_of_players.append(self._run_play_to_evaluate(agents, device, evaluations_directory, env))
            environment_names.append(f"environment_{i}")

        results_logger = aggregate_results_logger(path_of_players, evaluations_directory, ("environment", environment_names))
        
        print("HERE")
        
        return self.base_evaluator.evaluate(results_logger)
    

    def _run_play_to_evaluate(self, agents, device, evaluations_directory, env):
                
        rl_player : RLPlayer = gen_component_from(self.input["rl_player_definition"])


        rl_player.pass_input({
            "agents" : agents,
            "num_episodes" : self.number_of_episodes,
            "device" : device,
            "base_directory" : evaluations_directory,
            "artifact_relative_directory" : "evaluation",
            "create_new_directory" : True
        })
        
        env = self.__generalize_get_environment(env, rl_player)
        
        if env is not None:
            rl_player.pass_input({"environment" : env})    
        
        
        
        rl_player.run()
        
        return rl_player.get_artifact_directory()