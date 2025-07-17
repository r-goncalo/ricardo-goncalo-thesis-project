
import os
from automl.component import Component, requires_input_proccess
from automl.core.advanced_input_management import ComponentInputSignature
from automl.utils.json_component_utils import gen_component_from
from automl.loggers.logger_component import ComponentWithLogging
from automl.rl.evaluators.rl_component_evaluator import RLPipelineEvaluator
from automl.rl.rl_pipeline import RLPipelineComponent
from automl.loggers.result_logger import ResultLogger

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
        "environment" : ComponentInputSignature(mandatory=False),
    }
    

    def proccess_input_internal(self):
        
        super().proccess_input_internal()
        
        self.base_evaluator : RLPipelineEvaluator = ComponentInputSignature.get_component_from_input(self, "base_evaluator")
        self.number_of_episodes = self.input["number_of_episodes"]
        
        self._setup_environment()


    def _setup_environment(self):
        
        self.env = None
        
        if "environment" in self.input.keys():
            self.env = ComponentInputSignature.get_component_from_input(self, "environment")
            
    # EVALUATION -------------------------------------------------------------------------------
    
    def get_metrics_strings(self) -> list[str]:
        return [*super().get_metrics_strings(), *self.base_evaluator.get_metrics_strings()]
    
    @requires_input_proccess
    def evaluate(self, component_to_evaluate : RLPipelineComponent):
        
        
        rl_player : RLPlayer = gen_component_from(self.input["rl_player_definition"])
        
        if self.env is not None:
            env = self.env
        else:
            env = component_to_evaluate.env
        
        
        rl_player.pass_input({
            "environment" : env,
            "agents" : component_to_evaluate.agents,
            "num_episodes" : self.number_of_episodes,
            "device" : component_to_evaluate.device,
            "base_directory" : os.path.join(component_to_evaluate.get_artifact_directory(), "evaluations"),
            "artifact_relative_directory" : "evaluation"
        })
        
        
        rl_player.run()
        
        return self.base_evaluator.evaluate(rl_player)
        