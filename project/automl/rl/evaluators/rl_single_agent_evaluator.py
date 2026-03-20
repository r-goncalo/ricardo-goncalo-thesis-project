
from automl.component import Component, requires_input_proccess
from automl.basic_components.evaluator_component import EvaluatorComponent
from automl.rl.rl_pipeline import RLPipelineComponent
from automl.rl.evaluators.rl_component_evaluator import RLPipelineEvaluator
from automl.core.input_management import ParameterSignature


class RlSingleAgentEvaluator(RLPipelineEvaluator):
    
    '''
    Evaluates a single agent
    '''
    
    parameters_signature = {
        "agent_name" : ParameterSignature(),
    }
    

    def _proccess_input_internal(self):
        
        super()._proccess_input_internal()

        self.agent_name : str = self.get_input_value("agent_name")
        