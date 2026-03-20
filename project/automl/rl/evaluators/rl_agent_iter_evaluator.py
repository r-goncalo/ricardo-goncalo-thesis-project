
from automl.component import Component, requires_input_proccess
from automl.basic_components.evaluator_component import EvaluatorComponent
from automl.rl.rl_pipeline import RLPipelineComponent
from automl.core.advanced_input_management import ComponentParameterSignature
from project.automl.rl.evaluators.rl_single_agent_evaluator import RlSingleAgentEvaluator


class RLAgentIterEvaluator(EvaluatorComponent):
    
    '''
    An evaluator specific for RL pipelines
    
    '''
    
    parameters_signature = {
        "single_agent_evaluator" : ComponentParameterSignature()
    }
    

    def _proccess_input_internal(self):
        
        super()._proccess_input_internal()

        self.single_agent_evaluator : RlSingleAgentEvaluator = self.get_input_value("single_agent_evaluator")
        


    # EVALUATION -------------------------------------------------------------------------------
    
    @requires_input_proccess
    def _evaluate(self, component_to_evaluate : RLPipelineComponent):

        agents_dict = component_to_evaluate.get_agents()
        
        agent_evaluations = {}

        for agent_name in agents_dict.keys():
            self.single_agent_evaluator.pass_input({"directory_to_store_evaluation_prefix" : f"evaluations\\{agent_name}"})
            agent_evaluations[agent_name] = self.single_agent_evaluator.evaluate_agent(component_to_evaluate, agent_name)

        resutls = [agent_evaluation["result"] for agent_evaluation in agent_evaluations.values()]

        agent_evaluations["result"] = sum(resutls) / len(resutls)

        return agent_evaluations