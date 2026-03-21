
from automl.component import Component, requires_input_proccess
from automl.basic_components.evaluator_component import EvaluatorComponent
from automl.rl.rl_pipeline import RLPipelineComponent
from automl.core.advanced_input_management import ComponentListParameterSignature
from automl.rl.evaluators.rl_single_agent_evaluator import RlSingleAgentEvaluator
from automl.rl.evaluators.rl_component_evaluator import RLPipelineEvaluator
from automl.rl.agent.agent_components import AgentSchema


class RLAgentIterEvaluator(RLPipelineEvaluator):
    
    '''
    An evaluator specific for RL pipelines
    
    '''
    
    parameters_signature = {
        "single_agent_evaluators" : ComponentListParameterSignature()
    }
    

    def _proccess_input_internal(self):
        
        super()._proccess_input_internal()

        self.single_agent_evaluators : list[RlSingleAgentEvaluator] = self.get_input_value("single_agent_evaluators")
        


    # EVALUATION -------------------------------------------------------------------------------

    def make_evaluation_for_single_agent_evaluator(self, component_to_evaluate : RLPipelineComponent, agents_dict : dict[str, AgentSchema], single_agent_evaluator : RlSingleAgentEvaluator, directory_to_store_evaluation_prefix : str):
    
        agent_evaluations = {}

        for agent_name in agents_dict.keys():
            single_agent_evaluator.pass_input({"directory_to_store_evaluation_prefix" : f"{directory_to_store_evaluation_prefix}\\{agent_name}"})
            single_agent_evaluator.pass_input({"agent_name" : agent_name})
            agent_evaluations[agent_name] = single_agent_evaluator.evaluate(component_to_evaluate)


        resutls = [agent_evaluation["result"] for agent_evaluation in agent_evaluations.values()]

        agent_evaluations["result"] = sum(resutls) / len(resutls)

        return agent_evaluations
    

    @requires_input_proccess
    def _evaluate(self, component_to_evaluate : RLPipelineComponent):

        agents_dict = component_to_evaluate.get_agents()
        
        agent_evaluations_for_evaluators = {}

        if len(self.single_agent_evaluators) > 0:

            for single_agent_evaluator in self.single_agent_evaluators:
                agent_evaluations_for_evaluators[single_agent_evaluator.name] = self.make_evaluation_for_single_agent_evaluator(
                    component_to_evaluate=component_to_evaluate,
                    agents_dict=agents_dict,
                    single_agent_evaluator=single_agent_evaluator,
                    directory_to_store_evaluation_prefix=f"evaluations\\{single_agent_evaluator.name}"
                )

            resutls = [agent_evaluation["result"] for agent_evaluation in agent_evaluations_for_evaluators.values()]

            agent_evaluations_for_evaluators["result"] = sum(resutls) / len(resutls)

        else:
            
            single_agent_evaluator = self.single_agent_evaluators[0]

            agent_evaluations_for_evaluators = self.make_evaluation_for_single_agent_evaluator(
                    component_to_evaluate=component_to_evaluate,
                    agents_dict=agents_dict,
                    single_agent_evaluator=single_agent_evaluator,
                    directory_to_store_evaluation_prefix=f"evaluations"
                )

        return agent_evaluations_for_evaluators