

from automl.rl.evaluators.rl_single_agent_evaluator import RlSingleAgentEvaluator
from automl.rl.evaluators.rl_evaluator_player import EvaluatorWithPlayer
from automl.rl.agent.agent_components import AgentSchema
from automl.rl.policy.random_policy import RandomPolicy
from automl.rl.rl_pipeline import RLPipelineComponent
from automl.rl.rl_player.rl_player import RLPlayer
from automl.core.input_management import ParameterSignature
from automl.rl.policy.policy import Policy




class AgentVsAgents(RlSingleAgentEvaluator, EvaluatorWithPlayer):

    '''
    Evaluates a single agent vs other agents
    '''

    def _proccess_input_internal(self):

        super()._proccess_input_internal()

        self.base_evaluator.pass_input({"value_to_use" : f"{self.agent_name}_reward"})
    

    def _initialize_other_agent(self, rl_player : RLPlayer, agent_name : str, agent : AgentSchema, component_to_evaluate : RLPipelineComponent =None):
        pass


    def _initialize_player_to_run(self, agents : dict[str, AgentSchema], device, evaluations_directory, env, seed_for_player=None, component_to_evaluate : RLPipelineComponent = None):

        rl_player = super()._initialize_player_to_run(agents, device, evaluations_directory, env, seed_for_player, component_to_evaluate)

        agents = {**agents} # this is to be sure we're not changing a used dict

        for agent_name, agent in agents.items():

            if agent_name != self.agent_name:
                agents[agent_name] = self._initialize_other_agent(rl_player, agent_name, agent, component_to_evaluate)

        rl_player.pass_input({"agents" : agents})

        return rl_player
    


class AgentVsAgentsWithPolicy(AgentVsAgents):

    '''
    Evaluates the reward of the agent when the other agents are following a random policy
    '''

    parameters_signature = {
        "policy_type_for_others" : ParameterSignature(default_value=RandomPolicy),

    }
    

    

    def _proccess_input_internal(self):
        
        super()._proccess_input_internal()

        self.policy_type_for_others : type[Policy] = self.get_input_value("policy_type_for_others")


    def _initialize_other_agent(self, rl_player : RLPlayer, agent_name, agent : AgentSchema, component_to_evaluate : RLPipelineComponent =None):
                
        new_agent_input = {**agent.input}

        new_agent_input["policy"] = self.policy_type_for_others()
                
        new_agent = rl_player.initialize_child_component(type(agent), new_agent_input)

        return new_agent
