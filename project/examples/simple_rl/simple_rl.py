from automl.rl_components.agent_components import AgentComponent
from automl.rl_components.model_components import ConvModelComponent
from automl.rl_components.optimizer_components import AdamOptimizer

import torch.optim as optim


def initialize_agents_components(lg, env, learning_rate=0.001, state_memory_size=1, agents_input = {}):

    agents = {}
    
    agentId = 1        
    for agent in env.agents(): #worth remembering that the order of the initialization of the agents is defined by the environment
        
        agent_input = {**agents_input} #the input of the agent, with base values defined by "agents_input"

        
        agent_name = "agent_" + str(agentId)
        agent_input["name"] = agent_name
        
        agent_logger = lg.openChildLog(logName=agent_name)
        agent_input["logger"] = agent_logger
                
        state = env.observe(agent)
        lg.writeLine("State for agent " + agent_name + " has shape: Z: " + str(len(state)) + " Y: " + str(len(state[0])) + " X: " + str(len(state[0][0])))
        
        z_input_size = len(state) * state_memory_size
        y_input_size = len(state[0])
        x_input_size = len(state[0][0])
        
        n_actions = env.action_space(agent).n
        print(f"Action spac of agent {agent}: {env.action_space(agent)}")
        
        agent_model = ConvModelComponent(input={"board_x" : x_input_size, "board_y" : y_input_size, "board_z" : z_input_size, "output_size" : n_actions})
        agent_input["policy_model"] = agent_model
        agent_model.proccess_input()  
        
        agent_opimizer = AdamOptimizer( {"learning_rate" : learning_rate } )
        agent_input["optimizer"] = agent_opimizer         
                
        agents[agent] = AgentComponent(input=agent_input)
        
        lg.writeLine("Created agent in training " + agent_name)
        
        agentId += 1
        
    lg.writeLine("Initialized " + str(agents) + " agents")
    
    return agents

def initialize_rl_trainer_component(lg):
    pass