from automl.component import Component, ParameterSignature, requires_input_proccess


from automl.core.advanced_input_management import ComponentParameterSignature, ComponentListParameterSignature
from automl.fundamentals.acessories import AcessoryComponent
from automl.ml.memory.memory_utils import interpret_unit_values, interpret_values
from automl.rl.agent.agent_components import AgentSchema
import torch

class LearnerSchema(Component):
        
    parameters_signature = {
        "agent" : ParameterSignature(),
        "optimizations_per_learn" : ParameterSignature(default_value=1,custom_dict={
                                    "hyperparameter_suggestion" : ("int", {"low" : 1, "high" : 32})
                                }),

        "learning_acessories" : ComponentListParameterSignature(mandatory=False),

        "agent_trainer" : ComponentParameterSignature(mandatory=False),


    }
        
    def _proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._proccess_input_internal()
        
        self.agent : AgentSchema = self.get_input_value("agent")

        self.optimizations_per_learn : int = self.get_input_value("optimizations_per_learn")

        self.learning_acessories : list[AcessoryComponent] = self.get_input_value("learning_acessories")

        if self.learning_acessories is None:
            self.learning_acessories = []

        else:
            for acessory in self.learning_acessories:
                acessory.pass_input({"affected_component" : self})

        self.agent_trainer = self.get_input_value("agent_trainer")



    
    def _learn(self, trajectory, discount_factor):
        pass
        
    @requires_input_proccess
    def learn(self, trajectory, discount_factor) -> None:
        
        '''
            Learns the policy using the current policy and the trajectory it is supposed to learn
            
            Remember that On-Policy algorithms expect the trajectory to have been generated
            
            Args:
                trajectory: batch of transitions [ (all states), (all actions), (all next states), (all rewards) ]
                
        '''

        for acessory in self.learning_acessories:
            acessory.pre_fun()
        
        for _ in range(self.optimizations_per_learn):
            values = self._learn(trajectory, discount_factor)

        for acessory in self.learning_acessories:
            acessory.pos_fun(values)


    
    def interpret_trajectory(self, trajectory):

        interpreted_trajectory = {**trajectory}
        
        interpreted_trajectory["observation"] = interpret_values(trajectory["observation"], self.device)

        interpreted_trajectory["action"] = interpret_values(trajectory["action"], self.device)

        interpreted_trajectory["next_observation"] = interpret_values(trajectory["next_observation"], self.device)
            
        interpreted_trajectory["reward"] = interpret_unit_values(trajectory["reward"], self.device)

        interpreted_trajectory["done"]  = interpret_unit_values(trajectory["done"], self.device)
            
        return interpreted_trajectory
    
    
    def _non_final_states_mask(self, next_observation_batch):
        
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              next_observation_batch)), dtype=torch.bool)

        
        return non_final_mask