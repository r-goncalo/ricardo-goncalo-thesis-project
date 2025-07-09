from automl.basic_components.state_management import StatefulComponent
from automl.component import Component, InputSignature, requires_input_proccess

from automl.core.advanced_input_management import ComponentInputSignature
from automl.loggers.logger_component import ComponentWithLogging
from automl.ml.models.model_components import ModelComponent
from automl.ml.optimizers.optimizer_components import OptimizerSchema, AdamOptimizer
from automl.rl.learners.learner_component import LearnerSchema

import torch

from automl.rl.policy.policy import Policy


class DeepQLearnerSchema(LearnerSchema, ComponentWithLogging):
    
    '''
    This represents a Deep Q Learner
    It has decouples the prediction of the q values by having not only having the policy network but also a target network
    '''

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
                               
                        "target_update_rate" : InputSignature(default_value=0.05),
                        "update_target_at_optimization" : InputSignature(default_value=True),
                        "device" : InputSignature(ignore_at_serialization=True),
                        
                        "optimizer" : ComponentInputSignature(
                            default_component_definition=(
                                AdamOptimizer,
                                {}
                            )
                            ),
                        
                        "target_network" : InputSignature(mandatory=False, possible_types=[ModelComponent], description="The target network if it already exists, a clone of the network of the policy")

                        }    
    
    exposed_values = {
        
        "target_network" : 0
        
    }
    
    def proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super().proccess_input_internal()
        
        self.agent : Component = self.input["agent"]
        
        self.device = self.input["device"]
        
        self.TAU = self.input["target_update_rate"] #the update rate of the target network
        
        self.update_target_at_optimization = self.input["update_target_at_optimization"]
        
        self.policy : Policy = self.agent.get_policy()
        
        self.model = self.policy.model
        
        self.initialize_target_network()
        self.initialize_optimizer()
        

    def initialize_target_network(self):
        
        if self.values["target_network"] != 0:
                        
            self.lg.writeLine("Target network found already in exposed values, using that...")
            self.target_net : ModelComponent = self.values["target_network"]

        elif "target_network" in self.input.keys():
            
            self.lg.writeLine("There was already a target network defined in the input")
            self.target_net : ModelComponent  = self.input["target_network"]
            self.values["target_network"] = self.target_net
            
        else:
            self.lg.writeLine("No target network previously defined, creating a new one...")
            
            self.target_net : ModelComponent = self.model.clone() #the target network has the same initial parameters as the policy being trained
            self.define_component_as_child(self.target_net)
            
            self.values["target_network"] = self.target_net
        
        
    def initialize_optimizer(self):
        
        self.optimizer : OptimizerSchema = ComponentInputSignature.get_component_from_input(self, "optimizer")        
        self.optimizer.pass_input({"model" : self.model})

    
    # EXPOSED METHODS --------------------------------------------------------------------------
    
    
    @requires_input_proccess
    def learn(self, trajectory, discount_factor) -> None:
        
        super().learn(trajectory, discount_factor)
        
        state_batch, _, next_state_batch, reward_batch = self._interpret_trajectory(trajectory)
            
        non_final_mask = self._non_final_states_mask(next_state_batch) 
                
        #predict the action we would take given the current state
        predicted_actions_values = self.model.predict(state_batch)
        predicted_actions_values, predicted_actions = predicted_actions_values.max(1)
        
        #compute the q values our target net predicts for the next_state (perceived reward)
        #if there is no next_state, we can use 0
                
        next_state_values = torch.zeros(len(trajectory[0]), device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net.predict(next_state_batch[non_final_mask]).max(1).values # it returns the maximum q-action values of the next action
            
        # Compute the expected Q values (the current reward of this state and the perceived reward we would get in the future)
        next_state_values.mul_(discount_factor).add_(reward_batch)
                
        #Optimizes the model given the optimizer defined
        self.optimizer.optimize_model(predicted_actions_values, next_state_values)        
        
        if self.update_target_at_optimization:
            self.update_target_model()
        
        
    @requires_input_proccess            
    def update_target_model(self):
        
        self.target_net.update_model_with_target(self.model, self.TAU)