from automl.basic_components.state_management import StatefulComponent
from automl.component import InputSignature, requires_input_proccess

from automl.core.advanced_input_management import ComponentInputSignature
from automl.loggers.logger_component import ComponentWithLogging
from automl.ml.models.model_components import ModelComponent
from automl.ml.optimizers.optimizer_components import OptimizerSchema, AdamOptimizer
from automl.rl.learners.learner_component import LearnerSchema

import torch

from automl.rl.policy.policy import Policy


class QLearnerSchema(LearnerSchema, ComponentWithLogging):
    
    '''
    This represents a Deep Q Learner
    It has decouples the prediction of the q values by having not only having the policy network but also a target network
    '''

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
                        }    
    

    
    def _proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._proccess_input_internal()
                
        
        
        
    def initialize_optimizer(self):
        
        self.optimizer : OptimizerSchema = self.get_input_value("optimizer")        
        self.optimizer.pass_input({"model" : self.model})

    
    # EXPOSED METHODS --------------------------------------------------------------------------

    def _apply_model_prediction_given_state_action_pairs(self, state_batch, action_batch):

        '''Returns the values predicted by the current model and the values for the specific actions that were passed'''
        pass
    

    def _apply_value_prediction_to_next_state(self, next_state_batch, done_batch, reward_batch, discount_factor):

        '''
        Returns the predicted values for the next state
        
        They are given by appying the Q function to them and then chosing the next 

        '''
        pass
    

    def _calculate_chosen_actions_correct_q_values(self, next_state_v_values, discount_factor, reward_batch):
        pass
    
    def _optimize_with_predicted_model_values_and_correct_values(self, predicted_values, correct_values):
        pass
    
    def _learn(self, trajectory, discount_factor) -> None:
        
        super()._learn(trajectory, discount_factor)

        self.batch_size = len(trajectory[0])

        state_batch, action_batch, next_state_batch, reward_batch, done_batch = self.interpret_trajectory(trajectory)
                    
        predicted_actions_values, state_action_values = self._apply_model_prediction_given_state_action_pairs(state_batch, action_batch) 

        next_state_q_values, next_state_v_values = self._apply_value_prediction_to_next_state(next_state_batch, done_batch, reward_batch, discount_factor)

        correct_q_values_for_chosen_action = self._calculate_chosen_actions_correct_q_values(next_state_v_values, discount_factor, reward_batch)
                        
        self._optimize_with_predicted_model_values_and_correct_values(state_action_values.squeeze(-1), correct_q_values_for_chosen_action)  
        

        


class DeepQLearnerSchema(QLearnerSchema):
    
    '''
    This represents a Deep Q Learner
    It has decouples the prediction of the q values by having not only having the policy network but also a target network
    '''

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
                               
                        "target_update_rate" : InputSignature(
                            default_value=0.05,
                            custom_dict={"hyperparameter_suggestion" : [ "float", {"low": 0.5, "high": 1.0 }]}
                            ),

                        "target_update_learn_interval" : InputSignature(default_value=1, description="How many optimization times before we update the target model",
                                                                        custom_dict={"hyperparameter_suggestion" : [ "int", {"low": 1, "high": 10 }]}),
                        
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
    
    def _proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._proccess_input_internal()
                
        self.device = self.get_input_value("device")
        
        self.TAU = self.get_input_value("target_update_rate") #the update rate of the target network
        
        self.target_update_learn_interval = self.get_input_value("target_update_learn_interval")
        
        self.policy : Policy = self.agent.get_policy()
        
        self.model = self.policy.model
        
        self.number_optimizations_done = 0
        
        self.initialize_target_network()
        self.initialize_optimizer()
        


    def initialize_target_network(self):
        
        if self.values["target_network"] != 0:
                        
            self.lg.writeLine("Target network found already in exposed values, using that...")
            self.target_net : ModelComponent = self.values["target_network"]

        elif "target_network" in self.input.keys():
            
            self.lg.writeLine("There was already a target network defined in the input")
            self.target_net : ModelComponent  = self.get_input_value("target_network")
            self.values["target_network"] = self.target_net
            
        else:
            self.lg.writeLine("No target network previously defined, creating a new one...")
            
            self.target_net : ModelComponent = self.model.clone(
                save_in_parent=False,
                input_for_clone=
                    {"name" : "target_network", 
                    "base_directory" : self,
                    "create_new_directory" : True}, 
                is_deep_clone=True) #the target network has the same initial parameters as the policy being trained
            
            self.define_component_as_child(self.target_net)
            
            self.values["target_network"] = self.target_net

        
        
    def initialize_optimizer(self):
        
        self.optimizer : OptimizerSchema = self.get_input_value("optimizer")        
        self.optimizer.pass_input({"model" : self.model})

    
    # EXPOSED METHODS --------------------------------------------------------------------------

    def _apply_model_prediction_given_state_action_pairs(self, state_batch, action_batch):

        '''Returns the values predicted by the current model and the values for the specific actions that were passed'''

        predicted_actions_values = self.model.predict(state_batch)
        predicted_values_for_actions = predicted_actions_values.gather(1, action_batch) 

        return predicted_actions_values, predicted_values_for_actions
    

    def _apply_value_prediction_to_next_state(self, next_state_batch, done_batch, reward_batch, discount_factor):

        '''
        Returns the predicted values for the next state
        
        They are given by appying the Q function to them and then chosing the next 

        '''

        with torch.no_grad():
            next_state_q_values = self.target_net.predict(next_state_batch) # it returns the maximum q-action values of the next action
            
            next_state_v_values = next_state_q_values.max(1).values
            next_state_v_values = next_state_v_values * (1 - done_batch)

        return next_state_q_values, next_state_v_values
    

    def _calculate_chosen_actions_correct_q_values(self, next_state_v_values, discount_factor, reward_batch):

        correct_q_values_for_chosen_action = next_state_v_values * discount_factor + reward_batch

        return correct_q_values_for_chosen_action.detach()
    
    def _optimize_with_predicted_model_values_and_correct_values(self, predicted_values, correct_values):
    
        #Optimizes the model given the optimizer defined
        self.optimizer.optimize_model(predicted_values, correct_values)        
        
        self.number_optimizations_done += 1
    
    def _learn(self, trajectory, discount_factor) -> None:
        
        super()._learn(trajectory, discount_factor)

        if self.number_optimizations_done % self.target_update_learn_interval == 0:
            self.update_target_model()

        
        
    @requires_input_proccess            
    def update_target_model(self):
        
        self.target_net.update_model_with_target(self.model, self.TAU)



class DoubleDeepQLearnerSchema(DeepQLearnerSchema):


    def _apply_value_prediction_to_next_state(self, next_state_batch, done_batch, reward_batch, discount_factor):


        with torch.no_grad():

            q_values_that_would_be_given_next = self.model.predict(next_state_batch)
            actions_that_would_be_chosen_next = q_values_that_would_be_given_next.argmax(1, keepdim=True)

            next_state_q_values = self.target_net.predict(next_state_batch) # it returns the maximum q-action values of the next action
            
            next_state_v_values = next_state_q_values.gather(1, actions_that_would_be_chosen_next).squeeze(1)
            next_state_v_values = next_state_v_values * (1 - done_batch)

        return next_state_q_values, next_state_v_values