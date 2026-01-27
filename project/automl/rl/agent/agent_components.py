
# DEFAULT COMPONENTS -------------------------------------

from automl.core.advanced_input_management import ComponentInputSignature

from automl.rl.policy.policy import Policy

from automl.basic_components.state_management import StatefulComponent

from automl.loggers.logger_component import ComponentWithLogging

# ACTUAL AGENT COMPONENT ---------------------------

from automl.component import InputSignature, requires_input_proccess
from automl.fundamentals.translator.translator import Translator

from automl.utils.shapes_util import torch_zeros_for_space


def no_proccess_state_for_agent(state):
    return state


class AgentSchema(ComponentWithLogging, StatefulComponent):


    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = { 
                        "name" : InputSignature(),
                       "device" : InputSignature(get_from_parent=True, ignore_at_serialization=True),
                                   
                       "state_shape" : InputSignature(default_value='', description='The shape received by the model, only used when the model was not passed already initialized'),
                       "action_shape" : InputSignature(default_value='', description='Shape of the output of the model, only used when the model was not passed already'),
                                              
                       "policy" : ComponentInputSignature(
                            priority=100, description="The policy to use for the agent, if not defined it will be created using the policy_class and policy_input"
                       ),
                    
                    "state_translator" : ComponentInputSignature(mandatory=False)

                       
                    }
        
    def _proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._proccess_input_internal()

        self.lg.writeLine(f"Processing agent input with values {self.values}\n")
        
        self.device = self.get_input_value("device")
    
        self.state_shape = self.get_input_value("state_shape") #shape of the state as the environment sees it
                
        self.model_input_shape = self.state_shape #shape of the state as it is processed by the model of the policy
        
        self.model_output_shape = self.get_input_value("action_shape") #shape of the action


        self.initialize_state_translator()
        self.initialize_state_memory()
        self.initialize_policy()
        self.initialize_necessary_cache()

        self.lg.writeLine(f"Finished processing basic agent input with values {self.values}\n")

    
    def initialize_state_translator(self):

        self.state_translator : Translator = self.get_input_value("state_translator")

        if self.state_translator is not None:

            self.state_translator.pass_input({"original_shape" : self.state_shape})

            self.state_translator.proccess_input_if_not_proccesd()

            self.lg.writeLine(f"Agent has state translator: {self.state_translator}")
            self.proccess_env_state = self.state_translator.translate_state
            
            old_model_input_shape = self.model_input_shape
            self.model_input_shape = self.state_translator.get_shape(self.model_input_shape)

            self.lg.writeLine(f"Model input shape was translated from {old_model_input_shape} to {self.model_input_shape}")


        else:
            self.lg.writeLine(f"Agent has no state translator")
            self.proccess_env_state = no_proccess_state_for_agent
        

    def initialize_state_memory(self):
        
        self.lg.writeLine(f"Initializing state memory, the agent will remember the last state for convenience in computations, but it will not be used by itself")
        self.state_memory = None #simple pointer to a to define state, this method is more to be extended
        

    
    def initialize_necessary_cache(self):
        
        self.lg.writeLine(f"Agent is allocating memory for its computations, using two tensors with shape {self.model_input_shape}")
        
        self.state_memory = self.allocate_tensor_for_state() # makes a list of tensors for the state_memory, using them to store memory of the states
        
    
    
    def allocate_tensor_for_state(self):
        
        return torch_zeros_for_space(self.model_input_shape, device=self.device)


        
    def initialize_policy(self):
        
        self.lg.writeLine("Initializing policy...")    
        
        self.policy : Policy = self.get_input_value("policy", look_in_value_with_key="policy", look_in_attribute_with_name="policy")
                
        self.policy.pass_input({"state_shape" : self.model_input_shape,
                               "action_shape" : self.model_output_shape,})
        
    
    
    
    
    # EXPOSED TRAINING METHODS -----------------------------------------------------------------------------------
    
    def get_policy(self):
        return self.policy
    
    @requires_input_proccess
    def policy_predict(self, state):
        
        '''makes a prediction based on the new state for a new action, using the current memory'''
        
        to_return = self.policy.predict(self.proccess_env_state(state))

        self.lg.writeLine(f"Policy chose action: {to_return}", file="actions.txt")

        return to_return
    
    @requires_input_proccess
    def policy_predict_with_memory(self):
        
        '''makes a prediction based on the new state for a new action, using the current memory'''
        
        return self.policy.predict(self.state_memory)
    
    @requires_input_proccess
    def call_policy_method(self, policy_method, state):
        
        '''calls the method of the policy with this Agent's state management strategy'''
        
        return policy_method(self.proccess_env_state(state))
                    
    
    @requires_input_proccess
    def policy_random_predict(self):
        to_return =  self.policy.random_prediction()

        self.lg.writeLine(f"Policy chose random action: {to_return}", file="actions.txt")

        return to_return
          
    
    # STATE MEMORY --------------------------------------------------------------------
            
    
    @requires_input_proccess
    def reset_agent_in_environment(self, initial_state): # resets anything the agent has saved regarding the environment
        self.update_state_memory(self.proccess_env_state(initial_state))
    

    @requires_input_proccess    
    def update_state_memory(self, new_state): #update memory shared accross agents
        self.state_memory.copy_(self.proccess_env_state(new_state))
        

    @requires_input_proccess
    def get_current_state_in_memory(self):
        
        '''
        Gets current state in memory, note this returns the actual tensor of the state memory
        It should be cloned if needed
        '''
                
        return self.state_memory
             
         
        
        

