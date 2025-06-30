

from automl.component import requires_input_proccess
from automl.core.input_management import InputSignature
from automl.rl.agent.agent_components import AgentSchema
from automl.utils.shapes_util import torch_zeros_for_space
import torch


class AgentSchemaWithStateMemory(AgentSchema):


    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = { 
                       
                       "state_memory_size" : InputSignature(default_value=2, description="This makes the agent remember previous states of the environment and concatenates them"),

                    }

        
    def proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super().proccess_input_internal()
        
        self.name = self.input["name"]                
        self.device = self.input["device"]
    
        self.initialize_state_memory()
        self.initialize_policy_with_state_memory()

        


    def initialize_state_memory(self):
            
        self.state_shape = self.input["state_shape"]
        self.state_memory_size = self.input["state_memory_size"]
        
        if self.state_memory_size <= 1:
            raise Exception("State memory size must be greater than 1 to use this agent schema")
        
        self.model_input_shape = tuple(self.state_shape for _ in range(self.state_memory_size))

        self.state_memory_size = self.input["state_memory_size"]
        
        self.state_memory = torch_zeros_for_space(self.model_input_shape, device=self.device) # makes a list of tensors for the state_memory, using them to store memory of the states

        self.lg.writeLine(f"Initializing agent with more than one state memory size ({self.state_memory_size})")

        if self.input["state_shape"] == '':
            raise Exception("More than one state memory size and undefined model input shape")

        self.state_length = self.input["state_shape"][2]

        self.lg.writeLine(f"State length is {self.state_length}")
     
        if torch.cuda.is_available():   
            self.lg.writeLine(f"Initialized state memory, Cuda memory allocated: {torch.cuda.memory_allocated()}, Cuda memory reserved: {torch.cuda.memory_reserved()}, Cuda memory available: {torch.cuda.mem_get_info()}")
        
        
    def initialize_policy_with_state_memory(self):
        
        self.lg.writeLine("Initializing policy...")
        
        self.model_input_shape = self.input["state_shape"]
        self.model_output_shape = self.input["action_shape"]
        
        self.model_input_shape = (self.state_memory_size, self.model_input_shape)
                
        self.policy.pass_input({"state_shape" : self.model_input_shape,
                               "action_shape" : self.model_output_shape,})

    
    # EXPOSED TRAINING METHODS -----------------------------------------------------------------------------------
    
    def get_policy(self):
        return self.policy
    
    @requires_input_proccess
    def policy_predict(self, state):
        
        '''makes a prediction based on the new state for a new action, using the current memory'''
        
        self.update_state_memory(state) #updates memory with the new state
        
        possible_state_memory = self.get_state_memory_with_new(state)
        
        return self.policy.predict(torch.cat([element for element in possible_state_memory])).item()
        
    
    # STATE MEMORY --------------------------------------------------------------------
            
    
    @requires_input_proccess
    def observe_transiction_to(self, new_state, action, reward):
        
        '''Makes agent observe and remember a transiction from a state to another'''
        
        prev_state_memory = torch.cat([element for element in self.state_memory])
        
        self.update_state_memory(new_state)
        
        next_state_memory = torch.cat([element for element in self.state_memory])
                
        self.memory.push(prev_state_memory, action, next_state_memory, reward)
        
        
    
    @requires_input_proccess
    def reset_state_in_environment(self, initial_state): #setup memory shared accross agents
        
        super().reset_agent_in_environment(initial_state)
        
        if self.state_memory_size > 1:
            
            for i in range(self.state_memory_size):
                self.state_memory[i] = initial_state
            
        else:
                        
            self.state_memory = initial_state
         
             
    @requires_input_proccess    
    def update_state_memory(self, new_state): #update memory shared accross agents
        self.state_memory = self.get_state_memory_with_new(new_state)
        
       
    @requires_input_proccess     
    def get_state_memory_with_new(self, new_state):
        
        '''Returns a new state memory with the new state added, shifting the previous states'''
        
        new_state_memory = [state  for state in self.state_memory]
        
        if self.state_memory_size > 1:   
            
            for i in range(1, self.state_memory_size):
                new_state_memory[i - 1] = new_state_memory[i]
            
            new_state_memory[self.state_memory_size - 1] = new_state

            return new_state_memory
        
        else:
            
            return new_state
         
        
        

