

from automl.component import requires_input_proccess
from automl.core.input_management import InputSignature
from automl.ml.memory.torch_memory_component import TorchMemoryComponent
from automl.rl.agent.agent_components import AgentSchema
from automl.utils.shapes_util import torch_zeros_for_space
import torch


class AgentSchemaWithStateMemory(AgentSchema):
    
    '''An agent which state has memory of previous states, such as a robot remembering its previous positions'''


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
        self.initialize_memory_with_state_memory()

        


    def initialize_state_memory(self):
        
            
        self.state_shape = self.input["state_shape"]
        self.state_memory_size = self.input["state_memory_size"]
        
        print(f"State shape before change: {self.state_shape}")
        
        if self.state_memory_size <= 1:
            raise Exception("State memory size must be greater than 1 to use this agent schema")
        
        self.model_input_shape = (self.state_memory_size, *self.state_shape)
        
        print(f"State shape after change: {self.model_input_shape}")

        self.state_memory_size = self.input["state_memory_size"]
        
        self.state_memory = torch_zeros_for_space(self.model_input_shape, device=self.device) # makes a list of tensors for the state_memory, using them to store memory of the states
        self.temp_cache_state_memory = torch_zeros_for_space(self.model_input_shape, device=self.device) # a reserved memory space to store 

        self.lg.writeLine(f"Initializing agent with more than one state memory size ({self.state_memory_size})")

        if self.input["state_shape"] == '':
            raise Exception("More than one state memory size and undefined model input shape")

        self.state_length = self.input["state_shape"][2]

        self.lg.writeLine(f"State length is {self.state_length}")
     
        if torch.cuda.is_available():   
            self.lg.writeLine(f"Initialized state memory, Cuda memory allocated: {torch.cuda.memory_allocated()}, Cuda memory reserved: {torch.cuda.memory_reserved()}, Cuda memory available: {torch.cuda.mem_get_info()}")
        
        else:
            self.lg.writeLine(f"Initialized state memory, Cuda not available, using CPU memory")
        
        
    def initialize_policy_with_state_memory(self):
        
        self.lg.writeLine("Initializing policy...")
                
        self.policy.pass_input({"state_shape" : self.model_input_shape,
                               "action_shape" : self.model_output_shape,})
        
        
    def initialize_memory_with_state_memory(self):
                
        if isinstance(self.memory, TorchMemoryComponent):
            
            self.memory.pass_input({"state_dim" : self.model_input_shape, 
                                    "action_dim" : self.model_output_shape, 
                                    "device" : self.device}
                                )

    
    # EXPOSED TRAINING METHODS -----------------------------------------------------------------------------------
    
    def get_policy(self):
        return self.policy
    
    @requires_input_proccess
    def policy_predict(self, state):
        
        '''
        Makes a prediction based on the new state for a new action, using the current memory
        Note that this does not observe the transition
        '''
        
        self.update_state_memory(state) #updates memory with the new state
        
        possible_state_memory = self.get_state_memory_with_new(state)
        
        return self.policy.predict(possible_state_memory).item()
        
    
    # STATE MEMORY --------------------------------------------------------------------
            
    
    @requires_input_proccess
    def observe_transiction_to(self, new_state, action, reward):
        
        '''Makes agent observe and remember a transiction from a state to another'''
        
        self.temp_cache_state_memory.copy_(self.state_memory)
        
        self.update_state_memory(new_state)
                        
        self.memory.push(self.temp_cache_state_memory, action, self.state_memory, reward)
        
        
    
    @requires_input_proccess
    def reset_state_in_environment(self, initial_state : torch.Tensor): #setup memory shared accross agents
        
        super().reset_agent_in_environment(initial_state)
                    
        self.state_memory = initial_state.unsqueeze(0).expand(self.state_memory_size, *self.state_shape).clone()

            
         
             
    @requires_input_proccess    
    def update_state_memory(self, new_state): #update memory shared accross agents
        self.state_memory = self.get_state_memory_with_new(new_state)
        
       
    @requires_input_proccess     
    def get_state_memory_with_new(self, new_state):
        '''
        Returns a new state memory tensor with the new_state added,
        shifting the previous states to the left.
        
        Note that this tensor is cloned and not used anymore by the agent, so it can be safely used
        '''
        # shift previous memory left by one position
        self.temp_cache_state_memory[:-1].copy_(self.state_memory[1:])

        # insert new state into the last position
        self.temp_cache_state_memory[-1].copy_(new_state)

        return self.temp_cache_state_memory.clone()
    
         
        
        

