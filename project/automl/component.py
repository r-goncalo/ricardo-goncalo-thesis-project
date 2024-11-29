
from enum import Enum
from abc import ABC, abstractmethod

#Every component may receieve input and return output

class Component: # a component that receives and verifies input 
    
    # a dictionary with { "input_name" : (default_value, validity verification) }
    # if default_value is None, an exception error is raised when input is missing
    # if validity verification function is not none, it will be applied to the input value
    # the actual input values will be saved in self.input
    input_signature = {}
    
    def __init__(self, input=None): #we can immediatly receive the input
        
        self.input = {} #the input will be a dictionary    
        
        if input != None:
            self.pass_and_proccess_input(input)
        
        
        self.output = {} #output, if any, will be a dictionary

    def get_output(self): #return output
        return self.output
    
    def pass_and_proccess_input(self, input : dict):
        self.pass_input(input)
        self.proccess_input()
    
    def pass_input(self, input: dict): # pass input to this component
        for passed_key in input.keys():
            self.input[passed_key] = input[passed_key]


    def proccess_input(self): #verify the input to this component and add default values, to the self.input dict
        
        passed_keys = self.input.keys()
        
        for input_key in self.input_signature.keys():
            
            (default_value, possible_types) = self.input_signature[input_key]
                        
            if input_key in passed_keys: #if this value was in input
                
                input_value = self.input[input_key] #get the value passed
                    
                if not verify_if_correct_type(input_key, input_value, possible_types):
                    raise Exception(f"Input with key '{input_key}' with type {type(input_value)} is not of any of the available types {possible_types}")
                                    
            else:
                                
                if default_value == None:
                    raise Exception(f"Did not set input for value {input_key} and has no default value")
                
                else:
                    self.input[input_key] = default_value  #the value used will be the default value
                    

def verify_if_correct_type(input_key, input_value, validity_verification):
    
    if validity_verification == None: #if there was no specified validity_verification
        return True 
    
    is_a_correct_type = validity_verification(input_value)
    
    if not isinstance(is_a_correct_type, bool): #validity_verification must return bool
        raise Exception(f"Validity verification on key '{input_key}' returned a type other than bool") 
            
    return is_a_correct_type
            
    


#Some components are executables

class ExecComponent(Component):
    
    class State(Enum):
        IDLE = 0
        RUNNING = 1
        OVER = 2
        ERROR = 4

    @abstractmethod
    def algorithm(self): #the algorithm of the component
        pass
    
    def pre_algorithm(self): # a component may extend this for certain behaviours
        self.running_state = ExecComponent.State.RUNNING
        
    def pos_algorithm(self): # a component may extend this for certain behaviours
        self.running_state = ExecComponent.State.OVER 
    
    def pass_input_and_exec(self, input : dict): #
        
        self.pass_and_proccess_input(input) #before the algorithm starts running, we define the input    
        return self.execute()
    
    def execute(self): #the universal execution flow for all components
         
        try:
            self.pre_algorithm()
            self.algorithm()
            self.pos_algorithm()
            return self.get_output()
        
        except Exception as e:
            self.running_state = ExecComponent.State.ERROR
            self.onException(e)
    
    def onException(self, exception):
        raise exception
        
    
    
