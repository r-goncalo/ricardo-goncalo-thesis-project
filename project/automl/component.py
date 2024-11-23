
from enum import Enum
from abc import ABC, abstractmethod


class ComponentState(Enum):
    IDLE = 0
    RUNNING = 1
    OVER = 2
    ERROR = 4
    


class Component:
    
    def __init__(self):
        self.running_state = ComponentState.IDLE

    @abstractmethod    
    def algorithm(self): #the algorithm of the component
        pass
    
    def pre_algorithm(self):
        self.running_state = ComponentState.RUNNING
        
    def pos_algorithm(self):
        self.running_state = ComponentState.OVER 
    
    def execute(self): #the universal execution flow for all components
         
        try:
            self.pre_algorithm()
            self.algorithm()
            self.pos_algorithm()
            return self.getOutput()
        
        except Exception as e:
            self.running_state = ComponentState.ERROR
            self.onException(e)

    
    def getOutput(self): #return output
        return None
    
    def onException(self, exception):
        raise exception
        
    
    
class IOComponent(Component): # a component that receives and verifies input 
    
    # a dictionary with { "input_name" : (default_value, [possible_type*]) }
    # if default_value is None, an exception error is raised when input is missing
    # if the list of [possible_type*] is empty, no type verification will be done
    # the actual input values will be saved in self.input
    input_signature = {} 
    
    def pre_algorithm(self):
        super().pre_algorithm()
        self.output = {} #output will be a dictionary

    def getOutput(self): #return output
        return self.output
    
    def execute(self, input : dict):
        
        self.__passInput(input) #before the algorithm starts running, we define the input    
        return super().execute()

    def __passInput(self, input: dict): #pass and verify the input to this component, to the self.input dict
        
        passed_keys = input.keys()
        self.input = {}
        
        for input_key in self.input_signature.keys():
            
            (default_value, possible_types) = self.input_signature[input_key]
                        
            if input_key in passed_keys: #if this value was in input
                
                input_value = input[input_key]
                    
                if not verify_if_correct_type(input_value, possible_types):
                    raise Exception(f"Input with key '{input_key}' with type {type(input_value)} is not of any of the available types {possible_types}")
                    
                self.input[input_key] = input_value #the value used will be the one passed in the input
                
            else:
                                
                if default_value == None:
                    raise Exception(f"Did not set input for value {input_key} and has no default value")
                
                else:
                    self.input[input_key] = default_value  #the value used will be the default value
                    

def verify_if_correct_type(input_value, possible_types):
    
    is_a_correct_type = False
    
    if not isinstance(possible_types, list): #if the possible types is only one not defined in a list
        is_a_correct_type = isinstance(input_value, possible_types)
        
    else: #(else) if the possible types is a list with the possible types in it
    
        for possible_type in possible_types: 
            if isinstance(input_value, possible_type):
                is_a_correct_type = True
                break
            
    return is_a_correct_type