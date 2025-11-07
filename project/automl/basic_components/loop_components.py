
from ..core.exec_component import ExecComponent
from ..component import InputSignature
import types

class WhileFunDoFunComponent(ExecComponent):

    #The inputs this component receives
    parameters_signature = { 
                       "execution" : InputSignature(validity_verificator= lambda x : isinstance(x, types.FunctionType)), #have no default values (None) and are functions
                       "condition" : InputSignature(validity_verificator= lambda x : isinstance(x, types.FunctionType)),
                       "pre_execution" : InputSignature(validity_verificator= lambda x : isinstance(x, types.FunctionType)),
                       "post_execution" : InputSignature(validity_verificator= lambda x : isinstance(x, types.FunctionType))}
    
    def algorithm(self):
                
        condition = self.get_input_value("condition")
        execution = self.get_input_value("execution")
        pre_execution = self.get_input_value("pre_execution")
        post_execution = self.get_input_value("post_execution")
        
        pre_execution(self)
        
        while condition(self):
            execution(self) 
            
        post_execution(self)
        
    
class DoNTimesComponent(ExecComponent):
    
    DEFAULT_TIMES_TO_DO = 10
        
    parameters_signature = {
        "execution" : InputSignature(possible_types=[types.FunctionType]),
        "pre_execution" : InputSignature(possible_types=[types.FunctionType]),
        "post_execution" : InputSignature(possible_types=[types.FunctionType]),
        "times_to_do" : InputSignature(default_value=DEFAULT_TIMES_TO_DO, validity_verificator=lambda x : isinstance(x, int))
        }
    
    def algorithm(self):
                
        execution = self.get_input_value("execution")
        pre_execution = self.get_input_value("pre_execution")
        post_execution = self.get_input_value("post_execution")
        
        pre_execution(self)
        
        i : int = self.get_input_value("times_to_do")
        
        while i > 0:
            execution(self)
            i -= 1 
            
        post_execution(self)
