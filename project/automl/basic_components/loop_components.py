
from ..component import ExecComponent
import types

class WhileFunDoFunComponent(ExecComponent):

    #The inputs this component receives
    input_signature = {**ExecComponent.input_signature, #the inputs of the parent 
                       "execution" : (None, lambda x : isinstance(x, types.FunctionType)), #have no default values (None) and are functions
                       "condition" : (None, lambda x : isinstance(x, types.FunctionType)),
                       "pre_execution" : (None, lambda x : isinstance(x, types.FunctionType)),
                       "post_execution" : (None, lambda x : isinstance(x, types.FunctionType))}
    
    def algorithm(self):
                
        condition = self.input["condition"]
        execution = self.input["execution"]
        pre_execution = self.input["pre_execution"]
        post_execution = self.input["post_execution"]
        
        pre_execution(self)
        
        while condition(self):
            execution(self) 
            
        post_execution(self)
        
    
class DoNTimesComponent(ExecComponent):
    
    DEFAULT_TIMES_TO_DO = 10
        
    input_signature = {**ExecComponent.input_signature, 
                       "execution" : (None, lambda x : isinstance(x, types.FunctionType)),
                       "pre_execution" : (None, lambda x : isinstance(x, types.FunctionType)),
                       "post_execution" : (None, lambda x : isinstance(x, types.FunctionType)),
                       "times_to_do" : (DEFAULT_TIMES_TO_DO, lambda x : isinstance(x, int))}
    
    def algorithm(self):
                
        execution = self.input["execution"]
        pre_execution = self.input["pre_execution"]
        post_execution = self.input["post_execution"]
        
        pre_execution(self)
        
        i : int = self.input["times_to_do"]
        
        while i > 0:
            execution(self)
            i -= 1 
            
        post_execution(self)
