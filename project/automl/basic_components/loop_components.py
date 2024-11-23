
from ..component import IOComponent, Component
import types

class WhileFunDoFunComponent(IOComponent):

    input_signature = {**IOComponent.input_signature, 
                       "execution" : (None, [ types.FunctionType]), #have no default values (None) and are functions
                       "condition" : (None, [ types.FunctionType]),
                       "pre_execution" : (None, [ types.FunctionType]),
                       "post_execution" : (None, [types.FunctionType])}
    
    def algorithm(self):
                
        condition = self.input["condition"]
        execution = self.input["execution"]
        pre_execution = self.input["pre_execution"]
        post_execution = self.input["post_execution"]
        
        pre_execution(self)
        
        while condition(self):
            execution(self) 
            
        post_execution(self)
        
    
class DoNTimesComponent(IOComponent):
    
    DEFAULT_TIMES_TO_DO = 10
        
    input_signature = {**IOComponent.input_signature, 
                       "execution" : (None, [ types.FunctionType]),
                       "pre_execution" : (None, [ types.FunctionType]),
                       "post_execution" : (None, [types.FunctionType]),
                       "times_to_do" : (DEFAULT_TIMES_TO_DO, int)}
    
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
