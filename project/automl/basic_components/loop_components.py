
from ..component import IOComponent, Component
import types

class WhileFunDoFunComponent(IOComponent):

    input_signature = {**IOComponent.input_signature, 
                       "execution" : (None, [ types.FunctionType]),
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
