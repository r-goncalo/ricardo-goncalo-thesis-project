from ...automl.basic_components import loop_components
from ...automl.component import Component, input_signature
import unittest


class SimpleComponent(Component):
    
    input_signature = {"number" : input_signature()}
    

class SimpleExtendendComponent(SimpleComponent):
    
    input_signature = {"number" : input_signature(default_value=10)}
    
    
class SimpleComponent_2(Component):
    
    input_signature = {"number" : input_signature(default_value=10)}
    

class SimpleExtendendComponent_2(SimpleComponent_2):
    
    input_signature = {"number" : input_signature()}


class TestDefaultOverlap(unittest.TestCase):
        
    def test_default_in_super(self):
                
        simple_extended_component = SimpleExtendendComponent(input={})
            
        assert simple_extended_component.input["number"] == 10
        
    def test_default_in_child(self):
        
        
        simple_extended_component_2 = SimpleExtendendComponent_2(input={})
        assert simple_extended_component_2.input["number"] == 10
              
        

if __name__ == '__main__':
    unittest.main()