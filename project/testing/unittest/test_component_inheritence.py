from ...automl.component import Component, input_signature
import unittest


class _SimpleComponent(Component):
    
    input_signature = {"number" : input_signature()}
    

class _SimpleExtendendComponent(_SimpleComponent):
    
    input_signature = {"number" : input_signature(default_value=10)}
    
    
class _SimpleComponent_2(Component):
    
    input_signature = {"number" : input_signature(default_value=10)}
    

class _SimpleExtendendComponent_2(_SimpleComponent_2):
    
    input_signature = {"number" : input_signature()}


class TestDefaultOverlap(unittest.TestCase):
        
    def test_default_in_super(self):
                
        simple_extended_component = _SimpleExtendendComponent(input={})
        
        simple_extended_component.proccess_input()
            
        assert simple_extended_component.input["number"] == 10
        
    def test_default_in_child(self):
        
        
        simple_extended_component_2 = _SimpleExtendendComponent_2(input={})
        simple_extended_component_2.proccess_input()
        assert simple_extended_component_2.input["number"] == 10
              
        

if __name__ == '__main__':
    unittest.main()