from ...automl.component import Schema, parameters_signature
import unittest


class _SimpleComponent(Schema):
    
    parameters_signature = {"number" : parameters_signature()}
    

class _SimpleExtendendComponent(_SimpleComponent):
    
    parameters_signature = {"number" : parameters_signature(default_value=10)}
    
    
class _SimpleComponent_2(Schema):
    
    parameters_signature = {"number" : parameters_signature(default_value=10)}
    

class _SimpleExtendendComponent_2(_SimpleComponent_2):
    
    parameters_signature = {"number" : parameters_signature()}


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