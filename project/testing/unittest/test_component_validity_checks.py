from ...automl.component import Schema, parameters_signature
import unittest



    

class TestComponentLambdaVerificator(unittest.TestCase):
        
    class _ComponentLambdaVerificator(Schema):
    
        N_MUST_BE_GREATER_THAN = 5
    
        parameters_signature = {"number" : parameters_signature(validity_verificator=lambda n : n > TestComponentLambdaVerificator._ComponentLambdaVerificator.N_MUST_BE_GREATER_THAN)}        
        
    def test_wrong_input(self):
                
        simple_extended_component = TestComponentLambdaVerificator._ComponentLambdaVerificator(input={"number" : TestComponentLambdaVerificator._ComponentLambdaVerificator.N_MUST_BE_GREATER_THAN - 1})
        
        try:
            simple_extended_component.proccess_input()
            assert False
        
        except:
            assert True    
        
    def test_right_input(self):
                
        simple_extended_component = TestComponentLambdaVerificator._ComponentLambdaVerificator(input={"number" : TestComponentLambdaVerificator._ComponentLambdaVerificator.N_MUST_BE_GREATER_THAN + 1})
        
        try:
            simple_extended_component.proccess_input()
            assert True
        
        except:
            assert False    
        

if __name__ == '__main__':
    unittest.main()