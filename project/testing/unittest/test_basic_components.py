from ...automl.basic_components import loop_components
import unittest

def condition(self):
    return self.i < 10

def execution(self):
    print(f"Executing {self.i}")
    self.i = self.i + 1

def pre_execution(self):
    self.i = 0

def post_execution(self):
    self.output["result"] = self.i

class TestWhileComponent(unittest.TestCase):
    
    def test_correct(self):
        
        while_component = loop_components.WhileFunDoFunComponent()
        
        output = while_component.pass_input_and_exec(
            {
                "condition": condition,
                "execution": execution,
                "pre_execution": pre_execution,
                "post_execution" : post_execution
            }
        )
                
        assert output["result"] == 10
        
    def test_wrong_type(self):
        
        while_component = loop_components.WhileFunDoFunComponent()
        
        try:
            
            print("Trying to run while component with wrong condition type")
        
            output = while_component.pass_input_and_exec(
                {
                    "condition": 5,
                    "execution": execution,
                    "pre_execution": pre_execution,
                    "post_execution" : post_execution
                }
            )
            
            assert False
        
        except:
            print("Success in caugthing an exception")
            assert True
                
        


        

class TestDoNTimesComponent(unittest.TestCase):
    
    def test_correct_default(self):
        
        do_n_times_component = loop_components.DoNTimesComponent()
        
        output = do_n_times_component.pass_input_and_exec(
            {
                "execution": execution,
                "pre_execution": pre_execution,
                "post_execution" : post_execution
            }
        )
        
        assert output["result"] == loop_components.DoNTimesComponent.DEFAULT_TIMES_TO_DO
        
    def test_correct_non_default(self):
        
        do_n_times_component = loop_components.DoNTimesComponent()
        
        custom_times_to_do = 20
        
        output = do_n_times_component.pass_input_and_exec(
            {
                "execution": execution,
                "pre_execution": pre_execution,
                "post_execution" : post_execution,
                "times_to_do" : custom_times_to_do 
            }
        )
        
        assert output["result"] == custom_times_to_do
        

        

if __name__ == '__main__':
    unittest.main()