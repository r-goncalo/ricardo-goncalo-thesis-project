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
        
        while_component = loop_components.WhileFunDoFunComponent({
                "condition": condition,
                "execution": execution,
                "pre_execution": pre_execution,
                "post_execution" : post_execution
            })
        
        output = while_component.execute()
                
        assert output["result"] == 10
        
    def test_wrong_type(self):
        

        
        try:
            
            print("Trying to run while component with wrong condition type")
            
            while_component = loop_components.WhileFunDoFunComponent(                {
                    "condition": 5,
                    "execution": execution,
                    "pre_execution": pre_execution,
                    "post_execution" : post_execution
                })
        
            output = while_component.execute()
            
            assert False
        
        except:
            print("Success in caugthing an exception")
            assert True
                
        


        

class TestDoNTimesComponent(unittest.TestCase):
    
    def test_correct_default(self):
        
        do_n_times_component = loop_components.DoNTimesComponent(            {
                "execution": execution,
                "pre_execution": pre_execution,
                "post_execution" : post_execution
            })
        
        output = do_n_times_component.execute()
        
        assert output["result"] == loop_components.DoNTimesComponent.DEFAULT_TIMES_TO_DO
        
    def test_correct_non_default(self):
        
        custom_times_to_do = 20

        
        do_n_times_component = loop_components.DoNTimesComponent(
                        {
                "execution": execution,
                "pre_execution": pre_execution,
                "post_execution" : post_execution,
                "times_to_do" : custom_times_to_do 
            }
        )
        
        output = do_n_times_component.execute()
        
        assert output["result"] == custom_times_to_do
        
    def test_correct_non_default_input_in_init(self):
        
        custom_times_to_do = 30
        
        do_n_times_component = loop_components.DoNTimesComponent( #pass input in initialization
            {
                "execution": execution,
                "pre_execution": pre_execution,
                "post_execution" : post_execution,
                "times_to_do" : custom_times_to_do 
            })
        
        
        output = do_n_times_component.execute()
        
        assert output["result"] == custom_times_to_do
        
    def test_correct_non_default_input_in_between(self):
        
        custom_times_to_do = 30
        
        do_n_times_component = loop_components.DoNTimesComponent(
                        {
                "execution": execution,
                "pre_execution": pre_execution,
                "post_execution" : post_execution,
                "times_to_do" : custom_times_to_do 
            }
        )
        
        
        output = do_n_times_component.execute()
        
        assert output["result"] == custom_times_to_do

        

if __name__ == '__main__':
    unittest.main()