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
        
        output = while_component.execute(
            {
                "condition": condition,
                "execution": execution,
                "pre_execution": pre_execution,
                "post_execution" : post_execution
            }
        )
        
        print(output)
        
        assert output["result"] == 10
        

        

if __name__ == '__main__':
    unittest.main()