

from automl.core.advanced_input_management import ComponentInputSignature
from automl.core.input_management import InputSignature
from automl.loggers.global_logger import globalWriteLine
from automl.ml.memory.memory_components import MemoryComponent
from automl.ml.memory.torch_memory_component import TorchMemoryComponent
from automl.rl.learners.learner_component import LearnerSchema
from automl.rl.learners.learning_acessory import LearningAcessory


class ConvergenceNoticer(LearningAcessory):

    parameters_signature = {
        "memory" : ComponentInputSignature(default_component_definition=(
            TorchMemoryComponent,
            {"capacity" : 256}
        )),
        "convergence_treshold" : InputSignature(default_value=0.1)
    }

    def _proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._proccess_input_internal()

        self.learner : LearnerSchema = self.learner

        self.memory : MemoryComponent = self.get_input_value("memory")
        self.convergence_treshold = self.get_input_value("convergence_treshold")

        self.memory.pass_input({
               "transition_data" : [("convergence_value", 1)]
               })
        
        self.convergence_noticed = False
        
        
    def calculate_change(self, learning_values : dict):
        pass


    def post_learning(self, learning_values : dict):

        if not self.convergence_noticed:
        
            value = self.calculate_change(learning_values)
            self.memory.push({"convergence_value" : value})
            
            if self.memory.is_full():
                avg_value = sum(self.memory.get_all()["convergence_value"]) / self.memory.capacity
    
                if avg_value < self.convergence_treshold:
                    self.convergence_noticed = True
                    globalWriteLine(f"Convergence noticer {self.name} noticed that learner {self.learner.name} has reached convergence, should stop training")


        

class PPOConvergenceNoticer(ConvergenceNoticer):

    parameters_signature = {
    }

    def _proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._proccess_input_internal()
                
    
    def calculate_change(self, learning_values : dict):
        pass


