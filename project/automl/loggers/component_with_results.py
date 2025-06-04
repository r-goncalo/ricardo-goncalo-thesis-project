
from automl.component import Component, requires_input_proccess
from automl.basic_components.artifact_management import ArtifactComponent
from automl.core.input_management import InputSignature
from automl.loggers.result_logger import ResultLogger


def on_log_pass(self : Component):
            
    self.results_lg  = self.input["results_logger_object"]
    
def generate_logger_for_component(self : ArtifactComponent):
    
    if not hasattr(self,"results_columns") or self.results_columns == None or self.results_columns == []:
        return None
    
    else:
        return self.initialize_child_component(ResultLogger, input={
            "create_new_directory" : False,
            "base_directory" : self.get_artifact_directory(), 
            "artifact_relative_directory" : "",
            "results_columns" : self.results_columns}
        )


# TODO: In the future, components may extend this, but not the LoggingComponent
#TODO: Components should be able to easily have multiple ResultLoggers
class ComponentWithResults(ArtifactComponent):

    '''
    A component that generalizes the behaviour of a component that has a logger object 
    '''
    
    
    parameters_signature = {
                                                                                
                       "results_logger_object" : InputSignature(ignore_at_serialization=True, priority=10, 
                                                        generator = generate_logger_for_component , 
                                                        on_pass=on_log_pass)
                       }
    
    
    #every class that extends ComponentWithResults should define this, so that the ResultLogger can be initialized with the right columns
    #if not or if it is empty, it will be responsibility of the class to initialize the ResultLogger
    results_columns = [] 
    


    def proccess_input(self): #this is the best method to have initialization done right after
            
        super().proccess_input()
        
        self.results_lg : ResultLogger = self.input["results_logger_object"] if not hasattr(self, "results_lg") else self.results_lg #changes self.lg if it does not already exist
        

    @requires_input_proccess
    def get_results_columns(self):
        return self.results_lg.get_results_columns()
    
    @requires_input_proccess
    def log_results(self, results):
        '''
            Used internally to log the results of a component with results
        '''
        self.results_lg.log_results(results)
    
    @requires_input_proccess 
    def calculate_results(self) -> dict:
        '''
            Used internally to calculate the results of a component with results
        '''
        raise NotImplementedError("This method should be implemented in the child class")

    def calculate_and_log_results(self):
        '''
            Used internally to calculate the results of a component with results and then log them
        '''
        
        self.log_results(self.calculate_results())
    
    @requires_input_proccess
    def get_last_results(self):
        
        return self.results_lg.get_last_results()
    
    @requires_input_proccess
    def get_results_logger(self) -> ResultLogger:
        
        return self.results_lg