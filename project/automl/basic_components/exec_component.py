from ..component import InputSignature, Component, requires_input_proccess

from abc import abstractmethod
from enum import Enum

from typing import final

class State(Enum): #an enumerator to track state of executable component
        IDLE = 0
        RUNNING = 1
        OVER = 2
        ERROR = 4

# EXECUTABLE COMPONENT --------------------------

class ExecComponent(Component):
    
    parameters_signature = {
        
            "times_to_run" : InputSignature(mandatory=False, description="The number of times to run the component"),
    
    }

    exposed_values = {"running_state" : State.IDLE, "times_ran" : 0}    
    
    def proccess_input_internal(self):
        super().proccess_input_internal()

        if "times_to_run" not in self.input:
            self.times_to_run = None
        else:
            self.times_to_run = self.input["times_to_run"]

        


    # METHODS TO OVERRIDE --------------------------------


    def algorithm(self): #the algorithm of the component
        pass
    
    def pre_algorithm(self): # a component may extend this for certain behaviours
        self.values["running_state"] = State.RUNNING
        
    def pos_algorithm(self): # a component may extend this for certain behaviours
        
        self.values["times_ran"] += 1
        
        if self.times_to_run != None and self.values["times_ran"] < self.times_to_run:
            self.values["running_state"] = State.IDLE
            
        else:
            self.values["running_state"] = State.OVER
            
        

    def __on_exception(self, exception):
        '''To be called when a non treated exception happens'''
        self._deal_with_exceptionn(exception)
    
    def _deal_with_exceptionn(self, exception):
        pass

    # RUNNABLE METHOD --------------------------------
    
    @requires_input_proccess
    @final
    def run_all(self):
        
        '''Runs the algorithm the number of times specified in the input.'''
        
        if self.times_to_run == None:
            return self.run()
        
        else:
            for i in range(self.times_to_run - self.values["times_ran"]):
                self.run()
                
            return self.get_output()
    
    @requires_input_proccess
    @final
    def run(self): #the universal execution flow for all components
        
        '''Runs the specified algorithm once.'''
         
        try:
            self.pre_algorithm()
            self.algorithm()
            self.pos_algorithm()
            return self.get_output()
        
        except Exception as e:
            self.values["running_state"] = State.ERROR
            self.__on_exception(e)
            raise e # TODO: decide on this, are the elements that run this responsible for caughting the exception?
    
