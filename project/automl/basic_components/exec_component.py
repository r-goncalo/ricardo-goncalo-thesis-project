from automl.loggers.component_with_results import save_all_dataframes_of_component_and_children
from automl.basic_components.state_management import save_state
from automl.utils.smart_enum import SmartEnum
from automl.loggers.logger_component import flush_text_of_all_loggers_and_children
from automl.loggers.global_logger import globalWriteLine
from ..component import InputSignature, Component, requires_input_proccess

from abc import abstractmethod
from enum import Enum

from typing import final

class State(SmartEnum): #an enumerator to track state of executable component
        IDLE = 0
        RUNNING = 1
        OVER = 2
        ERROR = 4
        INTERRUPTED = 5

class StopExperiment(Exception):
    pass

# EXECUTABLE COMPONENT --------------------------

class ExecComponent(Component):
    
    parameters_signature = {
        
            "times_to_run" : InputSignature(mandatory=False, description="The number of times to run the component"),
            "save_state_on_run_end" : InputSignature(default_value=True, ignore_at_serialization=True),
            "save_dataframes_on_run_end" : InputSignature(default_value=True, ignore_at_serialization=True)
    
    }

    exposed_values = {"running_state" : State.IDLE, "times_ran" : 0}    
    
    def _proccess_input_internal(self):
        super()._proccess_input_internal()

        self._times_to_run = self.get_input_value("times_to_run")
        
        self.__save_state_on_run_end = self.get_input_value("save_state_on_run_end")
        self.__save_dataframes_on_run_end = self.get_input_value("save_dataframes_on_run_end")

        self._received_signal_to_stop = False        


    # METHODS TO OVERRIDE --------------------------------


    def _algorithm(self): #the algorithm of the component
        pass
    
    def _pre_algorithm(self): # a component may extend this for certain behaviours
        self.values["running_state"] = State.RUNNING
        self._received_signal_to_stop = False
        
    def _pos_algorithm(self): # a component may extend this for certain behaviours
        
        self.values["times_ran"] += 1
        
        if self._times_to_run != None and self.values["times_ran"] < self._times_to_run:
            self.values["running_state"] = State.IDLE
            
        else:
            self.values["running_state"] = State.OVER
        
        if self.__save_state_on_run_end:
            save_state(self)
        
        if self.__save_dataframes_on_run_end:
            save_all_dataframes_of_component_and_children(self)
            flush_text_of_all_loggers_and_children(self)
            
    
    def _stop_earlier_signal_received(self):
        return self._received_signal_to_stop
    
    def stop_execution_earlier(self):
        self._received_signal_to_stop = True

    def _check_if_should_stop_execution_earlier(self):
        return False
    
    
    def check_if_should_stop_execution_earlier(self):

        '''Raises exception if the execution should stop earlier'''

        if self._stop_earlier_signal_received():
            raise StopExperiment()
        
        elif self._check_if_should_stop_execution_earlier():
            self.stop_execution_earlier()
            raise StopExperiment()
        
        else:
            return False
        
    def _on_earlier_interruption(self):
        pass
    

    def __on_exception(self, exception):
        '''To be called when a non treated exception happens'''

        self._deal_with_exception(exception)
    
    def _deal_with_exception(self, exception):
        '''Called internally when an exception happens'''
        pass

    # RUNNABLE METHOD --------------------------------
    
    @requires_input_proccess
    @final
    def run_all(self):
        
        '''Runs the algorithm the number of times specified in the input.'''
        
        if self._times_to_run == None:
            return self.run()
        
        else:
            for i in range(self._times_to_run - self.values["times_ran"]):
                self.run()
                
            return self.get_output()
    
    @requires_input_proccess
    @final
    def run(self): #the universal execution flow for all components
        
        '''Runs the specified algorithm once.'''
         
        try:
            self._pre_algorithm()
            self._algorithm()
            self._pos_algorithm()
            return self.get_output()
        
        except Exception as e:

            if isinstance(e, StopExperiment):
                self.values["running_state"] = State.INTERRUPTED
                self._on_earlier_interruption()

                if self.__save_state_on_run_end:
                    save_state(self)

                if self.__save_dataframes_on_run_end:
                    save_all_dataframes_of_component_and_children(self)
                    flush_text_of_all_loggers_and_children(self)

            else:
                self.values["running_state"] = State.ERROR
                self.__on_exception(e)

            self.values["times_ran"] += 1

            if self.__save_state_on_run_end:
                save_state(self)

            if self.__save_dataframes_on_run_end:
                save_all_dataframes_of_component_and_children(self)
                flush_text_of_all_loggers_and_children(self)

            raise e
    
