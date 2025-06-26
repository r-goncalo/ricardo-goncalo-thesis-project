


from automl.component import Component, requires_input_proccess, InputSignature
from automl.basic_components.exec_component import ExecComponent
from automl.basic_components.seeded_component import SeededComponent
from automl.basic_components.evaluator_component import ComponentWithEvaluator, EvaluatorComponent
from automl.loggers.component_with_results import ComponentWithResults
from automl.loggers.logger_component import ComponentWithLogging
from automl.loggers.result_logger import ResultLogger, get_results_logger_from_file
from automl.utils.json_component_utils import gen_component_from_dict, json_string_of_component, component_from_json_string
from automl.utils.files_utils import write_text_to_file, read_text_from_file
from automl.basic_components.artifact_management import ArtifactComponent
import os

from typing import Union

from automl.basic_components.state_management import StatefulComponent, StatefulComponentLoader

from automl.consts import CONFIGURATION_FILE_NAME

Component_in_group_type = Union[ExecComponent, StatefulComponent, SeededComponent, ComponentWithResults]

class RunnableComponentGroup(ExecComponent, SeededComponent, StatefulComponent, ComponentWithLogging, ComponentWithResults, EvaluatorComponent):

    '''A component which represents a group of runnable seeded components which can be run and evaluated'''
    
    parameters_signature = {
        
        "number_of_components" : InputSignature(
            description="The number of components to be used in the group",
            default_value=3,
        ),
        
        "component_class" : InputSignature(
            description="The class of the component to be used in the group",
            mandatory=False
        ),
        
        "component_dic" : InputSignature(mandatory=False),
        
        "component_parameters" : InputSignature(
            description="The parameters of the component to be used in the group",
        ),
        
        "component_in_group_evaluator" : InputSignature(
            description="The evaluator of the component to be used in the group",
            mandatory=False
        ),
        
    }
    
    
    def proccess_input_internal(self):
        super().proccess_input_internal()
        
        self.runnable_components = []
        self.last_ran_component = -1
        
        self.aggregate_results_lg = None
        
        self.instantiate_evaluator()
        
        self.check_component_class()
        
        self.instantiate_components_in_group()
        
        
    
    def instantiate_evaluator(self):
        
        if "component_in_group_evaluator" not in self.input:
            self.component_in_group_evaluator = None
        
        else:
            self.component_in_group_evaluator : EvaluatorComponent  = self.input["component_in_group_evaluator"]
            
            
        
    # AGGREGATE RESULTS LOGGING ----------------------------------------------
        
    def initialize_aggregate_results_lg(self, component : Component_in_group_type):
        
        component_results = component.get_results_columns()
        
        
        if self.component_in_group_evaluator != None:
            evaluation_results = self.component_in_group_evaluator.get_metrics_strings()
            
        else:
            evaluation_results = []
            
        
        #the results shown in the aggregate results is a combination of the evaluation and the results shown by the components
        results_columns = ["index", "times_ran", *component_results, *evaluation_results]
        
        self.aggregate_results_lg : ResultLogger = self.initialize_child_component(ResultLogger, { 
            "results_filename" : "aggregate_results.csv",
            "artifact_relative_directory" : "",
            "results_columns" : results_columns
            })
        
        
        
    def get_aggregate_results_lg(self) -> ResultLogger:
        
        '''
        Returns the aggregate results logger
        This does not need the component to be initialized, as it is capable of getting them from file if needed
        '''
        
        if hasattr(self, "aggregate_results_lg") and self.aggregate_results_lg != None:
            return self.aggregate_results_lg
        
        else:
            return get_results_logger_from_file(self.get_artifact_directory(), "aggregate_results.csv")
    
    
    def get_results_by_component(self) -> dict:
        
        '''
        Returns the results of the components in the group, grouped by component index
        If the input is not processed, this will simply return the results save in file
        '''
        
        aggregate_results_lg = self.get_aggregate_results_lg()
        
        results_per_component = aggregate_results_lg.get_grouped_dataframes("component_index")
        
        return results_per_component
    
    
        
        
    def log_aggregate_results(self, component : Component_in_group_type, component_index : int, component_results, evaluation_results):
        
        self.aggregate_results_lg.log_results({
            "component_index" : component_index,
            "times_ran" : component.values["times_ran"],
            **component_results,
            **evaluation_results
        })
        
            
    # COMPONENT GROUP INITIALIZATION ----------------------------------------------
            
    def check_component_class(self):
        
        if not issubclass(self.input["component_class"], SeededComponent):
            raise Exception("The component class must be a subclass of SeededComponent")
        
        if not issubclass(self.input["component_class"], ExecComponent):
            raise Exception("The component class must be a subclass of ExecComponent")
        
        self.using_stateful_components = issubclass(self.input["component_class"], StatefulComponent)
        
        
        
    def generate_stateful_component(self) -> Component_in_group_type:
        
        '''Generates a component in the group from the class or the dic'''
        
        if "component_class" not in self.input:
            
            if "component_dic" in self.input: #generate the component from the dic
                stateful_component = gen_component_from_dict(self.input["component_dic"])
            
            else:
                raise Exception("Either the component class is defined or the component dic")
            
        else: #generate the component from the class
            stateful_component = self.input["component_class"](
                self.input["component_parameters"]
            )
            
        if isinstance(stateful_component, ComponentWithEvaluator):
            stateful_component.pass_input({"component_evaluator" : self.component_in_group_evaluator})
            
        return stateful_component
    
        
    def add_component(self):

        '''Adds a component to the group'''
        
        component_parameters = self.input["component_parameters"]
        component_parameters["create_new_directory"] = True #this is to make sure that each component has its own directory

        stateful_component = self.generate_stateful_component()

        if self.using_stateful_components:
                                    
            component = StatefulComponentLoader()
            
            component.define_component_to_save_load(stateful_component)
            
            
        else:
            self.define_component_as_child(stateful_component)
            component = stateful_component
        
        self.runnable_components.append(component)
        
        
    def instantiate_components_in_group(self):
        '''Initializes the components in the group'''
        
        for _ in range(self.input["number_of_components"]):
            
            self.add_component()
            
    
    def get_component(self, index) -> Component_in_group_type:
        
        '''Returns the component at the given index'''

        if self.using_stateful_components:
            
            component_loader : StatefulComponentLoader = self.runnable_components[index]
            
            return component_loader.load_component()
            
        else:
            
            return self.runnable_components[index]
            
    
    # RUNNING COMPONENTS -------------------------------------------------
    
    def evaluate_component_in_group(self, component : Component_in_group_type):
        
        if isinstance(component, ComponentWithEvaluator):
            return component.evaluate_this_component()
        
        elif self.component_in_group_evaluator == None:
            return {}
        
        else:
            self.component_in_group_evaluator.evaluate(component)
            
            
            
    def run_component(self, index):
        
        '''Runs the component at the given index'''
        
        component = self.get_component(index)
            
        component.run()
        
        evaluation = self.evaluate_component_in_group(component)
        
        if self.aggregate_results_lg == None: #if this was not initialized yet, we initialize it
            self.initialize_aggregate_results_lg(component)
        
        self.log_aggregate_results(component, index, component.get_last_results(), evaluation)
        
        if self.using_stateful_components:
            loader : StatefulComponentLoader = self.runnable_components[index]
            loader.save_and_onload_component()
            
        
        self.last_ran_component = index

    
    def run_next_component(self):
        
        '''Runs the next component in the group'''
        
        index = self.last_ran_component + 1
        
        if self.last_ran_component >= len(self.runnable_components) - 1:
            index = 0
        
        self.run_component(index)
        
        
    def run_all_components(self):
        
        '''Runs all the components in the group once'''
        
        for i in range(len(self.runnable_components)):
            self.run_component(i)
            
    
    # Execution ----------------------------------------
    
    def algorithm(self):
        self.run_all_components()