
from enum import Enum
from abc import abstractmethod

import json

# Reserved attributes: input, values, parameters_signature, exposed_values, output, _input_was_proccessed

class Schema: # a component that receives and verifies input
    
    # a dictionary with { "input_name" : (default_value, validity verification) }
    # if default_value is None, an exception error is raised when input is missing
    # if validity verification function is not none, it will be applied to the input value
    # the actual input values will be saved in self.input
    # worth noting that child components can freely repeat values in their parameters_signature.
    #   Default values and generators of child components will have priority
    #   Validity and type checkers will be applied from Parent to child     
    parameters_signature = {}
    
    #A dictionary { "value_name" -> initial_value }
    #it tells other components what are the values exposed by this component, useful when checking the validity of the program before running it
    #this does not need to be stored in the component, the only reason it is is to standardize this kind of exposure
    exposed_values = {}
    
    
    # INITIALIZATION -------------------------------------------------------------------------
    
    def __init__(self, input : dict[str, any] = {}): #we can immediatly receive the input
                
        self.input = {} #the input will be a dictionary
        self.__set_exposed_values_with_super(type(self)) #updates the exposed values with the ones in super classes
        self.values = self.exposed_values.copy() #this is where the exposed values will be stored
        self.child_components = []
        self.parent_component : Schema = None
        
        self.name = str(type(self).__name__) #defines the initial component name


        self.pass_input(input) #passes the input but note that it does not proccess it
        
        self.output = {} #output, if any, will be a dictionary
        
        self._input_was_proccessed = False #to track if the instance has had its input proccessing before any operations that needed it
        
            
    
    def pass_input(self, input: dict): # pass input to this component, may need verification of input
        '''Pass input to this component
           It is supposed to be called only before proccess_input and, if not, it mean proccess_input may be called a multitude of times
           '''
        self._input_was_proccessed = False #when we pass new input, it means that we need to proccess it again
        
        for passed_key in input.keys():
            
            parameter_signature = self.get_parameter_signature(passed_key)
            
            if parameter_signature != None:
                
                self.__verified_pass_input(passed_key, input[passed_key], parameter_signature)
                
            else:
                print(f"WARNING: input with key {passed_key} passed to component {self.name} but not in its input signature, will be ignored")

                
                
    def __verified_pass_input(self, key, value, parameter_signature): #for logic of passing the input, already verified
        
        self.input[key] = value
        
        if parameter_signature.on_pass != None: #if there is verification of logic for when the input is passed
            
            parameter_signature.on_pass(self)



    def proccess_input(self): #verify the input to this component
        '''Verify validity of input and add default values''' 
        
        #get input signature priorities specified, sorted
        #and the input signatures organized by priorities
        parameters_signature_priorities, organized_parameters_signature = self.__get_organized_parameters_signature()
        
        passed_keys = self.input.keys()
        
        for priority in parameters_signature_priorities:
                        
            self.__add_default_values_of_class_input(passed_keys, organized_parameters_signature[priority])
            self.__verify_input(passed_keys, organized_parameters_signature[priority])
            
        self._input_was_proccessed = True


    # OUTPUT -------------------------
    
    def get_output(self): #return output
        return self.output
    
    # CHILD AND PARENT COMPONENTS ---------------------------------
    
    def initialize_child_component(self, component_type, input={}):
        
        '''Explicitly initializes a component of a certain type as a child component of this one'''
        
        initialized_component = component_type(input)
        
        self.child_components.append(initialized_component)
        
        initialized_component.parent_component = self
        
        return initialized_component
    
    def get_attr_from_parent(self, attr_name : str):
        '''Gets an attribute from a parent component, None if non existent'''
                
        if self.parent_component == None:
            return None
        
        else:
            if hasattr(self.parent_component, attr_name):
                return getattr(self.parent_component, attr_name)
            else:
                return self.parent_component.get_attr_from_parent(attr_name)
            
    
    def get_localization(self):
        
        current_component = self
        
        full_localization = [] # where we'll store the full localization of the component
                        
        while True: #while we have not reached the source component
                        
            if current_component.parent_component != None:
                
                child_components_of_parent : list = current_component.parent_component.child_components
            
                index_of_this_in_parent = child_components_of_parent.index(current_component)
            
                full_localization.insert(0, index_of_this_in_parent)
            
                current_component = current_component.parent_component
                
            else:
                break #we reached the source component
            
        return full_localization 
    
    # EXPOSED VALUES -------------------------------------------
    
    # TODO : maybe this should happen at the end of class definition for all classes that extend Component, statically? Instead of happening multiple times per class redundantly
    def __set_exposed_values_with_super(self, class_component):
        
        '''Updates the exposed values with super classes'''
                
        if len(class_component.__mro__) > 2:
            self.__set_exposed_values_with_super(class_component.__mro__[1]) # adds the exposed values of the super component if any
        
            
        self.exposed_values =  {**class_component.exposed_values, **self.exposed_values}


    # INPUT PROCCESSING ---------------------------------------------  
    
    def __get_parameter_signature(self, key, class_component):
        
        if key in class_component.parameters_signature.keys():
            return class_component.parameters_signature[key]
          
        elif len(class_component.__mro__) > 2:
            return self.__get_parameter_signature(key, class_component.__mro__[1])
        
        else:
            return None
    
    
    
    def get_parameter_signature(self, key):
        return self.__get_parameter_signature(key, type(self))   
    
    
    
    def __in_parameters_signature(self, key, class_component): #checks if the key is in the input signature of this class, if not, checks in super classes
        
        if key in class_component.parameters_signature.keys():
            return True
        
        elif len(class_component.__mro__) > 2:
            return self.__in_parameters_signature(key, class_component.__mro__[1])
        
        else:
            return False
    
    def in_parameters_signature(self, key): #checks if the key is in input signature of component or its parents components
        
        return self.__in_parameters_signature(key, type(self))
    
    
    
    # TODO: This is missing the capability of a Schematic re-writing the InputSignature of its super
    def __get_organized_parameters_signature(self): 
        
        current_class_component = self.__class__
        
        parameters_signature_priorities = []
        organized_parameters_signatures = {}
        
        while True: #for each component class
            
            current_parameters_signature = current_class_component.parameters_signature #get its input signature 
            
            for key, parameter_signature in current_parameters_signature.items():
                
                priority = parameter_signature.priority
                
                if not priority in parameters_signature_priorities:
                    parameters_signature_priorities.append(priority)
                    organized_parameters_signatures[priority] = []
                    
                organized_parameters_signatures[priority].append((key, parameter_signature)) #put its key, parameter_signature pair in the list of respective priority                
            
            if current_class_component == Schema: #if this was the Component class, we reached the end
                break
            
            current_class_component = current_class_component.__bases__[0] #gets the super class

        parameters_signature_priorities.sort()
        
        return parameters_signature_priorities, organized_parameters_signatures
        
            
    # DEFAULT VALUES -------------------------------------------------
            
    def __add_default_values_of_class_input(self, passed_keys, list_of_signatures):
        
        for (input_key, parameter_signature) in list_of_signatures:
                             
            #if this values was not already defined
            if not input_key in passed_keys: 
                                                                   
                if not parameter_signature.default_value == None:
                    self.input[input_key] = parameter_signature.default_value  #the value used will be the default value

                elif not parameter_signature.generator == None:
                    self.input[input_key] = parameter_signature.generator(self) #generators have access to the instance              
                

    # VALIDITY VERIFICATION ---------------------------------------------

    def __verify_input(self, passed_keys, list_of_signatures): #verify the input to this component and add default values, to the self.input dict

        for (input_key, parameter_signature) in list_of_signatures:
                                    
            if input_key in passed_keys: #if this value was in input
                
                input_value = self.input[input_key] #get the value passed
                    
                self.verify_validity(input_key, input_value, parameter_signature.possible_types, parameter_signature.validity_verificator) #raises exceptions if input is not valid
                                    
            else: #if there was no specified value for this attribute in the input
                raise Exception(f"In component of type {type(self)}, when cheking for the inputs: Did not set input for value  with key '{input_key}' and has no default value nor generator")     
                


    def verify_validity(self, input_key, input_value, possible_types, validity_verificator):

        if validity_verificator == None: #if there was no specified validity_verification, we check the available types
            return self.verify_one_of_types(input_key, input_value, possible_types) 

        is_a_correct_type = validity_verificator(input_value) #use the verificator the Component has specified in its input signature

        if not isinstance(is_a_correct_type, bool): #validity_verification must return bool
            raise Exception(f"In component of type {type(self)}: Validity verification on key '{input_key}' returned a type other than bool")

        elif is_a_correct_type == False:
            raise Exception(f"In component of type {type(self)}: Value with key '{input_key}' did not pass Component specified validity verificator") 




    def verify_one_of_types(self, input_key, input_value, possible_types):

            if len(possible_types) == 0:
                return #if there were no possible_types defined, there is no need to check the type

            for possible_type in possible_types:

                if isinstance(input_value, possible_type):
                    return #break the loop and the functon, value is of one of the possible types

            #if we reach the end of the function, then the value is of none of the types
            raise Exception(f"No validity verificator specified for key '{input_key}' and its type ({type(input_key)}) is of none of the possible types: {possible_types}")
          

# VALIDITY VERIFICATION (static methods for validating input) -----------------------------          

def requires_input_proccess(func):
    '''An annotation that makes the input be proccessed, if it was not already, when a function is called'''
    def wrapper(self : Schema, *args, **kwargs):
        if not self._input_was_proccessed:
            self.proccess_input()
        return func(self, *args, **kwargs)
    return wrapper

def uses_component_exception(func):
    '''A wrapper for functions so its errors have more information regarding the component they appeared in'''

    def wrapper(self : Schema, *args, **kwargs):
        
        try:
            return func(self, *args, **kwargs)
        
        except Exception as e:
            original_args = e.args #arguments of an error
            e.args = (f"On Component {self.name} of type {type(self).__name__}:\n {original_args[0]}", *original_args[1:])
            raise e
            
    return wrapper
        

class InputSignature():
    
    
    def __init__(self, default_value=None, generator=None, validity_verificator=None, possible_types : list = [], description='', ignore_at_serialization=False, priority=1, on_pass=None):
        
        self.default_value = default_value
        self.generator = generator
        self.validity_verificator = validity_verificator
        self.possible_types = possible_types
        self.description = description
        self.ignore_at_serialization = ignore_at_serialization
        self.priority = priority
        self.on_pass = on_pass
        
        




