
from enum import Enum
from abc import abstractmethod

import json  

class ComponentInputEncoder(json.JSONEncoder):
    
    def default(self, obj):
        
        if isinstance(obj, Component):

            return {
                "__type__": type(obj).__name__,
                "name": obj.name,
                "localization" : obj.get_localization()
            }
            
        elif isinstance(obj, (int, float, str, dict, list)):
            return obj
            
        return None

class ComponentEncoder(json.JSONEncoder):
    
    def default(self, obj):
        
        if isinstance(obj, Component):
            
            toReturn = {
                "__type__": type(obj).__name__,
                "name": obj.name,
                "input": json.loads(json.dumps(obj.input, cls=ComponentInputEncoder))
                }
            
            if len(obj.child_components) > 0:
                
                toReturn["child_components"] = self.default(obj.child_components)
                        
            return toReturn
            
        elif isinstance(obj, (int, float, str, dict, list)):
            return obj
            
        return None

        

                
# Reserved attributes: input, values, input_signature, exposed_values, output, _input_was_proccessed

class Component: # a component that receives and verifies input
    
    # a dictionary with { "input_name" : (default_value, validity verification) }
    # if default_value is None, an exception error is raised when input is missing
    # if validity verification function is not none, it will be applied to the input value
    # the actual input values will be saved in self.input
    # worth noting that child components can freely repeat values in their input_signatures.
    #   Default values and generators of child components will have priority
    #   Validity and type checkers will be applied from Parent to child     
    input_signature = {}
    
    #A dictionary { "value_name" -> initial_value }
    #it tells other components what are the values exposed by this component, useful when checking the validity of the program before running it
    #this does not need to be stored in the component, the only reason it is is to standardize this kind of exposure
    exposed_values = {}
    
    
    # INITIALIZATION -------------------------------------------------------------------------
    
    def __init__(self, input : dict = {}): #we can immediatly receive the input
                
        self.input = {} #the input will be a dictionary
        self.__set_exposed_values_with_super(type(self)) #updates the exposed values with the ones in super classes
        self.values = self.exposed_values.copy() #this is where the exposed values will be stored
        self.child_components = []
        self.parent_component = None

        self.pass_input(input) #passes the input but note that it does not proccess it
        
        self.output = {} #output, if any, will be a dictionary
        
        self._input_was_proccessed = False #to track if the instance has had its input proccessing before any operations that needed it
        
        self.name = str(type(self).__name__) #defines the initial component name
            
    
    def pass_input(self, input: dict): # pass input to this component
        '''Pass input to this component
           It is supposed to be called only before proccess_input and, if not, it mean proccess_input may be called a multitude of times
           '''
        self._input_was_proccessed = False #when we pass new input, it means that we need to proccess it again
        
        for passed_key in input.keys():
            
            if self.in_input_signature(passed_key):
                
                self.input[passed_key] = input[passed_key]
                
            else:
                print(f"WARNING: input with key {passed_key} passed to component {self.name} but not in its input signature, will be ignored")

    def proccess_input(self): #verify the input to this component
        '''Verify validity of input and add default values''' 
        self.__add_default_values_of_class_and_super(type(self))
        self.__proccess_input_of_class_and_super(type(self))
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
    
    def get_localization(self):
        
        current_component = self
        
        self.full_localization = [] # where we'll store the full localization of the component
                        
        while current_component != None:
            
            self.full_localization.insert(0, current_component.name)
            current_component = current_component.parent_component
    

    # EXPOSED VALUES -------------------------------------------
    
    # TODO : maybe this should happen at the end of class definition for all classes that extend Component, statically? Instead of happening multiple times per class redundantly
    def __set_exposed_values_with_super(self, class_component):
        
        '''Updates the exposed values with super classes'''
                
        if len(class_component.__mro__) > 2:
            self.__set_exposed_values_with_super(class_component.__mro__[1]) # adds the exposed values of the super component if any
        
            
        self.exposed_values =  {**class_component.exposed_values, **self.exposed_values}


    # INPUT PROCCESSING ---------------------------------------------  
    
    def __in_input_signature(self, key, class_component): #checks if the key is in the input signature of this class, if not, checks in super classes
        
        if key in class_component.input_signature.keys():
            return True
        
        elif len(class_component.__mro__) > 2:
            return self.__in_input_signature(self, key, class_component.__mro__[1])
        
        else:
            return False
    
    def in_input_signature(self, key): #checks if the key is in input signature of component or its parents components
        
        return self.__in_input_signature(key, type(self))
    
    
     
    def __proccess_input_of_class_and_super(self, class_component):

        if len(class_component.__mro__) > 2: #if this class has a super class that is not the object class
            self.__proccess_input_of_class_and_super(class_component.__mro__[1]) # verifies first the input according to the super class
                
        self.__proccess_input_specific_of_class(class_component)
        
            
    def __proccess_input_specific_of_class(self, class_component): #verify the input to this component and add default values, to the self.input dict

        input_signature = class_component.input_signature  #the input signature specified in this class          
        passed_keys = self.input.keys() #note that default values and verifications of super values could already be done
                
        for input_key in input_signature.keys():
            
            single_input_signature : InputSignature = input_signature[input_key]
                        
            if input_key in passed_keys: #if this value was in input
                
                input_value = self.input[input_key] #get the value passed
                    
                self.verify_validity(input_key, input_value, single_input_signature.possible_types, single_input_signature.validity_verificator) #raises exceptions if input is not valid
                                    
            else: #if there was no specified value for this attribute in the input
                raise Exception(f"In component of type {type(self)}, when cheking for the inputs required by class {class_component}: Did not set input for value  with key '{input_key}' and has no default value nor generator")     
                
    # DEFAULT VALUES -------------------------------------------------
    
    def __add_default_values_of_class_and_super(self, class_component):    
                
        self.__add_default_values_of_class(class_component) #adds first the default values of child classe(s)
        
        if len(class_component.__mro__) > 2:
            self.__add_default_values_of_class_and_super(class_component.__mro__[1]) # adds second the default values of the super component if any
        
            
    def __add_default_values_of_class(self, class_component):
        
        input_signature = class_component.input_signature  #the input signature specified in this class          
        passed_keys = self.input.keys() #note that default values of child classes are already here
                
        for input_key in input_signature.keys():
             
            #if this values was not already defined
            if not input_key in passed_keys: 
                
                single_input_signature : InputSignature = input_signature[input_key]
                                                   
                if not single_input_signature.default_value == None:
                    self.input[input_key] = single_input_signature.default_value  #the value used will be the default value

                elif not single_input_signature.generator == None:
                    self.input[input_key] = single_input_signature.generator(self) #generators have access to the instance              
                

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
            raise Exception(f"No validity verificator specified for key '{input_key}' and its type is of none of the possible types: {possible_types}")
          

# VALIDITY VERIFICATION (static methods for validating input) -----------------------------          

def requires_input_proccess(func):
    '''An annotation that makes the input be proccessed, if it was not already, when a function is called'''
    def wrapper(self : Component, *args, **kwargs):
        if not self._input_was_proccessed:
            self.proccess_input()
        return func(self, *args, **kwargs)
    return wrapper


class InputSignature():
    
    def __init__(self, default_value=None, generator=None, validity_verificator=None, possible_types : list = [], description=''):
        
        self.default_value = default_value
        self.generator = generator
        self.validity_verificator = validity_verificator
        self.possible_types = possible_types
        self.description = description
        
        




