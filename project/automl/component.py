from automl.core.input_management import InputMetaData, InputSignature
from types import FunctionType
import copy
from abc import ABCMeta

# Reserved attributes: input, values, parameters_signature, exposed_values, output, _input_was_proccessed

def on_name_pass(self):
    self.name = self.input["name"] #sets the name of the component to the input name
    

#TODO: this should make all pre computation necessary for input processing
class Scheme(ABCMeta): # the meta class of all component classes, defines their behavior (not the behavior of the instances)
    
    def __init__(self_class, *args, **kwargs):
        # Create the new class
        super().__init__(*args, **kwargs)
                
    
    def __prepare__(cls_name, bases, **kwds): #all Schemes have a parameter_signature and exposed_values
        
        return {
            "parameters_signature": {},
            "exposed_values": {}
        }


class Component(metaclass=Scheme): # a component that receives and verifies input

    '''This is the basic Squema, all other Squemas should extend it'''
    
    # a dictionary with { "input_name" : (default_value, validity verification) }
    # if default_value is None, an exception error is raised when input is missing
    # if validity verification function is not none, it will be applied to the input value
    # the actual input values will be saved in self.input
    # worth noting that child components can freely repeat values in their parameters_signature.
    #   Default values and generators of child components will have priority
    #   Validity and type checkers will be applied from Parent to child     
    
    parameters_signature : dict[str, InputSignature] = {
        "name" : InputSignature(on_pass=on_name_pass, ignore_at_serialization=True, mandatory=False, priority=0), #the name of the component
    }
    
    #A dictionary { "value_name" -> initial_value }
    #it tells other components what are the values exposed by this component, useful when checking the validity of the program before running it
    #this does not need to be stored in the component, the only reason it is is to standardize this kind of exposure
    exposed_values : dict[str, any] = {}
    
    
    # INITIALIZATION -------------------------------------------------------------------------
    
    def __init__(self, input : dict[str, any] = {}): #we can immediatly receive the input
                        
        self.input : dict[str, any] = {} #the input will be a dictionary
        
        self.__input_meta : dict[str, InputMetaData] = {} # will store meta data according to the input
        self.__initialize_input_meta_data()
        
        self.__set_exposed_values_with_super() #updates the exposed values with the ones in super classes
        self.values = self.exposed_values.copy() #this is where the exposed values will be stored
        
        self.child_components : list[Component] = []
        self.parent_component : Component = None
        
        self.name = str(type(self).__name__) #defines the initial component name

        self.pass_input(input) #passes the input but note that it does not proccess it
        
        self.output = {} #output, if any, will be a dictionary
        
        self._input_was_proccessed = False #to track if the instance has had its input proccessing before any operations that needed it
                
            
    
    def pass_input(self, input: dict): # pass input to this component, may need verification of input
        '''Pass input to this component
           It is supposed to be called only before proccess_input and, if not, it mean proccess_input may be called a multitude of times
           '''
        
        if not isinstance(input, dict):
            raise Exception("Passed input is not of type dict")
           
        self._input_was_proccessed = False #when we pass new input, it means that we need to proccess it again
        
        for passed_key in input.keys():
            
            parameter_signature = self.get_parameter_signature(passed_key)
            
            if parameter_signature != None:
                
                self.__verified_pass_input(passed_key, input[passed_key])
                
            else:
                print(f"WARNING: input with key {passed_key} passed to component {self.name} but not in its input signature, will be ignored")

                
                
    def __verified_pass_input(self, key, value):
        
        '''for logic of passing the input, already verified'''
        
        self.input[key] = value
        self.__input_meta[key].custom_value_passed()
        
        parameters_signatures_of_key : list[InputSignature] = self.get_list_of_parameter_signatures_for_key(key)
        
        for parameter_signature in parameters_signatures_of_key: # TODO: this should all be defined in the creation of the Scheme, not computed every time an input is passed
        
            if parameter_signature.on_pass != None: #if there is verification of logic for when the input is passed

                try:
                
                    parameter_signature.on_pass(self)

                except:
                    raise Exception(f"In component of type {type(self)}, when passing input: Exception while using the on_pass for {key}, named {parameter_signature.on_pass.__name__}")
                
            

    def proccess_input_internal(self): #verify the input to this component
        '''
        Verify validity of input and add default values, following initializing the attributes
        This can and should be extended by child Schemas
        ''' 
        
        
        #get input signature priorities specified, sorted
        #and the input signatures organized by priorities
        parameters_signature_priorities, organized_parameters_signature = self.__get_organized_parameters_signature()
        
        passed_keys = self.input.keys()
        
        
        for priority in parameters_signature_priorities:
                        
            self.__add_default_values_of_class_input(passed_keys, organized_parameters_signature[priority])
            
            self.__verify_input(passed_keys, organized_parameters_signature[priority])            
            
        self._input_was_proccessed = True
        
    def proccess_input(self):
        '''
        This method is called by external parties
        Instead of extending it, extend proccess_input_internal
        '''
        self.proccess_input_internal()
        self.post_proccess_input()
        
    
    def post_proccess_input(self):
        '''Called after the input was proccessed, to do any post processing necessary'''
        
        pass
    
        
    def proccess_input_if_not_proccesd(self): 
           
        if not self._input_was_proccessed:
            
            self.proccess_input()


    # OUTPUT -------------------------
    
    def get_output(self): #return output
        return self.output
    
    # CHILD AND PARENT COMPONENTS ---------------------------------
    
    def initialize_child_component(self, component_type : type, input : dict ={}):
        
        '''Explicitly initializes a component of a certain type as a child component of this one'''
        
        initialized_component : Component = component_type(input)
        
        self.define_component_as_child(initialized_component)
        
        return initialized_component
    
    
    def define_component_as_child(self, new_child_component):
        
        '''Defines target component as being child component of this one'''
        
        self.child_components.append(new_child_component)
        
        new_child_component.parent_component = self
        
        new_child_component.on_parent_component_defined()
        
    def on_parent_component_defined(self):
        '''called after this component as another parent component defined'''
        pass
        
    
    def get_attr_from_parent(self, attr_name : str):
        '''Gets an attribute from a parent component, None if non existent'''
                
        if self.parent_component == None:
            return None
        
        else:
            if hasattr(self.parent_component, attr_name):
                return getattr(self.parent_component, attr_name)
            else:
                return self.parent_component.get_attr_from_parent(attr_name)
            
    
    def get_child_by_name(self, name):
        
        '''Gets child component, looking for it by its name'''
                
        if self.name == name:
            return self
        
        for child_component in self.child_components:
            
            to_return = child_component.get_child_by_name(name)
            
            if to_return != None:
                return to_return
            
        return None
    
    def get_child_by_localization(self, localization : list):
        
        '''
        Gets child component by its location
        Note that an emty localization will return the component itself
        '''
        
        current_component : Component = self

        for index in localization:
            
            if isinstance(index, int):
                current_component = current_component.child_components[index]

            elif isinstance(index, str):
                current_component = current_component.get_child_by_name(index)
            
            else:
                raise Exception(f"Invalid index {index} in localization {localization} for component {self.name}")

        return current_component
    
    def get_child_component(self, component_localizer):
        
        if isinstance(component_localizer, str):
                    
            return self.get_child_by_name(component_localizer)
        
        elif isinstance(component_localizer, int):
            
            return self.get_child_by_localization([component_localizer])
                    
        elif isinstance(component_localizer, list):
                    
            return self.get_child_by_localization(component_localizer)
        
        raise Exception(f"Could not find component in source_component {self.name} given localization {component_localizer}")
                
    
    def get_index_localization(self):
        
        '''Gets localization of this component, stopping the definition of localization when it finds a source componen (without parent)'''
        
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
    
    
    def get_source_component(self):
        '''Gets the source component, the one without parent'''
        
        current_component = self
        
        while True:
            
            if current_component.parent_component != None:
                current_component = current_component.parent_component
            else:
                return current_component
    
    # CLONING -------------------------------------------------
    
    def clone(self):
        
        '''
        Creates a clone of this component, with the same input and exposed values.
        Not that the clone will not have the same parent component
        '''
        
        cloned_component = type(self)(copy.deepcopy(self.input))
        cloned_component.values = copy.deepcopy(self.values) #copy the exposed values
        cloned_component.output = copy.deepcopy(self.output) #copy the output
        
        return cloned_component
    
    # EXPOSED VALUES -------------------------------------------
    
    # TODO : maybe this should happen at the end of class definition for all classes that extend Component, statically? Instead of happening multiple times per class redundantly
    def __set_exposed_values_with_super(self):
        
        '''Updates the exposed values with super classes'''
                
        current_class_index = 0
        method_resolution_list = type(self).__mro__
        current_squeme : type[Component] = method_resolution_list[current_class_index]
        
        while True:
            
            self.exposed_values =  {**current_squeme.exposed_values, **self.exposed_values} #updates the exposed values with the exposed values in super classes
            
            if current_squeme == Component:
                break #this is the last squeme
            
            else:
                current_class_index += 1
                current_squeme : type[Component] = method_resolution_list[current_class_index]



    # INPUT PROCCESSING ---------------------------------------------      
    
    def get_list_of_parameter_signatures_for_key(self, key) -> list[InputSignature]:
        
        current_class_index = 0
        method_resolution_list = type(self).__mro__
        current_squeme : type[Component] = method_resolution_list[current_class_index]
        
        toReturn : list[InputSignature]= []
        
        while True:
            
            if key in current_squeme.parameters_signature.keys():
                toReturn.append(current_squeme.parameters_signature[key])
                
                if current_squeme == Component:
                    break 
                
                current_class_index += 1
                current_squeme : type[Component] = method_resolution_list[current_class_index]
            
            elif current_squeme == Component:
                break 
            
            else:
                current_class_index += 1
                current_squeme : type[Component] = method_resolution_list[current_class_index]
        
        return toReturn
    
    
    
    @classmethod
    def get_schema_parameter_signature(cls, key) -> InputSignature:
        
        '''Returns the parameter signature for a key for this Schema'''
        
        current_class_index = 0
        method_resolution_list = cls.__mro__
        current_squeme : type[Component] = method_resolution_list[current_class_index]
        
        while True:
            
            if key in current_squeme.parameters_signature.keys():
                return current_squeme.parameters_signature[key]
            
            elif current_squeme == Component:
                break #this will return false
            
            else:
                current_class_index += 1
                current_squeme : type[Component] = method_resolution_list[current_class_index]
            
        return None #there was no key
    
    

    def get_parameter_signature(self, key) -> InputSignature:
        
        '''Gets the parameter signature for a key for this component'''
        
        return type(self).get_schema_parameter_signature(key) #uses the class method to get the parameter signature for a key
    
    def in_parameters_signature(self, key): #checks if the key is in input signature of component or its parents components
        
        return self.get_parameter_signature(key) != None
    
    
    def __proccess_squeme_from_priorities(self, current_squeme, parameters_signature_priorities : list[int], organized_parameters_signatures : dict[int, list[InputSignature]]):
    
        current_parameters_signature : dict[str, InputSignature] = current_squeme.parameters_signature #get its input signature 
        
        for key, parameter_signature in current_parameters_signature.items():
            
            priority = parameter_signature.priority
            
            if not priority in parameters_signature_priorities: #if this is the first key for that priority, initialize the list for that priority
                parameters_signature_priorities.append(priority)
                organized_parameters_signatures[priority] = []
                
            organized_parameters_signatures[priority].append((key, parameter_signature)) #put its key, parameter_signature pair in the list of respective priority                

    
    
    # TODO: This is missing the capability of a Schematic re-writing the InputSignature of its super
    def __get_organized_parameters_signature(self): 
        
        current_class_index = 0
        method_resolution_list = type(self).__mro__
        current_squeme : type[Component] = method_resolution_list[current_class_index]
        
        parameters_signature_priorities : list[int] = [] #all priorities defined
        organized_parameters_signatures : dict[int, list[InputSignature]] = {} #InputSignatures organized by priorities
        
        while True: #for each component class
            
            self.__proccess_squeme_from_priorities(current_squeme, parameters_signature_priorities, organized_parameters_signatures)              
            
            if current_squeme == Component:
                break #this is the last squeme
            
            else:
                current_class_index += 1
                current_squeme : type[Component] = method_resolution_list[current_class_index]

        parameters_signature_priorities.sort()
        
        return parameters_signature_priorities, organized_parameters_signatures
        
            
    # DEFAULT VALUES -------------------------------------------------
            
    def __add_default_values_of_class_input(self, passed_keys, list_of_signatures : list[tuple[int, InputSignature]]):
                        
        for (input_key, parameter_signature) in list_of_signatures:
                             
            #if this values was not already defined
            if not input_key in passed_keys: 
                
                if parameter_signature.get_from_parent:
                    
                    self.input[input_key] = self.get_attr_from_parent(input_key)
                    
                    if self.input[input_key] == None:
                        raise Exception(f"In component of type {type(self)}, when cheking for the inputs: Getting attribute from parent resulted in a None value")
                    
                                                                   
                elif not parameter_signature.default_value == None:
                    self.input[input_key] = parameter_signature.default_value  #the value used will be the default value

                elif not parameter_signature.generator == None:
                    try:
                        self.input[input_key] = parameter_signature.generator(self) #generators have access to the instance        
                    
                    except:
                        raise Exception(f"In component of type {type(self)}, when cheking for the inputs: Exception while using the generator for {input_key}, named {parameter_signature.generator.__name__}")
                    
    
    # INPUT META DATA ----------------------------------------------------------------
    
    def __initialize_input_meta_data(self):
        
        '''Initializes the dictionary __input_meta'''
            
        current_class_index = 0
        method_resolution_list = type(self).__mro__
        current_squeme : type[Component] = method_resolution_list[current_class_index]
        
        while True:
            
            for key in current_squeme.parameters_signature.keys():
                self.__input_meta[key] = InputMetaData(parameter_signature=current_squeme.parameters_signature[key])
            
            if current_squeme == Component:
                break #this is the last squeme
            
            else:
                current_class_index += 1
                current_squeme : type[Component] = method_resolution_list[current_class_index]
        

            
            
    def get_input_meta(self) -> dict[str, InputMetaData]:
        return self.__input_meta
        
    # VALIDITY VERIFICATION ---------------------------------------------

    def __verify_input(self, passed_keys, list_of_signatures : list[tuple[int, InputSignature]]): #verify the input to this component and add default values, to the self.input dict

        for (input_key, parameter_signature) in list_of_signatures:
                                    
            if input_key in passed_keys: #if this value was in input
                
                input_value = self.input[input_key] #get the value passed
                    
                self.verify_validity(input_key, input_value, parameter_signature.possible_types, parameter_signature.validity_verificator) #raises exceptions if input is not valid
                                    
            elif parameter_signature.mandatory: #if there was no specified value for this attribute in the input
                raise Exception(f"In component of type {type(self)}, when cheking for the inputs: Did not set input for mandatory key '{input_key}' and has no default value nor generator")     


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
            raise Exception(f"In component of type {type(self)}: No validity verificator specified for key '{input_key}' and its type ({type(input_key)}) is of none of the possible types: {possible_types}")
          

# VALIDITY VERIFICATION (static methods for validating input) -----------------------------          

def requires_input_proccess(func : FunctionType):
    '''
    An annotation that makes the input be proccessed, if it was not already, when a function is called
    
    Note that if a method has its super method with this annotation, adding it will be redundant
    '''
        
    def wrapper(self : Component, *args, **kwargs):
        self.proccess_input_if_not_proccesd()
        return func(self, *args, **kwargs)
    
    return wrapper


def uses_component_exception(func):
    '''A wrapper for functions so its errors have more information regarding the component they appeared in'''

    def wrapper(self : Component, *args, **kwargs):
        
        try:
            return func(self, *args, **kwargs)
        
        except Exception as e:
            original_args = e.args #arguments of an error
            e.args = (f"On Component {self.name} of type {type(self).__name__}:\n {original_args[0]}", *original_args[1:])
            raise e
            
    return wrapper
        
        


