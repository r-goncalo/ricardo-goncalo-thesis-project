from automl.core.input_management import InputSignature
from abc import ABCMeta


    
#TODO: this should make all pre computation necessary for input processing
class Schema(ABCMeta): # the meta class of all component classes, defines their behavior (not the behavior of the instances)
    
    def __init__(self_class, name, bases, namespace):
        
        # Create the schema, with its defined attributes
        super().__init__(name, bases, namespace)

        self_class.__set_exposed_values_with_super(bases)
        self_class.__set_parameter_signatures_with_super(bases)
        self_class.__setup_default_values_in_parameter_signatures()
        self_class.__setup_organized_parameters_signature()

    def __prepare__(cls_name, bases, **kwds): #all Schemes have a parameter_signature and exposed_values
        
        return {
            "parameters_signature": {},
            "exposed_values": {}
        }
        

    def __set_exposed_values_with_super(self_class, bases):
        
        '''
        Updates the exposed values with super classes for this squema
        This assumes those squemas also had their exposed values processed, and so it only needs to deal with the super
        '''

        for base_class in bases:

            self_class.exposed_values =  {**self_class.exposed_values, **base_class.exposed_values}



    def __set_parameter_signatures_with_super(self_class, bases):
        
        '''
        Updates the exposed values with super classes for this squema
        This assumes those squemas also had their exposed values processed, and so it only needs to deal with the super
        '''

        self_parameters_signature : dict[str, InputSignature] = self_class.parameters_signature

        # for each of the explicitly defined super classes
        for base_class in bases:

            # we look into its parameters signature
            base_class_parameters_signature : dict[str, InputSignature] = base_class.parameters_signature

            for input_key in base_class_parameters_signature.keys():

                base_class_signature = base_class_parameters_signature[input_key]

                # if input key 
                if input_key in self_parameters_signature.keys():

                    self_signature = self_parameters_signature[input_key]

                    self_parameters_signature[input_key] = base_class_signature.fuse_with_new(self_signature)

                else:

                    self_parameters_signature[input_key] = base_class_signature



    def __setup_default_values_in_parameter_signatures(self_class):

        self_parameters_signature : dict[str, InputSignature] = self_class.parameters_signature

        for parameter_signature in self_parameters_signature.values():

            parameter_signature.setup_default_values() 

    
    def __setup_organized_parameters_signature(self_class):

        '''
        Sets up the parameters signatures organized by priority
        '''

        self_class.parameters_signature_priorities : list[int] = [] #all priorities defined
        self_class.organized_parameters_signatures : dict[int, list[InputSignature]] = {} #InputSignatures organized by priorities

        self_parameters_signature : dict[str, InputSignature] = self_class.parameters_signature

    
        for key, parameter_signature in self_parameters_signature.items():
            
            priority = parameter_signature.priority
            
            if not priority in self_class.parameters_signature_priorities: #if this is the first key for that priority, initialize the list for that priority
                self_class.parameters_signature_priorities.append(priority)
                self_class.organized_parameters_signatures[priority] = {}
                
            self_class.organized_parameters_signatures[priority][key] = parameter_signature #put its key, parameter_signature pair in the list of respective priority  

        self_class.parameters_signature_priorities.sort()