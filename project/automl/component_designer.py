
from .component import Component



    
def design(component_design):
    
    component = recursive_design(component_design)
    
    recursive_design_connector(component, component, component_design)
    
    return component



def recursive_design_connector(parent_component, current_component, component_design):
    
    '''Receives component system already designed and connects the components'''
    (_, _, component_input, connectors) = component_design
        
    for key in connectors.keys(): #connects current component with its components
        
        current_component.pass_input({key : look_for_component(parent_component, connectors[key])})    
        
                
    for key in component_input.keys(): #does the same for child components, recursively
        
        recursive_design_connector(component_input[key], parent_component, current_component[key], component_input[key])
        
        

def look_for_component(parent_component, full_localization):
    
    current_component = parent_component
    
    for input_key in full_localization:
        
        current_component = current_component.input[input_key]
        
    return current_component
    


def recursive_design(component_design):
    
    '''Receives component design and instantiates it'''
            
    (component_type, non_component_input, component_input, _) = component_design 
            
    component : Component = component_type(non_component_input)
    
    component.pass_input(non_component_input)
                
    for key in component_input.keys():
        
        input_component = design(component_input[key])
        
        component.pass_input({key : input_component})
                    
    
    return component


def print_design(component_design, ident_level=0):
    
    (component_type, non_component_input, component_input, connectors) = component_design
    
    space_before = ''
    for i in range(0, ident_level):
        space_before += '    '
        
    print(f'{space_before}("{component_type.__name__}",')
    
    print(space_before + '{', end='')
    
    if len(non_component_input) > 0:
        
        print()
        for key in non_component_input.keys():
            print(f'{space_before}  "{key}" : {non_component_input[key]}')
        
    print(space_before + '},')
    
    print(space_before + '{', end='')
    
    if len(non_component_input) > 0:
        
        print()
        for key in connectors.keys():
            print(f'{space_before}  "{key}" : {connectors.input[key]}')
    
    print(space_before + '},')    

    print(space_before + '{', end = '')
    
    if len(non_component_input) > 0:  
        
        print()
          
        for key in component_input.keys():
            print(f'{space_before} {key}: ') 
            print_design(component_input[key], ident_level + 1)
      
    print(space_before + '})')            
        
    
    
    
    
    