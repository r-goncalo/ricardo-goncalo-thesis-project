

import os



def open_or_create_folder(dir, folder_name, create_new=True):
    
            
    try:
        folders_in_dir = os.listdir(dir) #is the dir already created? if not, error
        
    except:
        os.makedirs(dir)
        folders_in_dir = os.listdir(dir) #now 
    
    
    folder_exists = len([l for l in folders_in_dir if l == (f"{folder_name}")]) > 0
    
    if folder_exists and create_new: 
    
        number_of_versioned_folders = len([l for l in folders_in_dir if l.startswith(f"{folder_name}_")]) #counts the number of dirs that start with specified name
    
        folder_name = f"{folder_name}_{number_of_versioned_folders}"
    
    full_path = f"{dir}\\{folder_name}"

    os.makedirs(full_path)
    
    return full_path