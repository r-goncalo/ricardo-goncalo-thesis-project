

import os



def open_or_create_folder(dir, folder_name='', create_new=True):
    
    ''''''
    
    if folder_name == '':
        dir, folder_name = os.path.split(dir) #if the last folder/file name is not defined, extract it from dir
    
            
    try:
        folders_in_dir = os.listdir(dir) #is the dir already created? if not, error
        
    except:
        os.makedirs(dir)
        folders_in_dir = os.listdir(dir) #now 
    
    
    folder_exists = len([l for l in folders_in_dir if l == (f"{folder_name}")]) > 0
    
    full_path = os.path.join(dir, folder_name)
    
    if folder_exists and create_new: 
    
        number_of_versioned_folders = len([l for l in folders_in_dir if l.startswith(f"{folder_name}_")]) #counts the number of dirs that start with specified name
    
        folder_name = f"{folder_name}_{number_of_versioned_folders}"
    
        full_path = f"{dir}\\{folder_name}"

        os.makedirs(full_path)
    
    return full_path



def write_text_to_file(dir, filename, text : str, create_new=True):
    full_path = os.path.join(dir, filename)

    # If the file doesn't exist and create_new is True, create it
    if not os.path.exists(full_path):
        if create_new:
            with open(full_path, 'w') as f:
                f.write("")
        else:
            raise FileNotFoundError(f"{full_path} does not exist and create_new is False.")

    # Append the text to the file
    with open(full_path, 'a') as f:
        f.write(text + "\n")


    
def read_text_from_file(dir, filename):
    full_path = os.path.join(dir, filename)

    # Append the text to the file
    with open(full_path, 'r') as f:
        return f.read()


