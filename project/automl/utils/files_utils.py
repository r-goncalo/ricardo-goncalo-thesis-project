

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

    elif not folder_exists:

        os.makedirs(full_path)
    
    return full_path

def new_path_if_exists(specific_path, dir = ''):

    '''Generates a string for a path, the specific path is what is used to version the path'''

    full_path = os.path.join(dir, specific_path)


    if os.path.exists(full_path): #file with that name already existed

        paths_in_dir = os.listdir(dir)

        if os.path.isfile(full_path):
            
            filename, ext = os.path.splitext(specific_path)

            # find existing versions like filename_1.ext, filename_2.ext, ...
            existing_versions = [
                f for f in paths_in_dir
                if f.startswith(filename + "_") and f.endswith(ext)
            ]

            version_number = len(existing_versions) + 1
            specific_path = f"{filename}_{version_number}{ext}"
            full_path = os.path.join(dir, specific_path)

        else: # if is dir
        
            number_of_versioned_paths = len([l for l in paths_in_dir if l.startswith(f"{specific_path}_")])

            specific_path = f"{specific_path}_{number_of_versioned_paths}"

            full_path = os.path.join(dir, specific_path)

    return full_path




def write_text_to_file(dir = '', filename = '', text : str = '', create_new=True):
    
    full_path = os.path.join(dir, filename)
    
    dir = os.path.dirname(full_path)
    
    os.makedirs(dir, exist_ok=True)
    
    # If the file exists and create_new is True, delete old and write new
    if os.path.exists(full_path):
        
        if create_new:
                        
            with open(full_path, 'w') as f:
                f.write(text)
        
        else:
            
            with open(full_path, 'a') as f:
                f.write(text)

    else:
        with open(full_path, 'w') as f: # write to file
            f.write(text)


    
def read_text_from_file(dir, filename):
    full_path = os.path.join(dir, filename)

    # Append the text to the file
    with open(full_path, 'r') as f:
        return f.read()


def saveDataframe(df, directory='', filename='dataframe.csv'): #saves a dataframe using this log object as a reference
        
        '''
        Saves dataframe in artifact directory
        This does not trigger input processing
        '''
        
        if(directory != ''):
            open_or_create_folder(directory, create_new=False)

        df.to_csv(os.path.join(directory, filename), index=False)