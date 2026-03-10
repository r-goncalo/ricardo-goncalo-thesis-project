

import sys


_global_logger = None
DEFAULT_TO_PRINT_GLOBAL = False


def print_general_information():
        
        import automl
        import torch

        globalWriteLine(f"Global logger activation done, activated in {_global_logger.get_artifact_directory()}", toPrint=DEFAULT_TO_PRINT_GLOBAL)
        
        try:
            globalWriteLine(f"Python version is {sys.version_info[0]}, {sys.version_info[1]}")

        except:
            globalWriteLine(f"Could not note python version")

        globalWriteLine(f"Current version of automl is {automl.__version__}", toPrint=DEFAULT_TO_PRINT_GLOBAL)

        if torch.cuda.is_available():
            globalWriteLine(f"Cuda is available with version {torch.version.cuda}")
        
        else:
            globalWriteLine(f"Cuda is not available")


def activate_global_logger(global_logger_directory, global_logger_input : dict ={}):

    from automl.loggers.logger_component import LoggerSchema
    import automl

    if DEFAULT_TO_PRINT_GLOBAL:
        print(f"Global logger is trying to be activated in directory: {global_logger_directory}")


    global _global_logger

    if is_global_logger_active():
        print(f"WARNING: Tried to activate global logger after it was already activated in directory {_global_logger.get_artifact_directory()}")

    else:

        if "create_new_directory" not in global_logger_input.keys():
            global_logger_input["create_new_directory"] = False

        if "artifact_relative_directory" not in global_logger_input.keys():
            global_logger_input["artifact_relative_directory"] = "_global_logger"

        if "default_print" not in global_logger_input.keys():
            global_logger_input["default_print"] = DEFAULT_TO_PRINT_GLOBAL

        if "write_to_file_when_text_lines_over" not in global_logger_input.keys():
            global_logger_input["write_to_file_when_text_lines_over"] = -1 # global writes should be

        global_logger_input["base_directory"] = global_logger_directory

        _global_logger = LoggerSchema(global_logger_input)

        print_general_information()


def is_global_logger_active():
    
    global _global_logger
     
    return _global_logger != None


def get_global_level_artifact_directory():
     
    if not is_global_logger_active():
        return None
    
    else:
        return _global_logger.get_artifact_directory()
    
def get_global_logger():

    if not is_global_logger_active():
        return None
    
    else:
        return _global_logger

def globalWriteLine(string : str, file=None, toPrint=None, use_time_stamp=None, str_before='', ident_level=0):
    

    global _global_logger

    if is_global_logger_active():

        _global_logger.writeLine(string, file, toPrint=toPrint, use_time_stamp=use_time_stamp, str_before=str_before, ident_level=ident_level)
            