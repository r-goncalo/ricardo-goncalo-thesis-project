



_global_logger = None


def activate_global_logger(global_logger_directory, global_logger_input : dict ={}):

    from automl.loggers.logger_component import LoggerSchema


    global _global_logger

    if is_global_logger_active():
        print("WARNING: Tried to activate global logger after it was already activated, ignoring it...")

    else:

        if "create_new_directory" not in global_logger_input.keys():
            global_logger_input["create_new_directory"] = False

        if "artifact_relative_directory" not in global_logger_input.keys():
            global_logger_input["artifact_relative_directory"] = "_global_logger"

        global_logger_input["base_directory"] = global_logger_directory

        _global_logger = LoggerSchema(global_logger_input)
        

def is_global_logger_active():
    
    global _global_logger
     
    return _global_logger != None


def get_global_level_artifact_directory():
     
    if not is_global_logger_active():
        return None
    
    else:
        return _global_logger.get_artifact_directory()

def globalWriteLine(string : str, file=None, toPrint=None, use_time_stamp=None, str_before='', ident_level=0):
    

        global _global_logger

        if is_global_logger_active():

            _global_logger[0].writeLine(string, file, toPrint=toPrint, use_time_stamp=use_time_stamp, str_before=str_before, ident_level=ident_level)
