

from automl.loggers.logger_component import DEBUG_LEVEL, LoggerSchema


_global_logger : list [LoggerSchema] = []


def activate_global_logger(global_logger_directory, global_logger_input={}):

    global _global_logger

    if len(_global_logger) > 0:
        print("WARNING: Tried to activate global logger after it was already activated, ignoring it...")

    else:
        if "artifact_relative_directory" not in global_logger_input.keys():
            global_logger_input["artifact_relative_directory"] = "global_logger"

        global_logger_input["base_directory"] = global_logger_directory

        _global_logger = [LoggerSchema(global_logger_input)]
        

def is_global_logger_active():
    
    global _global_logger
     
    return len(_global_logger) > 0

def globalWriteLine(string : str, file=None, level=DEBUG_LEVEL.INFO, toPrint=None, use_time_stamp=None, str_before='', ident_level=0):
    

        global _global_logger

        if is_global_logger_active():

            _global_logger[0].writeLine(string, file, level, toPrint, use_time_stamp, str_before, ident_level)
