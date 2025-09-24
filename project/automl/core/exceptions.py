

import os
from automl.loggers.logger_component import DEBUG_LEVEL, ComponentWithLogging


class NotInComponentTree(Exception):
    '''Component not found in tree of another component'''






def common_exception_handling(component : ComponentWithLogging, exception, error_report_path):

    '''Common exception handling for components with logging'''

    import traceback
        
    error_message = str(exception)
    full_traceback = traceback.format_exc()

    component.lg.writeLine("Error message:", file=error_report_path, level=DEBUG_LEVEL.ERROR)
    component.lg.writeLine(error_message, file=error_report_path, level=DEBUG_LEVEL.ERROR)

    component.lg.writeLine("\nFull traceback:", file=error_report_path, level=DEBUG_LEVEL.ERROR)
    component.lg.writeLine(full_traceback, file=error_report_path, level=DEBUG_LEVEL.ERROR)