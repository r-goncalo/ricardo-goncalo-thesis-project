from automl.loggers.logger_component import DEBUG_LEVEL, ComponentWithLogging
from automl.ml.memory.memory_components import MemoryComponent
from automl.component import requires_input_proccess


class MemoryDebug(MemoryComponent, ComponentWithLogging):

    is_debug_schema = True


    def _proccess_input_internal(self):
        super()._proccess_input_internal()
    
    
    def push(self, transition):    

        super().push(transition)

        str_pushing = ' '

        for field_name in self.field_names:
                        
            str_pushing = f"{str_pushing}{field_name}: {transition[field_name]} "

        self.lg.writeLine(f"Pushing: {str_pushing}", file="pushed_transitions.txt", use_time_stamp=False, level=DEBUG_LEVEL.DEBUG)

    @requires_input_proccess
    def clear(self):
        
        self.lg.writeLine(f"\nBefore cleaning ({len(self)}) transitions, noting their values...\n", file="on_clear_transitions.txt", use_time_stamp=False)

        for i in range(len(self)):

            str_pushing = ' '
    
            for field_name in self.field_names:
                            
                str_pushing = f"{str_pushing}{field_name}: {self.transitions[field_name][i]} " # TODO: there should be a clear way to get a single transition
    
            self.lg.writeLine(f"{str_pushing}", file="on_clear_transitions.txt", use_time_stamp=False, level=DEBUG_LEVEL.DEBUG)

        super().clear()