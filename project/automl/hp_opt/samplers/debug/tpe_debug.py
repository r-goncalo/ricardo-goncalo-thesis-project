from automl.loggers.debug.component_with_logging_debug import ComponentWithLoggingDebug
import optuna

class LoggingTPESampler(optuna.samplers.TPESampler, ComponentWithLoggingDebug):

    def sample_relative(self, study, trial, search_space):
        result = super().sample_relative(study, trial, search_space)

        if hasattr(self, "_search_space_group"):
            groups = self._search_space_group._groups
            self.lg.writeLine("Groups:", groups)

        return result