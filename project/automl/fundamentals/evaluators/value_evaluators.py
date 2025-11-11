from automl.basic_components.evaluator_component import EvaluatorComponent
from automl.component import Component
from automl.core.input_management import InputSignature


class ValueEvaluator(EvaluatorComponent):

    parameters_signature = {
        "value_to_use" : InputSignature(description="The exposed value to use")
    }

    def _proccess_input_internal(self):
        super()._proccess_input_internal()

        self.value_to_use = self.get_input_value("value_to_use")

    def _evaluate(self, component_to_evaluate : Component) -> dict:
        return {"result": component_to_evaluate.values[self.value_to_use] }