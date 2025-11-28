

from automl.fundamentals.translator.translator import Translator
from automl.core.input_management import InputSignature
from automl.utils.shapes_util import torch_shape_from_space
import torch

class ImageReverter(Translator):

    parameters_signature = {
                       "device" : InputSignature(default_value="", ignore_at_serialization=True, get_from_parent=True)
        }    

    def _proccess_input_internal(self):
        
        super()._proccess_input_internal()

        self.device = self.get_input_value("device")


    def translate_state(self, state):

        """
        Transforms a pixel observation (H,W,C) -> torch(C,H,W) normalized.
        """

        if state is None:
            return None

        return (
            torch.tensor(state, dtype=torch.float32, device=self.device)
            .permute(2, 0, 1) / 255.0
        )

    def get_shape(self, original_shape):


        shape = torch_shape_from_space(original_shape)  # torch.Size([H, W, C])

        if len(shape) != 3:
            raise ValueError(f"ImageReverter expected a 3D shape (H,W,C), got: {shape}")

        H, W, C = shape
        return torch.Size([C, H, W])