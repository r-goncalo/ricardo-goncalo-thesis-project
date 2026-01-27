

from automl.core.debug.debug_utils import Component, Schema
from automl.utils.class_util import get_class_from


def make_new_debug_class(debug_class : type[Component], base_class : type[Component]):
    
    debug_class = get_class_from(debug_class)
    base_class = get_class_from(base_class)

    new_debug_class : type[Component] = Schema(
            f"{debug_class.__name__}_{base_class.__name__}",
            (debug_class, base_class),
            {
                "default_name": f"{base_class.default_name}_debug",
            }
        )
    
    return new_debug_class