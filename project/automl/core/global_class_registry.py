import sys
import types
from typing import Type
import inspect

from automl.loggers.global_logger import globalWriteLine

CUSTOM_MODULE_NAME = "__custom_classes.custom_classes"


_CUSTOM_CLASSES: dict[str, Type] = {}

def register_class(cls: Type):

    '''Clones a class into the custom module and returns the clone'''

    module = get_or_create_custom_module()

    cloned_cls = clone_class_into_module(cls, CUSTOM_MODULE_NAME)

    setattr(module, cloned_cls.__name__, cloned_cls)

    key = f"{CUSTOM_MODULE_NAME}.{cloned_cls.__name__}"
    _CUSTOM_CLASSES[key] = cloned_cls

    globalWriteLine(f"Registered custom class: {key}", file="global_classes.txt")

    return cloned_cls

def get_registered_classes():
    return dict(_CUSTOM_CLASSES)

def has_registered_classes():
    return len(_CUSTOM_CLASSES) > 0

def clear_registry():
    _CUSTOM_CLASSES.clear()


def get_or_create_custom_module():

    if CUSTOM_MODULE_NAME in sys.modules:
        return sys.modules[CUSTOM_MODULE_NAME]

    pkg_name = "__custom_classes"

    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = []  # mark as package
        sys.modules[pkg_name] = pkg

    module = types.ModuleType(CUSTOM_MODULE_NAME)
    sys.modules[CUSTOM_MODULE_NAME] = module
    return module

def clone_class_into_module(cls, target_module_name: str):
    namespace = dict(cls.__dict__)

    # Remove runtime-only attributes
    namespace.pop("__dict__", None)
    namespace.pop("__weakref__", None)

    new_cls = type(
        cls.__name__,
        cls.__bases__,
        namespace
    )

    new_cls.__module__ = target_module_name
    return new_cls

def serialize_registered_classes() -> str:
    lines = []
    lines.append("# Auto-generated custom classes")
    lines.append("# DO NOT EDIT MANUALLY\n")

    for name, cls in get_registered_classes().items():
        try:
            source = inspect.getsource(cls)
        except OSError:
            raise RuntimeError(f"Cannot serialize class {cls}")

        lines.append(source)
        lines.append("\n")

    return "\n".join(lines)

def load_custom_classes(file):

    module_name = f"_automl_custom_classes"

    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file)
    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module
    spec.loader.exec_module(module)