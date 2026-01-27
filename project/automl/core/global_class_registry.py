import sys
import types
from typing import Type
import inspect

from automl.component import Schema
from automl.loggers.global_logger import globalWriteLine

from dataclasses import dataclass
from typing import Callable

CUSTOM_MODULE_NAME = "__custom_classes.custom_classes"


@dataclass
class ClassSpec:
    name : str
    generator: Callable
    args: tuple
    kwargs: dict

_CUSTOM_CLASS_SPECS: dict[str, ClassSpec] = {}

def register_class_generator(
    *,
    generator: Callable,
    args: tuple = (),
    kwargs: dict | None = None,
):
    if kwargs is None:
        kwargs = {}

    module = get_or_create_custom_module()

    cls = generator(*args, **kwargs)
    cls.__module__ = CUSTOM_MODULE_NAME

    setattr(module, cls.__name__, cls)

    spec = ClassSpec(
        name=cls.__name__,
        generator=generator,
        args=args,
        kwargs=kwargs,
    )

    key = f"{CUSTOM_MODULE_NAME}.{cls.__name__}"
    _CUSTOM_CLASS_SPECS[key] = spec

    globalWriteLine(f"Registered custom class: {key}", file="global_classes.txt")

    return cls


def get_registered_classes_generators():
    return dict(_CUSTOM_CLASS_SPECS)

def has_registered_classes_generators():
    return len(_CUSTOM_CLASS_SPECS) > 0

def clear_registry():
    _CUSTOM_CLASS_SPECS.clear()


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

    new_cls = Schema(
        cls.__name__,
        cls.__bases__,
        dict(cls.__dict__)
    )

    new_cls.__module__ = target_module_name

    if hasattr(cls, "__source__"):
        new_cls.__source__ = cls.__source__

    return new_cls


def serialize_registered_classes() -> str:

    lines = [
        "# Auto-generated custom classes",
        "from automl.component import Component",
        "from automl.schema import Schema",
        "from automl.utils.class_util import get_class_from",
        "from automl.core.global_class_registry import register_class_generator",
        "",
    ]

    for spec in _CUSTOM_CLASS_SPECS.values():
        gen = spec.generator

        try:
            gen_src = inspect.getsource(gen)
        except (OSError, TypeError):
            raise RuntimeError(
                f"Generator {gen.__name__} has no source; cannot serialize"
            )
        
        args_str = [f"\"{str(arg)}\""for arg in spec.args]
        args_str = ", ".join(args_str)

        lines.append(gen_src)
        lines.append("")
        lines.append(
            f"{spec.name} = register_class_generator("
            f"generator={gen.__name__}, "
            f"args=({args_str}), "
            f"kwargs={spec.kwargs}"
            f")"
        )
        lines.append("")

    return "\n".join(lines)

def load_custom_classes(file):

    module_name = f"_automl_custom_classes"

    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file)
    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module
    spec.loader.exec_module(module)