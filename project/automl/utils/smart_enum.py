

from enum import Enum

class SmartEnum(Enum):
    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        if isinstance(value, str):
            name = value.strip().upper()
            if name in cls.__members__:
                return cls[name]
            if name.lstrip('-').isdigit():
                return cls(int(name))
        raise ValueError(f"Cannot convert {value!r} to {cls.__name__}")