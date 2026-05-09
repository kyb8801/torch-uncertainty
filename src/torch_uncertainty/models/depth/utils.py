from enum import Enum


class BackboneName(str, Enum):
    """Enum of supported BTS backbone names."""

    DENSENET121 = "densenet121"
    DENSENET161 = "densenet161"
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    RESNEXT50 = "resnext50"
    RESNEXT101 = "resnext101"
