# Separate file to avoid problems due to circular imports

from luxonis_train.models.heads import *


def get_head_label_types(head_str: str):
    """Returns all label types defined as head class attributes"""
    return eval(head_str).label_types