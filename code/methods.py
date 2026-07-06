from augment import mixup, cutmix
import random


def apply_method(method, x, y):

    if method == "baseline":
        return x, y, None, None

    elif method == "mixup":
        return mixup(x, y)

    elif method == "cutmix":
        return cutmix(x, y)

    elif method == "finetune":
        return x, y, None, None

    else:
        raise ValueError("Unknown method")