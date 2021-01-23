"""Module contains preprocessing functions for correlations before they are used as weights """

from numpy import finfo

PRECISION_EPSILON = finfo(float).eps


def truncate_below_threshold(num: float, threshold: float = 0.5) -> float:
    """
    :param num: a correlation float
    :param threshold: threshold under which correlations should be disregarded
    :return: either the original correlation, or zero
    """
    if abs(num) < threshold:
        return 0

    return num


def inverse_distance_to_1(num: float) -> float:
    """
    :param num: a correlation float
    :return: a positive weight that tends to +inf as correlation approaches 1
    """
    return 1 / (1 - num + PRECISION_EPSILON)


def mixed_truncate_inverse_distance(num: float, threshold: float = 0.9) -> float:
    """
    :param num: a correlation float
    :param threshold: threshold under which correlations should be disregarded
    :return: either a positive weight that tends to +inf as correlation approaches 1, or zero
    """
    if abs(num) < threshold:
        return 0

    return num / (1 - num + PRECISION_EPSILON)
