"""Easing curves: map progress 0..1 to 0..1 (some overshoot on the way)."""
import math


def linear(p):
    return p


def smooth(p):
    return p * p * (3 - 2 * p)


def out_cubic(p):
    return 1 - (1 - p) ** 3


def out_back(p, k=1.7):
    """Overshoots slightly, then settles."""
    return 1 + (k + 1) * (p - 1) ** 3 + k * (p - 1) ** 2


def spring(p):
    """Damped wobble around the target."""
    return 1 - math.exp(-6 * p) * math.cos(3 * math.pi * p)
