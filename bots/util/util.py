def clamp11(value):
    return max(-1, min(1, value))

def clamp01(value):
    return max(0, min(1, value))

def transform_clamp(value, forwards: bool):
    if forwards: return (value + 1) / 2
    return value * 2 - 1
