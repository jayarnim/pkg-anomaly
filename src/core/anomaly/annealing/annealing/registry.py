ANNEALING_REGISTRY = {}

def register(name):
    def wrapper(cls):
        ANNEALING_REGISTRY[name] = cls
        return cls
    return wrapper