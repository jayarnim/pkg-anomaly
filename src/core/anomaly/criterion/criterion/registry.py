CRITERION_REGISTRY = {}

def register(name):
    def wrapper(cls):
        CRITERION_REGISTRY[name] = cls
        return cls
    return wrapper