DATASET_REGISTRY = {}

def register(name):
    def wrapper(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return wrapper