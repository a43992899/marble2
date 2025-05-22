_REGISTRIES = {
    "encoder": {},
    "decoder": {},
    "datamodule": {},
    "task": {},
    "metric": {},
    "postprocess": {},
    "transforms": {},
}

def register(kind, name=None):
    """Decorator: @register("encoder")"""
    def _wrap(cls):
        _REGISTRIES[kind][name or cls.__name__] = cls
        return cls
    return _wrap

def get(kind, name):
    return _REGISTRIES[kind][name]
