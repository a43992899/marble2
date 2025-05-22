# benchmark/tasks/GTZANGenre/decoder.py
from marble.core.registry import register
from marble.modules.decoders import MLPDecoder

@register("decoder", "GTZANGenre")
class GTZANGenreDecoder(MLPDecoder):
    pass
