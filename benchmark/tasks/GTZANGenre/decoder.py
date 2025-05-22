# benchmark/tasks/GTZANGenre/decoder.py
from benchmark.core.registry import register
from benchmark.modules.decoders import MLPDecoder

@register("decoder", "GTZANGenre")
class GTZANGenreDecoder(MLPDecoder):
    pass
