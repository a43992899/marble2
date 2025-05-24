import pytest
import torch
import inspect

from marble.encoders.MERT.model import MERT_v1_95M_FeatureExtractor
from marble.modules.transforms import LayerSelector, TimeAvgPool


def test_mert_feature_extractor_init():
    """
    Test that MERT_v1_95M_FeatureExtractor __init__ accepts the expected parameters.
    """
    # Should initialize without error
    extractor = MERT_v1_95M_FeatureExtractor(pre_trained_folder=None, squeeze=True)
    # Check constructor signature for required args
    sig = inspect.signature(MERT_v1_95M_FeatureExtractor.__init__)
    params = sig.parameters
    assert 'pre_trained_folder' in params, "Expected __init__ to have 'pre_trained_folder' parameter"
    assert 'squeeze' in params, "Expected __init__ to have 'squeeze' parameter"


def test_mert_feature_extractor_forward_smoke():
    """
    Basic smoke test for forward pass of the feature extractor.
    """
    dummy_waveform = torch.randn(24000)
    dummy_sr = 24000
    extractor = MERT_v1_95M_FeatureExtractor(pre_trained_folder=None, squeeze=True)
    sample = {
        'waveform': dummy_waveform,
        'sampling_rate': dummy_sr
    }
    output = extractor(sample)
    # Should return a dict
    assert isinstance(output, dict)
    # Expect at least one tensor output: either 'waveform' or 'features'
    tensor_keys = [k for k, v in output.items() if isinstance(v, torch.Tensor)]
    assert tensor_keys, f"Expected at least one tensor in output, got keys {output.keys()}"


def test_layer_selector_parse_layers():
    # Integer list
    sel1 = LayerSelector(layers=[1, 3, 5])
    assert sel1.layers == [1, 3, 5]

    # String integers
    sel2 = LayerSelector(layers=['2', '4'])
    assert sel2.layers == [2, 4]

    # Range string
    sel3 = LayerSelector(layers=['0..2'])
    assert sel3.layers == [0, 1, 2]

    # Mixed types and ranges
    sel4 = LayerSelector(layers=[2, '3..4', '6'])
    assert sel4.layers == [2, 3, 4, 6]

    # Invalid range should raise
    with pytest.raises(ValueError):
        LayerSelector(layers=['5..3'])


def test_layer_selector_forward():
    batch_size, seq_len, hidden = 2, 10, 8
    num_layers = 6
    # Create dummy hidden_states: list of tensors [B, T, H]
    hidden_states = [torch.randn(batch_size, seq_len, hidden) for _ in range(num_layers)]
    layers = [1, 4]
    sel = LayerSelector(layers=layers)
    out = sel(hidden_states)
    # Expect shape: (batch_size, num_selected, seq_len, hidden)
    assert out.shape == (batch_size, len(layers), seq_len, hidden)
    # Check that selected slices match
    torch.testing.assert_allclose(out[:, 0], hidden_states[1])
    torch.testing.assert_allclose(out[:, 1], hidden_states[4])


def test_time_avg_pool():
    batch_size, num_layers, seq_len, hidden = 3, 5, 12, 16
    x = torch.randn(batch_size, num_layers, seq_len, hidden)
    pool = TimeAvgPool()
    out = pool(x)
    # Expect shape: (batch_size, num_layers, 1, hidden)
    assert out.shape == (batch_size, num_layers, 1, hidden)
    # Verify averaging
    expected = x.mean(dim=2, keepdim=True)
    torch.testing.assert_allclose(out, expected)
