# marble/encoders/MERT/model.py
from typing import Sequence, Dict, Optional, Union, Tuple, List

import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor

from marble.encoders.MERT.MusicHubert import MusicHubertModel
from marble.core.base_encoder import BaseEncoder
from marble.core.base_transform import BaseAudioTransform


class MERT_v1_95M_Encoder(BaseEncoder):
    """
    A Hugging Face HuBERT-based wrapper with optional LoRA adapters, full fine-tuning, or freezing.
    """

    NAME = "MERT-v1-95M"
    HUGGINGFACE_MODEL_NAME = "m-a-p/MERT-v1-95M"
    TOKEN_RATE = 75  # Number of feature frames per second of audio
    SAMPLING_RATE = 24000  # Audio sampling rate expected by the model
    NUM_FEATURES = 768  # Hidden dimension of the HuBERT model
    N_TRANSFORMER_LAYERS = 12  # Number of transformer layers in the backbone
    PROCESSOR_NORMALIZE = True  # Whether to normalize audio in the feature extractor

    def __init__(
        self,
        pre_trained_folder: str = None,
        train_mode: str = "freeze",  # one of ["freeze", "full", "lora"]
        force_half: bool = False,
        preprocess_in_forward: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Sequence[str] = ["q_proj", "v_proj"],
    ) -> None:
        """
        Initialize the MERT HuBERT encoder.

        Args:
            pre_trained_folder (str, optional): Path or HF identifier of the pretrained model.
            train_mode (str): "freeze" to freeze base parameters, "full" for full fine-tuning,
                              or "lora" to freeze base and add LoRA adapters.
            force_half (bool): If True, cast model weights to float16.
            preprocess_in_forward (bool): If True, run feature extraction inside forward().
            lora_r (int): LoRA adapter rank (only if train_mode="lora").
            lora_alpha (int): LoRA scaling alpha (only if train_mode="lora").
            lora_dropout (float): Dropout probability for LoRA adapters.
        """
        super().__init__()
        self.sample_rate = self.SAMPLING_RATE
        self.preprocess_in_forward = preprocess_in_forward

        # Load the Wav2Vec2 feature extractor (normalizes and pads audio)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            pre_trained_folder or self.HUGGINGFACE_MODEL_NAME,
            do_normalize=self.PROCESSOR_NORMALIZE,
        )
        # Load the core MusicHuBERT model
        self.model = MusicHubertModel.from_pretrained(
            pre_trained_folder or self.HUGGINGFACE_MODEL_NAME
        )

        # Optionally cast model weights to half precision for memory savings
        if force_half:
            self.model = self.model.half()

        # Configure which parameters to train
        if train_mode == "freeze":
            # Freeze all backbone parameters
            for param in self.model.parameters():
                param.requires_grad = False

        elif train_mode == "lora":
            # Freeze backbone and add LoRA adapters
            from peft import get_peft_model, LoraConfig, TaskType

            for param in self.model.parameters():
                param.requires_grad = False

            peft_config = LoraConfig(
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
            self.model = get_peft_model(self.model, peft_config)

        elif train_mode == "full":
            # Enable training of all parameters
            for param in self.model.parameters():
                param.requires_grad = True

        else:
            raise ValueError(f"Unknown train_mode: {train_mode}")

        # Set model to train or eval mode
        if train_mode in ["lora", "full"]:
            self.model.train()
        else:
            self.model.eval()

    def forward(
        self,
        x: torch.Tensor,
        *args,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
        **kwargs
    ) -> dict:
        """
        Perform a forward pass through the HuBERT encoder.

        Args:
            x (torch.Tensor): Waveform tensor, shape (batch_size, num_samples), values in [-1, 1].
            output_hidden_states (bool): If True, return all intermediate hidden states.
            output_attentions (bool): If True, return attention weight matrices.
            *args, **kwargs: Additional arguments passed to the underlying model.

        Returns:
            A BaseModelOutput object containing:
                - last_hidden_state (torch.FloatTensor): Final-layer representations,
                  shape (batch_size, seq_len, NUM_FEATURES).
                - hidden_states (tuple of torch.FloatTensor, optional): All layer outputs
                  if output_hidden_states=True; each is (batch_size, seq_len, NUM_FEATURES).
                - attentions (tuple of torch.FloatTensor, optional): Attention maps
                  if output_attentions=True; each is (batch_size, num_heads, seq_len, seq_len).
        """
        # we generally do not recommend doing preprocessing in the forward pass
        # make sure batch size == 1 if preprocess_in_forward is True
        if self.preprocess_in_forward:
            assert isinstance(x, torch.Tensor)
            assert x.ndim == 1, "Input must be a 1D tensor (batch_size=1)"
            x = x.detach().cpu()
            proc = self.feature_extractor(
                x,
                sampling_rate=self.sample_rate, # we assume the input is already sampled at self.sample_rate
                return_tensors="pt",
                padding=True,
            )
            input_values = proc.input_values.to(self.model.device)
        else:
            # Assume `x` is already preprocessed values
            input_values = x.to(self.model.device)

        # Ensure input dtype matches model parameters (fp16 vs fp32)
        model_dtype = next(self.model.parameters()).dtype
        input_values = input_values.to(device=self.model.device, dtype=model_dtype)
        
        outputs = self.model(
            input_values=input_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        return outputs


class MERT_v1_95M_FeatureExtractor(BaseAudioTransform):
    """特征提取器"""
    NAME = "MERT-v1-95M"
    HUGGINGFACE_MODEL_NAME = "m-a-p/MERT-v1-95M"
    TOKEN_RATE = 75
    SAMPLING_RATE = 24000
    NUM_FEATURES = 768
    N_TRANSFORMER_LAYERS = 12
    PROCESSOR_NORMALIZE = True
    def __init__(self, pre_trained_folder: str = None, squeeze: bool = True) -> None:
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.HUGGINGFACE_MODEL_NAME if pre_trained_folder is None else pre_trained_folder,
            do_normalize=self.PROCESSOR_NORMALIZE,
        )
        self.squeeze = squeeze  # If True, squeeze the output to remove extra dimensions

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = sample["waveform"]
        assert isinstance(x, torch.Tensor)
        assert x.ndim == 1 or (x.ndim == 2 and x.shape[0] == 1) , "Input must be a 1D tensor (batch_size=1)"
        # while it also supports list of ndarray, we disable it for now
        # this class should be used in the dataloader
        proc = self.feature_extractor(
            x,
            sampling_rate=sample['sampling_rate'],
            return_attention_mask=False,
            return_tensors="pt",
            padding=True,
        )
        sample["waveform"] = proc.input_values # [batch_size, num_samples] but batch_size=1
        if self.squeeze:
            # note that MERT does not have a channel dimension since it is mono
            # better always squeeze the output
            sample["waveform"] = sample["waveform"].squeeze()
        assert sample["waveform"].ndim == 1, "Output waveform should be squeezed to 1D tensor"
        return sample


class MERT_v1_330M_Encoder(MERT_v1_95M_Encoder):
    """A Hugging Face HuBERT-based wrapper with optional LoRA, full-tuning or freezing."""
    NAME = "MERT-v1-330M"
    HUGGINGFACE_MODEL_NAME = "m-a-p/MERT-v1-330M"
    TOKEN_RATE = 75
    SAMPLING_RATE = 24000
    NUM_FEATURES = 1024
    N_TRANSFORMER_LAYERS = 24
    PROCESSOR_NORMALIZE = True


class MERT_v1_330M_FeatureExtractor(MERT_v1_95M_FeatureExtractor):
    """特征提取器"""
    HUGGINGFACE_MODEL_NAME = "m-a-p/MERT-v1-330M"
    SAMPLING_RATE = 24000
    PROCESSOR_NORMALIZE = True


if __name__ == "__main__":
    # 测试代码
    model = MERT_v1_330M_Encoder(preprocess_in_forward=True)
    x = torch.randn(24000 * 5)  # 1个5秒的音频
    out = model(x)
    print(out.last_hidden_state.shape)  # 应该是 (1, 24000 * 5 / 75, 1024)
    
    # 测试特征提取器
    feature_extractor = MERT_v1_330M_FeatureExtractor()
    x = torch.randn(24000 * 5)  # 1个5秒的音频
    features = feature_extractor({"waveform": x, "sampling_rate": 24000})
    print(features['waveform'].shape)
    
    
