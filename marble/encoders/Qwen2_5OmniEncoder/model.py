import torch
from typing import Dict, Optional

from marble.core.base_encoder import BaseEncoder
from marble.core.base_transform import BaseAudioTransform
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration


class Qwen2_5OmniEncoder(BaseEncoder):
    """
    A wrapper around the Qwen2.5-Omni encoder's audio tower with optional freezing or full fine-tuning.
    """
    NAME = "Qwen2_5OmniEncoder"
    HUGGINGFACE_MODEL_NAME = "Qwen/Qwen2.5-Omni-7B"
    N_TRANSFORMER_LAYERS = 32
    SAMPLING_RATE = 16000
    TOKEN_RATE = 25
    NUM_FEATURES = 1280

    def __init__(
        self,
        pre_trained_folder: str = None,
        train_mode: str = "freeze",  # one of ["freeze", "full"]
        attn_implementation: str = "flash_attention_2"
    ) -> None:
        super().__init__()
        repo = pre_trained_folder or self.HUGGINGFACE_MODEL_NAME

        # Load processor and feature extractor
        self.processor = Qwen2_5OmniProcessor.from_pretrained(repo)
        self.feature_extractor = self.processor.feature_extractor
        self.sample_rate = self.feature_extractor.sampling_rate

        # Load full Omni model and extract audio tower
        print(f"Loading Qwen2.5-Omni audio tower from {repo}")
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            repo,
            attn_implementation=attn_implementation,
        ).thinker.audio_tower

        # Configure training mode
        if train_mode == "freeze":
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        elif train_mode == "full":
            for param in self.model.parameters():
                param.requires_grad = True
            self.model.train()

        else:
            raise ValueError(f"Unknown train_mode: {train_mode}")

    def forward(
        self,
        input_features: torch.FloatTensor,
        **kwargs,
    ) -> dict:
        """
        Forward pass through the Omni audio tower.

        Args:
            input_features (torch.FloatTensor): Log-mel or raw audio features, shape (batch, num_features, seq_len)
            attention_mask (torch.BoolTensor, optional): Feature mask, shape (batch, seq_len)
        Returns:
            Model outputs (e.g., last_hidden_state, hidden_states)
        """
        device = next(self.model.parameters()).device
        input_features = input_features.to(device)

        outputs = self.model(
            input_features=input_features,
            # output_hidden_states=output_hidden_states,
            # return_dict=True,
            **kwargs,
        )
        return outputs


class Qwen2_5OmniFeatureExtractor(BaseAudioTransform):
    """
    Audio-to-feature transform using Qwen2.5-Omni processor.
    """
    NAME = "Qwen2_5OmniFeatureExtractor"
    HUGGINGFACE_MODEL_NAME = "Qwen/Qwen2.5-Omni-7B"
    N_TRANSFORMER_LAYERS = 32
    SAMPLING_RATE = 24000
    TOKEN_RATE = 25
    NUM_FEATURES = 1280

    def __init__(
        self,
        pre_trained_folder: str = None,
        squeeze: bool = True,
    ) -> None:
        super().__init__()
        repo = pre_trained_folder or self.HUGGINGFACE_MODEL_NAME
        self.processor = Qwen2_5OmniProcessor.from_pretrained(repo)
        self.feature_extractor = self.processor.feature_extractor
        self.squeeze = squeeze

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract audio features from a raw waveform tensor.

        Args:
            sample["input_features"]: 1D torch.Tensor or List[torch.Tensor]
            sample["sampling_rate"]: int (optional)

        Returns:
            sample with key "input_features": torch.FloatTensor of shape (num_features, seq_len)
        """
        # waveform: ensure 1D
        x = sample["input_features"].squeeze()
        assert isinstance(x, torch.Tensor) and x.ndim == 1, \
            f"Input must be 1D waveform Tensor, got shape {x.shape}"

        sr = sample.get("sampling_rate", self.feature_extractor.sampling_rate)
        feats = self.feature_extractor(
            x,
            sampling_rate=sr,
            return_tensors="pt",
        )
        features = feats["input_features"]  # shape (1, num_features, seq_len)
        if self.squeeze:
            features = features.squeeze(0)
        assert features.ndim == 2, \
            f"Extracted features must be 2D [num_features, seq_len], got {features.shape}"

        sample["input_features"] = features
        return sample


if __name__ == "__main__":
    # 简单测试
    import librosa

    repo = "Qwen/Qwen2.5-Omni-7B"
    encoder = Qwen2_5OmniEncoder(pre_trained_folder=repo)
    fe = Qwen2_5OmniFeatureExtractor(pre_trained_folder=repo)

    # 1. 读取音频
    audio_path = "./tests/output.wav"
    wav, _ = librosa.load(audio_path, sr=encoder.sample_rate)

    # 2. 特征提取
    sample = {"input_features": torch.tensor(wav), "sampling_rate": encoder.sample_rate}
    sample = fe(sample)
    inp = sample["input_features"].unsqueeze(0)

    # 3. 前向
    out = encoder(input_features=inp)
    print("Last hidden state shape:", out.last_hidden_state.shape)
    print("Hidden layers:", len(out.hidden_states))