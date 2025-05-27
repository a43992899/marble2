# marble/encoders/MuQMuLan/model.py
from typing import Sequence, Dict, Optional, Union, Tuple, List

import torch

from marble.encoders.MuQMuLan.muq_mulan import MuQMuLan
from marble.core.base_encoder import BaseEncoder


class MuQMuLan_Encoder(BaseEncoder):
    """
    A Hugging Face HuBERT-based wrapper with optional LoRA adapters, full fine-tuning, or freezing.
    """

    NAME = "MuQMuLan"
    HUGGINGFACE_MODEL_NAME = "OpenMuQ/MuQ-MuLan-large"
    TOKEN_RATE = 1  # Number of feature frames per second of audio
    SAMPLING_RATE = 24000  # Audio sampling rate expected by the model
    NUM_FEATURES = 512  # Hidden dimension of the HuBERT model

    def __init__(
        self,
        pre_trained_folder: str = None,
        train_mode: str = "freeze",  # one of ["freeze", "full", "lora"]
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
            lora_r (int): LoRA adapter rank (only if train_mode="lora").
            lora_alpha (int): LoRA scaling alpha (only if train_mode="lora").
            lora_dropout (float): Dropout probability for LoRA adapters.
        """
        super().__init__()
        self.sample_rate = self.SAMPLING_RATE

        # Load the core MusicHuBERT model
        self.model = MuQMuLan.from_pretrained(
            pre_trained_folder or self.HUGGINGFACE_MODEL_NAME
        )


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
        wavs: torch.Tensor,
        texts: Optional[Union[str, List[str]]] = None,
        *args,
        **kwargs
    ) -> dict:
        """
        Perform a forward pass through the HuBERT encoder.

        Args:
            wavs (torch.Tensor): Waveform tensor, shape (batch_size, num_samples), values in [-1, 1].
            texts (str or List[str], optional): Text input for the model, if applicable.
            *args, **kwargs: Additional arguments passed to the underlying model.

        Returns:
            A Tensor object (batch_size, seq_len, NUM_FEATURES).
        """
        # Ensure input dtype matches model parameters (fp16 vs fp32)
        model_dtype = next(self.model.parameters()).dtype
        wavs = wavs.to(device=self.model.device, dtype=model_dtype)
        
        outputs = self.model(wavs=wavs, texts=texts)
        
        # add seq_len dimension
        outputs = outputs.unsqueeze(1) 

        return outputs



if __name__ == "__main__":
    device = 'cuda'
    # fake wav for testing
    wav = torch.randn(2, 24000 * 10)  # 10 seconds of audio at 24kHz
    wavs = torch.tensor(wav).to(device) 

    # This will automatically fetch the checkpoint from huggingface
    muq = MuQMuLan_Encoder()
    muq = muq.to(device).eval()

    with torch.no_grad():
        output = muq(wavs)

    print(output.shape)
    
    
