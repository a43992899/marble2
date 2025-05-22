import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor

from MusicHubert import MusicHubertModel
from benchmark.core.registry import register
from benchmark.core.base_encoder import BaseEncoder
from benchmark.core.base_transform import BaseTransform

@register("encoder", "MERT-v1-95M")
class MERT_v1_95M_Encoder(BaseEncoder):
    """A Hugging Face HuBERT-based wrapper with optional LoRA, full-tuning or freezing."""
    NAME = "MERT-v1-95M"
    HUGGINGFACE_MODEL_NAME = "m-a-p/MERT-v1-95M"
    TOKEN_RATE = 75
    SAMPLING_RATE = 24000
    NUM_FEATURES = 768
    N_TRANSFORMER_LAYERS = 12
    PROCESSOR_NORMALIZE = True

    def __init__(
        self,
        pre_trained_folder: str = None,
        sample_rate: int = SAMPLING_RATE,
        train_mode: str = "freeze",  # one of ["freeze", "full", "lora"]
        force_half: bool = False,
        preprocess_in_forward: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.preprocess_in_forward = preprocess_in_forward

        # 加载特征提取器
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.HUGGINGFACE_MODEL_NAME if pre_trained_folder is None else pre_trained_folder,
            do_normalize=self.PROCESSOR_NORMALIZE,
        )
        # 加载 HuBERT 模型
        self.model = MusicHubertModel.from_pretrained(
            self.HUGGINGFACE_MODEL_NAME if pre_trained_folder is None else pre_trained_folder,
        )

        # 精度转换
        if force_half:
            self.model = self.model.half()

        # 参数策略
        if train_mode == "freeze":
            # 冻结全部权重
            for param in self.model.parameters():
                param.requires_grad = False
        elif train_mode == "lora":
            # 冻结基础模型参数
            for param in self.model.parameters():
                param.requires_grad = False
            # 配置 LoRA
            from peft import get_peft_model, LoraConfig, TaskType
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            self.model = get_peft_model(self.model, peft_config)
        elif train_mode == "full":
            # 解冻所有参数
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown train_mode: {train_mode}")

        # 训练或评估模式
        self.model.train() if train_mode in ["lora", "full"] else self.model.eval()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        前向计算：
        输入:
            x: FloatTensor, shape (batch_size, num_samples)，值范围[-1, 1]
        输出:
            features: FloatTensor, shape (batch_size, seq_len, hidden_dim)
        """
        if self.preprocess_in_forward:
            # waveform 处理
            wav_cpu = x.detach().cpu() if x.device != torch.device("cpu") else x
            proc = self.feature_extractor(
                wav_cpu,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            input_values = proc.input_values.to(self.model.device)
        else:
            # 直接使用输入
            input_values = x.to(self.model.device)

        outputs = self.model(input_values)
        return outputs.last_hidden_state


@register("transforms", "MERT-v1-95M_feature_extractor")
class MERT_v1_95M_FeatureExtractor(BaseTransform):
    """特征提取器"""
    HUGGINGFACE_MODEL_NAME = "m-a-p/MERT-v1-95M"
    SAMPLING_RATE = 24000
    PROCESSOR_NORMALIZE = True
    def __init__(self, pre_trained_folder: str = None):
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.HUGGINGFACE_MODEL_NAME if pre_trained_folder is None else pre_trained_folder,
            do_normalize=self.PROCESSOR_NORMALIZE,
        )

    def __call__(self, x: torch.Tensor) -> dict:
        proc = self.feature_extractor(
            x,
            sampling_rate=MERT_v1_95M_Encoder.SAMPLING_RATE,
            return_tensors="pt",
            padding=True,
        )
        return proc


@register("encoder", "MERT-v1-330M")
class MERT_v1_330M_Encoder(BaseEncoder):
    """A Hugging Face HuBERT-based wrapper with optional LoRA, full-tuning or freezing."""
    NAME = "MERT-v1-330M"
    HUGGINGFACE_MODEL_NAME = "m-a-p/MERT-v1-330M"
    TOKEN_RATE = 75
    SAMPLING_RATE = 24000
    NUM_FEATURES = 1024
    N_TRANSFORMER_LAYERS = 24
    PROCESSOR_NORMALIZE = False

    def __init__(
        self,
        pre_trained_folder: str = None,
        sample_rate: int = SAMPLING_RATE,
        train_mode: str = "freeze",  # one of ["freeze", "full", "lora"]
        force_half: bool = False,
        preprocess_in_forward: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.preprocess_in_forward = preprocess_in_forward

        # 加载特征提取器
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.HUGGINGFACE_MODEL_NAME if pre_trained_folder is None else pre_trained_folder,
            do_normalize=self.PROCESSOR_NORMALIZE,
        )
        # 加载 HuBERT 模型
        self.model = MusicHubertModel.from_pretrained(
            self.HUGGINGFACE_MODEL_NAME if pre_trained_folder is None else pre_trained_folder,
        )

        # 精度转换
        if force_half:
            self.model = self.model.half()

        # 参数策略
        if train_mode == "freeze":
            # 冻结全部权重
            for param in self.model.parameters():
                param.requires_grad = False
        elif train_mode == "lora":
            # 冻结基础模型参数
            for param in self.model.parameters():
                param.requires_grad = False
            # 配置 LoRA
            from peft import get_peft_model, LoraConfig, TaskType
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            self.model = get_peft_model(self.model, peft_config)
        elif train_mode == "full":
            # 解冻所有参数
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown train_mode: {train_mode}")

        # 训练或评估模式
        self.model.train() if train_mode in ["lora", "full"] else self.model.eval()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        前向计算：
        输入:
            x: FloatTensor, shape (batch_size, num_samples)，值范围[-1, 1]
        输出:
            features: FloatTensor, shape (batch_size, seq_len, hidden_dim)
        """
        if self.preprocess_in_forward:
            # waveform 处理
            wav_cpu = x.detach().cpu() if x.device != torch.device("cpu") else x
            proc = self.feature_extractor(
                wav_cpu,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            input_values = proc.input_values.to(self.model.device)
        else:
            # 直接使用输入
            input_values = x.to(self.model.device)

        outputs = self.model(input_values)
        return outputs.last_hidden_state


@register("transforms", "MERT-v1-330M_feature_extractor")
class MERT_v1_330M_FeatureExtractor(BaseTransform):
    """特征提取器"""
    HUGGINGFACE_MODEL_NAME = "m-a-p/MERT-v1-330M"
    SAMPLING_RATE = 24000
    PROCESSOR_NORMALIZE = False
    def __init__(self, pre_trained_folder: str = None):
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.HUGGINGFACE_MODEL_NAME if pre_trained_folder is None else pre_trained_folder,
            do_normalize=self.PROCESSOR_NORMALIZE,
        )

    def __call__(self, x: torch.Tensor) -> dict:
        proc = self.feature_extractor(
            x,
            sampling_rate=MERT_v1_95M_Encoder.SAMPLING_RATE,
            return_tensors="pt",
            padding=True,
        )
        return proc


if __name__ == "__main__":
    # 测试代码
    model = MERT_v1_95M_Encoder()
    x = torch.randn(2, 24000 * 5)  # 2个5秒的音频
    features = model(x)
    print(features.shape)  # 应该是 (2, seq_len, 768)
    
    # 测试特征提取器
    feature_extractor = MERT_v1_95M_FeatureExtractor()
    x = torch.randn(2, 24000 * 5)  # 2个5秒的音频
    features = feature_extractor(x)
    print(features)  # 应该是一个字典，包含 input_values 和 attention_mask
    
    
