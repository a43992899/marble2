# tests/test_qwen2audio_encoder.py
import torch
import librosa

from marble.encoders.Qwen2AudioInstructEncoder.processing_qwen2_audio import Qwen2AudioProcessor
from marble.encoders.Qwen2AudioInstructEncoder.modeling_qwen2_audio import (
    Qwen2AudioEncoder,
    Qwen2AudioEncoderConfig,
)

def main():
    # 1. 路径配置
    audio_path = "/aifs4su/mmcode/codeclm/marble2/data/GTZAN/genres/blues/blues.00000.wav"
    local_repo = "Qwen/Qwen2-Audio-7B-Instruct"

    # 2. 加载 Processor（仅为拿到 feature_extractor）
    processor = Qwen2AudioProcessor.from_pretrained(
        local_repo,
        trust_remote_code=True,
    )
    feature_extractor = processor.feature_extractor

    # 3. 加载 encoder 配置和权重
    config = Qwen2AudioEncoderConfig.from_pretrained(
        local_repo,
        trust_remote_code=True,
    )
    encoder = Qwen2AudioEncoder.from_pretrained(
        local_repo,
        config=config,
        trust_remote_code=True,
    ).eval()

    # 4. 读取并重采样音频
    audio, sr = librosa.load(audio_path, sr=feature_extractor.sampling_rate)

    # 5. 提取 log-mel 特征 & 1D attention mask
    fe_out = feature_extractor(
        audio,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding="max_length",
    )
    # fe_out["input_features"]: (batch, n_mels, mel_seq_len)
    # fe_out["attention_mask"]:   (batch, mel_seq_len)
    input_features = fe_out["input_features"]
    feature_mask_1d = fe_out["attention_mask"]

    # 6. 转到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    input_features = input_features.to(device)

    # 8. 前向计算，提取 embedding
    with torch.no_grad():
        outputs = encoder(
            input_features=input_features,
            attention_mask=None,
            return_dict=True,
        )
    embeddings = outputs.last_hidden_state  # (batch, max_seq_len, hidden_dim)

    print("Embedding shape:", embeddings.shape)

if __name__ == "__main__":
    main()
