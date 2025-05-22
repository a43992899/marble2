```bash
.
|── configs
|   |── probe.MERT-v1-95M.GTZANGenre.yaml
|── requirements.txt
|── marble
|   |── encoders
|   |   |── Qwen2AudioEncoder
|   |   |── MuQ
|   |   |── JukeBox
|   |   |── __init__.py
|   |   |── MusicFM
|   |   |── MERT
|   |   |   |── model.py
|   |   |   |── configuration_musichubert.py
|   |   |   |── MusicHubert.py
|   |   |   |── __init__.py
|   |   |── identity
|   |   |   |── model.py
|   |   |── YuE
|   |   |── W2V2BERT
|   |── modules
|   |   |── transforms.py
|   |   |── decoders.py
|   |   |── poolings.py
|   |   |── __init__.py
|   |── core
|   |   |── registry.py
|   |   |── base_task.py
|   |   |── __pycache__
|   |   |   |── base_encoder.cpython-310.pyc
|   |   |   |── __init__.cpython-310.pyc
|   |   |   |── registry.cpython-310.pyc
|   |   |   |── base_transform.cpython-310.pyc
|   |   |── utils.py
|   |   |── base_transform.py
|   |   |── __init__.py
|   |   |── base_decoder.py
|   |   |── base_encoder.py
|   |── tasks
|   |   |── GTZANGenre
|   |   |   |── metrics.py
|   |   |   |── download.py
|   |   |   |── postprocess.py
|   |   |   |── probe.py
|   |   |   |── preprocess.py
|   |   |   |── fewshot.py
|   |   |   |── decoder.py
|   |   |   |── datamodule.py
|── pyproject.toml
|── .gitignore
|── README.md
|── scripts
|   |── print_filetree.sh
|   |── utils.py
|── cli.py

```

```bash
pip install -e .
```


```python
@dataclass(frozen=True)
class InputSpec:
    fmt: Literal["wave", "spec"]     # 输入类型
    sr: int                          # 期望采样率 (Hz)
    n_fft: int | None = None         # ↓ 只有 fmt="spec" 时需要
    hop_length: int | None = None
    n_mels: int | None = None


@register("encoder", "pslaudio48_mel")
class PSLAudio48Mel(BaseEncoder):
    input_spec = InputSpec(fmt="spec", sr=48_000, n_mels=128, n_fft=2048, hop_length=512)

    def __init__(self, **kw):
        super().__init__(emb_transform=kw.get("emb_transform"))
        self.raw_dim = 512
        self.backbone = SomeConvNet(**kw)

    def forward(self, mel, mel_len=None):
        return self.backbone(mel)

class GenericAudioDM(BaseDataModule):
    def __init__(self, encoder_name, batch_size, **kw):
        super().__init__()
        self.enc_cls = registry.get("encoder", encoder_name)
        self.spec = self.enc_cls.input_spec
        self.build_transforms()

    def build_transforms(self):
        tf = []
        if self.spec.sr:
            tf.append(audio_tf.Resample(self.spec.sr))
        if self.spec.fmt == "spec":
            tf.append(audio_tf.WaveToMel(sr=self.spec.sr,
                                         n_fft=self.spec.n_fft,
                                         hop_length=self.spec.hop_length,
                                         n_mels=self.spec.n_mels))
        self.pipeline = torch.nn.Sequential(*tf)

    def __getitem__(self, idx):
        wav, sr, label = self._load_file(idx)
        wav = self.pipeline(wav)          # 已转成 mel 或重采样波形
        return {"audio": wav, "label": label}

# yaml
# model:
#   encoder: pslaudio48_mel        # 只需指名 encoder，框架推断 input_spec
# data:
#   class_path: tasks.generic_dm.GenericAudioDM
#   init_args:
#     encoder_name: ${model.encoder}
#     batch_size: 8

```