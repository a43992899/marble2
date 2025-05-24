```bash
.
|── configs
|   |── probe.MERT-v1-95M.GTZANGenre.yaml
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
|   |   |── ema.py
|   |   |── decoders.py
|   |   |── __init__.py
|   |   |── loss.py
|   |── core
|   |   |── base_task.py
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
|   |   |   |── datamodule.py
|   |── utils
|   |   |── __init__.py
|   |   |── io_utils.py
|── pyproject.toml
|── .gitignore
|── README.md
|── scripts
|   |── print_filetree.sh
|   |── utils.py
|── data
|── cli.py
```

```bash
# 1. 建议先创建并激活一个新环境
conda create -n marble python=3.10 -y
conda activate marble

# 2. 安装 ffmpeg
conda install -c conda-forge ffmpeg -y

# 3. 降级 pip 到 24.0（Fairseq 要求）
pip install pip==24.0

# 4. 用 pip 安装项目依赖
pip install -e .
```

