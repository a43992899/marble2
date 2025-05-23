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
|   |── utils
|   |   |── __init__.py
|   |   |── io_utils.py
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

