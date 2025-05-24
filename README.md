# MARBLE (v2): Music Audio Representation Benchmark for Universal Evaluation

Marble is a modular, configuration-driven benchmark suite for evaluating self-supervised music audio representations on downstream tasks. It leverages LightningCLI, easy extensibility, easy reproduction.


## Key Features
1. **Modularity**: Each component—encoders, tasks, transforms, decoders—is isolated behind a common interface. You can mix and match without touching core logic.
2. **Configurability**: All experiments are driven by YAML configs. No code changes are needed to switch datasets, encoders, or training settings.
3. **Reusability**: Common routines (data loading, training loop, metrics) are implemented once in `BaseTask`, `LightningDataModule`, and shared modules.
4. **Extensibility**: Adding new encoders or tasks requires implementing a small subclass and registering it via a config.

```text
┌──────────────────┐
│ DataModule       │  yields (waveform, label, path), optional audio transforms
└─▲────────────────┘
  │
  │ waveform                     Encoded →   hidden_states[B, L, T, H]
  ▼
┌─┴───────────────┐   embedding transforms (optional)
│ Encoder         │ ────────────────────────────────────────────────────┐
└─────────────────┘                                                     │
                                                                        ▼
                                                         (LayerSelector, TimeAvgPool…)
                                                                        │
                                                                        ▼
                                      ┌─────────────────────────────────┴──┐
                                      │ Decoder(s)                         │
                                      └────────────────────────────────────┘
                                                                  │ logits
                                                                  ▼
                                                   Loss ↔ Metrics ↔ Callbacks
```


## Getting Started

1. **Install dependencies**:
    ```bash
    # 1. create a new conda env
    conda create -n marble python=3.10 -y
    conda activate marble

    # 2. install ffmpeg
    conda install -c conda-forge ffmpeg -y

    # 3. downgrade pip to 24.0（Since we have some Fairseq models）
    pip install pip==24.0

    # 4. now install other dependencies
    pip install -e .
    ```

2. **Prepare data**: Place your datasets and JSONL metadata under `data/`.

3. **Configure**: Copy an existing YAML from `configs/` and edit paths, encoder settings, transforms, and task parameters.

4. **Run**:

   ```bash
   python cli.py fit --config configs/probe.MERT-v1-95M.GTZANGenre.yaml
   python cli.py test --config configs/probe.MERT-v1-95M.GTZANGenre.yaml
   ```

5. **Results**: Checkpoints and logs will be saved under `output/` and logged in Weights & Biases.







## Project Structure
```bash
.
├── marble/                   # Core code package
│   ├── core/                 # Base classes (BaseTask, BaseEncoder, BaseTransform)
│   ├── encoders/             # Wrapper classes for various SSL encoders
│   ├── modules/              # Shared transforms, callbacks, losses, decoders
│   ├── tasks/                # Downstream tasks (probe, few-shot, datamodules)
│   └── utils/                # IO utilities, instantiation helpers
├── cli.py                    # Entry-point for launching experiments
├── configs/                  # Experiment configs (YAML)
├── data/                     # Datasets and metadata files
├── scripts/                  # Run scripts & utilities
├── tests/                    # Unit tests for transforms & datasets
├── pyproject.toml            # Python project metadata
└── README.md                 # This file
```



## Adding a New Encoder

1. **Implement Encoder**:

   * Create a subclass of `marble.core.base_encoder.BaseAudioEncoder`:

     ```python
     class MyEncoder(BaseAudioEncoder):
         def __init__(self, ...):
             super().__init__()
             # load or define your model
         def forward(self, waveforms):
             # return hidden states: List[Tensor] of shape (batch, layer, seq_len, hidden_size)
             # or return a dict of representations, then you should write your own feature selector
     ```
   * It would be better if you can place the file under `marble/encoders/my_encoder.py`, but it is OK to just put it in e.g. `./my_project/my_encoder.py` since we are config driven.

2. **Register in Config**:

   * In your YAML experiment:

     ```yaml
     model:
       encoder:
         class_path: marble.encoders.my_encoder.MyEncoder
         init_args:
           my_arg: value
     ```

3. **Feature Extraction Config (Optional)**:

   * Consider reusing embedding transforms in `marble/modules/transforms.py`. If your encoder requires custom preprocessing, you can implement it as a subclass of `marble.core.base_transform.BaseEmbTransform`, and put it into `marble/modules/transforms.py`or `./my_project/my_transforms.py`.
   * For audio transforms, similarly, you can implement it as a subclass of `marble.core.base_transform.BaseAudioTransform`, and put it into `marble/modules/transforms.py`or `./my_project/my_transforms.py`.
   * In your YAML experiment:
      ```yaml
      ...
      emb_transforms:
         - class_path: marble.modules.transforms.MyEmbTransform
            init_args:
               param1: value1
      ...
      audio_transforms:
      train:
        - class_path: marble.modules.transforms.MyAudioTransform
          init_args:
            param1: value1
      ```


## Adding a New Task

1. **Define DataModule**:

   * Subclass `pl.LightningDataModule` and implement `setup`, `train_dataloader`, `val_dataloader`, `test_dataloader`. Use `instantiate_from_config` for consistency. Most importantly, implemnt the dataset.
   * Place under `marble/tasks/YourTaskName/datamodule.py` or `./my_project/datamodule.py`.
   * You may refer to `marble/tasks/GTZANGenre/datamodule.py` for an example. 

2. **Implement Task Logic**:

   * Subclass `marble.core.base_task.BaseTask`:

     ```python
     class YourTask(BaseTask):
         def __init__(self, encoder, emb_transforms, decoders, losses, metrics, sample_rate, use_ema):
             super().__init__(...)
             # any custom behavior
     ```
   * Override `training_step`, `validation_step`, or hooks if needed.
   * Place under `marble/tasks/YourTaskName/probe.py` or `./my_project/probe.py`. Or you can called it other names.
   * You may customize your decoders, losses, and metrics.

3. **Create Task Package**:

   * Organize under `marble/tasks/YourTaskName/`:

     * `__init__.py`
     * `datamodule.py`
     * `task.py` (your BaseTask subclass)
     * `probe.py` / `finetune.py` / `fewshot.py` for different evaluation protocols

4. **Configure in YAML**:

   * Add under `model.class_path`: `marble.tasks.YourTaskName.probe.YourTask`
   * Define `data` section pointing to your `DataModule`.

