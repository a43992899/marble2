# ssl_bench/cli.py
from lightning.pytorch.cli import LightningCLI
from marble.core import registry   # 确保 import 触发所有 @register


class BenchmarkCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # 统一公共选项
        parser.set_defaults(seed_everything=42)
        parser.add_argument("--pipeline", type=str, default="probing",
                            help="probing | finetune | fewshot")

    def instantiate_classes(self):
        # LightningCLI 里重写，先解析 config → str，再找 registry
        cfg = self.config
        enc_cls = registry.get("encoder", cfg.model.encoder)
        dec_cls = registry.get("decoder", cfg.model.decoder)
        task_cls = registry.get("task", cfg.model.task)
        pipe_cls = registry.get("pipeline", self.config.pipeline)

        cfg.model.encoder = enc_cls
        cfg.model.decoder = dec_cls
        cfg.model.task    = task_cls
        cfg.pipeline = pipe_cls
        return super().instantiate_classes()

def main():
    BenchmarkCLI(
        save_config_callback=None,  # 所有超参写回 log dir
        save_config_kwargs={"overwrite": True},
        ) 
if __name__ == "__main__":
    main()
