# marble/modules/callbacks.py
import os
import glob
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

class LoadLatestCheckpointCallback(Callback):
    """
    在 test 开始时，自动从 ModelCheckpoint 的 dirpath 目录里
    找到最新的 .ckpt 文件并 load 到 pl_module 中。
    """
    def on_test_start(self, trainer, pl_module):
        # 1) 从 trainer.callbacks 中找到你的 ModelCheckpoint 实例
        ckpt_cb: ModelCheckpoint | None = next(
            (cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)),
            None
        )
        if ckpt_cb is None:
            raise RuntimeError("没找到 ModelCheckpoint 回调，无法定位 ckpt 目录。")

        ckpt_dir = ckpt_cb.dirpath
        if not os.path.isdir(ckpt_dir):
            raise RuntimeError(f"Checkpoint 目录不存在：{ckpt_dir}")

        # 2) 列出所有 .ckpt，按文件修改时间选最新的那一个
        paths = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        if not paths:
            raise RuntimeError(f"{ckpt_dir} 里没有任何 .ckpt 文件。")
        latest_ckpt = max(paths, key=os.path.getmtime)

        # 3) load 到模型上
        #    map_location 选 pl_module 当前所在设备
        map_loc = {"cpu": "cpu"}
        if pl_module.device.type == "cuda":
            map_loc = {"cuda:0": f"cuda:{pl_module.device.index or 0}"}
        checkpoint = torch.load(latest_ckpt, map_location=map_loc)
        state_dict = checkpoint.get("state_dict", checkpoint)
        pl_module.load_state_dict(state_dict)

        # 4) 日志告知
        trainer.logger.log_metrics({"loaded_ckpt": os.path.basename(latest_ckpt)})
        print(f"[LoadLatestCheckpoint] loaded {latest_ckpt}")