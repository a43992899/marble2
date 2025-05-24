# marble/tasks/GTZANGenre/probe.py
import torch
import torch.nn as nn
from torchmetrics import MetricCollection

from marble.core.base_task import BaseTask  
from marble.core.utils import instantiate_from_config
from marble.modules.ema import LitEma as EMA


class ProbeAudioTask(BaseTask):
    """
    A probing task for GTZAN genre classification using MetricCollection.
    """

    def __init__(
        self,
        sample_rate: int,
        use_ema: bool,
        encoder: dict,
        emb_transforms: list[dict],
        decoders: list[dict],
        losses: list[dict],
        metrics: dict,
    ):
        super().__init__()
        # automatically log these args
        self.save_hyperparameters("sample_rate", "use_ema")

        self.sample_rate = sample_rate
        self.use_ema = use_ema

        # instantiate sub‐modules from configs
        self.encoder = instantiate_from_config(encoder)
        self.emb_transforms = nn.Sequential(
            *[instantiate_from_config(cfg) for cfg in emb_transforms]
        )
        self.decoders = nn.ModuleList([
            instantiate_from_config(cfg) for cfg in decoders
        ])
        self.loss_fns = [
            instantiate_from_config(cfg) for cfg in losses
        ]

        # build MetricCollections for each split
        train_mc = MetricCollection(
            {name: instantiate_from_config(cfg) for name, cfg in metrics["train"].items()},
            prefix="train/"
        )
        val_mc = MetricCollection(
            {name: instantiate_from_config(cfg) for name, cfg in metrics["val"].items()},
            prefix="val/"
        )
        test_mc = MetricCollection(
            {name: instantiate_from_config(cfg) for name, cfg in metrics["test"].items()},
            prefix="test/"
        )

        # register them as modules so Lightning knows about their parameters/buffers
        self.add_module("train_metrics", train_mc)
        self.add_module("val_metrics",   val_mc)
        self.add_module("test_metrics",  test_mc)

        self.train_metrics = train_mc
        self.val_metrics = val_mc
        self.test_metrics = test_mc

        # optional EMA on encoder weights
        if self.use_ema:
            self.ema = EMA(self.encoder)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        # waveforms: (batch, channels, time)
        emb = self.encoder(waveforms)
        emb = self.emb_transforms(emb)

        # pass through decoders
        logits = [dec(emb) for dec in self.decoders]
        return logits[0] if len(logits) == 1 else logits

    def _shared_step(self, batch, batch_idx, stage: str):
        waveforms, labels, file_paths = batch
        logits = self(waveforms)

        # compute total loss
        losses = [fn(logits, labels) for fn in self.loss_fns]
        loss = sum(losses)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # update + log metrics
        mc: MetricCollection = getattr(self, f"{stage}_metrics")
        metrics_out = mc(logits, labels)
        self.log_dict(metrics_out, prog_bar=(stage=="val"), on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        """
        batch = (waveforms, labels, file_paths)
        we return per‐slice preds + labels + file_ids
        """
        waveforms, labels, file_paths = batch
        logits = self(waveforms)  # (B, C)
        probs = torch.softmax(logits, dim=1).detach().cpu()
        preds = torch.argmax(probs, dim=1)

        return {
            "file_paths": file_paths,
            "probs": probs,
            "preds": preds,
            "labels": labels.detach().cpu(),
        }

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if self.use_ema:
            self.ema.update()

    def test_epoch_end(self, outputs):
        """
        聚合所有 slice 级别的输出，按文件计算平均概率，
        然后得到 file‐level 的预测，并计算准确率。
        """
        # 1) 收集到一个 dict[file_path] -> list of probs + label
        file_dict = {}
        for out in outputs:
            for fp, prob, pred, lb in zip(out["file_paths"],
                                          out["probs"],
                                          out["preds"],
                                          out["labels"]):
                if fp not in file_dict:
                    file_dict[fp] = {"probs": [], "label": int(lb)}
                file_dict[fp]["probs"].append(prob.numpy())

        # 2) 聚合：对每个文件，stack & mean
        file_preds = {}
        correct = 0
        total = 0
        for fp, info in file_dict.items():
            arr = torch.tensor(info["probs"])           # (num_slices, C)
            mean_prob = arr.mean(dim=0)                # (C,)
            file_pred = int(mean_prob.argmax().item())
            file_preds[fp] = file_pred

            total += 1
            if file_pred == info["label"]:
                correct += 1

        file_acc = correct / total
        # 3) log file-level acc
        self.log("test/file_acc", file_acc, prog_bar=True, on_epoch=True)

        # 4) 可选：将 file_preds 暴露到 Trainer.callback_metrics 或保存到成员变量
        # self.file_level_preds = file_preds
        return {"file_level_preds": file_preds, "file_acc": file_acc}
    
    def configure_optimizers(self):
        # delegate to BaseTask / LightningCLI
        return super().configure_optimizers()

