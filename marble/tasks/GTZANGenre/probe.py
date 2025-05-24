# marble/tasks/GTZANGenre/probe.py
import torch
import torch.nn as nn
from marble.core.base_task import BaseTask
from marble.core.utils import instantiate_from_config

class ProbeAudioTask(BaseTask):
    """
    GTZAN genre probe task.  Inherits training/val logic, multi‐head,
    losses, metrics and EMA support from BaseTask.
    """

    def __init__(
        self,
        sample_rate: int,
        use_ema: bool,
        encoder: dict,
        emb_transforms: list[dict],
        decoders: list[dict],
        losses: list[dict],
        metrics: dict[str, dict[str, dict]],
    ):
        # 1) build all submodules from your YAML configs
        enc = instantiate_from_config(encoder)
        tfs = [instantiate_from_config(cfg) for cfg in emb_transforms]
        decs = [instantiate_from_config(cfg) for cfg in decoders]
        loss_fns = [instantiate_from_config(cfg) for cfg in losses]

        # metrics comes in as nested dict: { split: { name: cfg, … }, … }
        metric_maps = {
            split: {
                name: instantiate_from_config(cfg)
                for name, cfg in metrics[split].items()
            }
            for split in ("train", "val", "test")
        }

        # 2) hand everything off to BaseTask
        super().__init__(
            encoder=enc,
            emb_transforms=tfs,
            decoders=decs,
            losses=loss_fns,
            metrics=metric_maps,
            sample_rate=sample_rate,
            use_ema=use_ema,
        )

    def test_step(self, batch, batch_idx):
        waveforms, labels, file_paths = batch
        logits = self(waveforms)
        probs = torch.softmax(logits, dim=1).cpu()
        preds = torch.argmax(probs, dim=1)
        return {
            "file_paths": file_paths,
            "probs": probs,
            "preds": preds,
            "labels": labels.cpu(),
        }

    def test_epoch_end(self, outputs):
        # aggregate per‐slice → per‐file
        file_dict = {}
        for out in outputs:
            for fp, prob, lb in zip(out["file_paths"], out["probs"], out["labels"]):
                info = file_dict.setdefault(fp, {"probs": [], "label": int(lb)})
                info["probs"].append(prob.numpy())

        total, correct = 0, 0
        file_preds = {}
        for fp, info in file_dict.items():
            arr = torch.tensor(info["probs"])      # (n_slices, C)
            mean_prob = arr.mean(dim=0)           # (C,)
            pred = int(mean_prob.argmax().item())
            file_preds[fp] = pred
            total += 1
            correct += int(pred == info["label"])

        file_acc = correct / total
        self.log("test/file_acc", file_acc, prog_bar=True, on_epoch=True)
        return {"file_level_preds": file_preds, "file_acc": file_acc}