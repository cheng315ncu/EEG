"""
Train a ViT-Hybrid model on spectrogram datasets produced by generate_graph.py +
datasets_make.py.

Configuration is done via the constants below.
"""

from pathlib import Path
import torch
import numpy as np
import datasets
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    ViTHybridForImageClassification,
    ViTHybridConfig,
    ViTHybridImageProcessor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
import evaluate
from torch import nn
from torchvision.transforms import (
    Compose, Resize, RandomRotation, RandomResizedCrop, RandomAffine,
    ToTensor, Normalize
)
from collections import deque
from sklearn.metrics import (
    cohen_kappa_score, confusion_matrix, balanced_accuracy_score,
    hamming_loss, jaccard_score, top_k_accuracy_score, f1_score
)
from tqdm import tqdm

# ---------------------- CONFIGURATION (edit these) ---------------------- #
DS_DIR       = Path("./DS")       # where your .arrow files live
OUTPUT_ROOT  = Path("./models")   # where checkpoints + logs go
MODEL_ID     = "google/vit-hybrid-base-bit-384"
CHANNELS     = ["ECG", "C4-M1"]    # e.g. ["ECG","C4-M1","E1-M2"]
SUBFOLDERS   = ["CWT", "SSQ"]      # e.g. ["CWT","SSQ"]
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- Utility: Compute Dataset Mean/Std ---------------------- #
def compute_mean_std(hf_dataset):
    csum = torch.zeros(3)
    csum2 = torch.zeros(3)
    npix = 0
    to_tensor = ToTensor()
    for item in tqdm(hf_dataset, desc="Mean/Std"):
        x = to_tensor(item["image"])  # C×H×W
        c, h, w = x.shape
        npix += h * w
        csum  += x.sum(dim=[1,2])
        csum2 += (x**2).sum(dim=[1,2])
    mean = csum / npix
    std  = (csum2/npix - mean**2).sqrt()
    return mean.tolist(), std.tolist()

# ---------------------- Adaptive Weighting Callback ---------------------- #
class AdaptiveWeightCallback(TrainerCallback):
    def __init__(self, num_classes, window_size=3, eps=1e-6, min_w=0.1, max_w=10.0):
        self.num_classes = num_classes
        self.eps = eps
        self.min_w = min_w
        self.max_w = max_w
        self.hist = {i: deque(maxlen=window_size) for i in range(num_classes)}
    def set_trainer(self, trainer):
        self.trainer = trainer
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "f1_per_class" in metrics:
            f1s = metrics["f1_per_class"]
            w = []
            for i, f in enumerate(f1s):
                self.hist[i].append(f)
                avgf = sum(self.hist[i]) / len(self.hist[i])
                w.append(1.0 / (avgf + self.eps))
            w = torch.tensor(w, device=DEVICE)
            w /= w.mean()
            w = torch.clamp(w, self.min_w, self.max_w)
            if hasattr(self.trainer, "criterion") and self.trainer.criterion:
                self.trainer.criterion.class_weights = w
                print("=> Updated class weights:", w.cpu().numpy())
        return control

# ---------------------- Focal Loss + Label Smoothing ---------------------- #
class FocalLossWithLabelSmoothing(nn.Module):
    def __init__(self, alpha=0.5, gamma=1.5, eps=0.05, num_classes=5, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.num_classes = num_classes
        self.class_weights = class_weights
    def forward(self, logits, targets):
        one_hot = torch.nn.functional.one_hot(targets, self.num_classes).float()
        smooth = (1 - self.eps) * one_hot + self.eps / self.num_classes
        probs = torch.nn.functional.softmax(logits, dim=-1)
        logp  = torch.log(probs + 1e-12)
        focal = (1 - probs) ** self.gamma
        loss  = -smooth * focal * logp
        loss  = loss.sum(dim=1)
        if self.class_weights is not None:
            loss = loss * self.class_weights[targets]
        return (self.alpha * loss).mean()

# ---------------------- Custom Trainer to Use Our Criterion ---------------------- #
class WeightedTrainer(Trainer):
    def __init__(self, *args, criterion=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = criterion
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.criterion(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

# ---------------------- Metrics ---------------------- #
accuracy  = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall    = evaluate.load("recall")
f1        = evaluate.load("f1")
mcc       = evaluate.load("matthews_correlation")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    refs  = p.label_ids
    out = accuracy.compute(predictions=preds, references=refs)
    out.update( precision.compute(predictions=preds, references=refs, average="macro") )
    out.update( recall.compute(   predictions=preds, references=refs, average="macro") )
    out.update( f1.compute(       predictions=preds, references=refs, average="macro") )
    out.update( mcc.compute( predictions=preds, references=refs) )
    out["cohen_kappa"]       = cohen_kappa_score(refs, preds)
    out["balanced_accuracy"] = balanced_accuracy_score(refs, preds)
    out["hamming_loss"]      = hamming_loss(refs, preds)
    out["jaccard_macro"]     = jaccard_score(refs, preds, average="macro")
    out["f1_per_class"]      = f1_score(refs, preds, average=None).tolist()
    if p.predictions.shape[1] > 2:
        out["top_2_acc"] = top_k_accuracy_score(refs, p.predictions, k=2)
    return out

# ---------------------- Main Loop ---------------------- #
for ch in CHANNELS:
    for sub in SUBFOLDERS:
        print(f"\n===== CHANNEL={ch}  SUBFOLDER={sub} =====")
        train_path = DS_DIR / f"{ch}_{sub}_train.arrow"
        test_path  = DS_DIR / f"{ch}_{sub}_test.arrow"
        if not train_path.exists() or not test_path.exists():
            print(f"⚠️ Missing {train_path} or {test_path}, skipping.")
            continue

        ds_train = datasets.load_from_disk(str(train_path))
        ds_test  = datasets.load_from_disk(str(test_path))

        # initial class weights
        labels = np.array(ds_train["labels"])
        cw = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
        cw = torch.tensor(cw, dtype=torch.float, device=DEVICE)
        cw /= cw.sum()

        # model config
        label2id  = {n: i for i,n in enumerate(ds_train.features["labels"].names)}
        id2label  = {i: n for n,i in label2id.items()}
        num_labels = len(label2id)

        cfg = ViTHybridConfig.from_pretrained(
            MODEL_ID,
            num_labels=num_labels, id2label=id2label, label2id=label2id,
            hidden_dropout_prob=0.12, attention_probs_dropout_prob=0.12,
            embedding_size=128, depth=[4,6,12]
        )
        model = ViTHybridForImageClassification.from_pretrained(
            MODEL_ID, config=cfg, ignore_mismatched_sizes=True
        ).to(DEVICE)

        # mean/std
        mean_tr, std_tr = compute_mean_std(ds_train)
        mean_te, std_te = compute_mean_std(ds_test)
        proc = ViTHybridImageProcessor.from_pretrained(MODEL_ID)
        proc.do_normalize = True
        proc.image_mean    = mean_tr
        proc.image_std     = std_tr

        # transforms
        sz = cfg.image_size
        train_tf = Compose([
            Resize(sz), RandomRotation(5),
            RandomResizedCrop(sz, scale=(0.97,1.03), ratio=(0.97,1.03)),
            RandomAffine(0, translate=(0.03,0.03), scale=(0.97,1.03)),
            ToTensor(), Normalize(mean_tr, std_tr),
        ])
        test_tf = Compose([ Resize(sz), ToTensor(), Normalize(mean_te, std_te) ])

        class HFDataset(torch.utils.data.Dataset):
            def __init__(self, hf_ds, transform):
                self.ds = hf_ds; self.tf = transform
            def __len__(self): return len(self.ds)
            def __getitem__(self, i):
                ex = self.ds[i]
                img = self.tf(ex["image"])
                return {"pixel_values": img, "labels": ex["labels"]}

        train_ds_t = HFDataset(ds_train, train_tf)
        eval_ds_t  = HFDataset(ds_test,  test_tf)

        out_dir = OUTPUT_ROOT / f"{ch}_{sub}"
        args = TrainingArguments(
            output_dir=str(out_dir),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            push_to_hub=False,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            fp16=True,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            num_train_epochs=25,
            learning_rate=1e-5,
            warmup_ratio=0.15,
            weight_decay=0.1,
            logging_dir=str(out_dir / "logs"),
            logging_steps=20,
        )

        criterion = FocalLossWithLabelSmoothing(
            alpha=0.5, gamma=1.5, eps=0.05,
            num_classes=num_labels, class_weights=cw
        )
        adaptive_cb = AdaptiveWeightCallback(num_classes=num_labels)

        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=train_ds_t,
            eval_dataset=eval_ds_t,
            data_collator=lambda b: {
                "pixel_values": torch.stack([x["pixel_values"] for x in b]),
                "labels":          torch.tensor([x["labels"] for x in b])
            },
            compute_metrics=compute_metrics,
            criterion=criterion,
            callbacks=[adaptive_cb],
        )

        trainer.train()
