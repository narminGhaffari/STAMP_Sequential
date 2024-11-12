import torch
from torch import nn
import torch.nn.functional as F
from fastai.vision.all import (
    Learner, DataLoader, DataLoaders, RocAuc,
    SaveModelCallback, CSVLogger, EarlyStoppingCallback,
    MixedPrecision, AMPMode, OptimWrapper
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .data import make_dataset, SKLearnEncoder
from .TransMIL import TransMILWithSequencePrediction

from typing import Any, Iterable, Optional, Sequence, Tuple, TypeVar
from pathlib import Path
from functools import partial
from sklearn.metrics import roc_auc_score

from fastai.callback.core import Callback

class DebugCallback(Callback):
    def after_epoch(self):
        print(f"Metrics after epoch {self.epoch + 1}: {self.recorder.metrics}")
        
__all__ = ['train', 'deploy']

T = TypeVar('T')

def train(
    *,
    bags: Sequence[Iterable[Path]],
    targets: Tuple[None, np.ndarray],  # Direct sequence of binary or categorical targets
    sequence_length: int,  # Number of time steps in each target sequence
    num_classes: int = 1,  # Number of classes for prediction at each time step (1 for binary classification)
    add_features: Iterable = [],  # No additional features
    valid_idxs: np.ndarray,
    n_epoch: int = 32,
    patience: int = 8,
    path: Optional[Path] = None,
    batch_size: int = 64,
    cores: int = 8,
    plot: bool = False
) -> Learner:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == "cuda":
        torch.set_float32_matmul_precision("medium")

    # Directly access the target sequence values
    _, targs = targets  # Expecting targs of shape (num_samples, sequence_length)
    # Determine the target type based on the number of classes
    target_type = torch.float32 if num_classes == 1 else torch.long

    # Convert targs to the appropriate numpy type before creating the dataset
    targs = targs.astype(np.float32 if target_type == torch.float32 else np.int64)

    # Create datasets, specifying target_type
    train_ds = make_dataset(
        bags=bags[~valid_idxs],
        targets=(targs[~valid_idxs]),
        add_features=[],
        bag_size=512,
        target_type=target_type  # Pass target_type here
    )

    valid_ds = make_dataset(
        bags=bags[valid_idxs],
        targets=(targs[valid_idxs]),
        add_features=[],
        bag_size=None,
        target_type=target_type  # Pass target_type here
    )

    # Build dataloaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=cores, device=device)
    valid_dl = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=cores, device=device)
    dls = DataLoaders(train_dl, valid_dl, device=device)

    # Initialize the model with sequence length and num_classes
    feature_dim = train_ds[0][0].shape[-1]
    model = TransMILWithSequencePrediction(
        num_classes=num_classes, sequence_length=sequence_length, input_dim=1024, dim=512,
        depth=2, heads=8, dim_head=64, mlp_dim=512, dropout=.0
    ).to(device)
    
    model.to(device)
    print(f"Model: {model}", end=" ")
    print(f"[Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}]")

    # Use a suitable loss function
    loss_func = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()

    class SequenceRocAuc:
        def __init__(self):
            self.name = "SequenceROC"
            self.reset()

        def reset(self):
            self.preds = []
            self.targets = []

        def accumulate(self, preds, targs):
            # Calculate probability for class 1 using sigmoid on logit differences
            preds = torch.sigmoid(preds[:, :, 1] - preds[:, :, 0])  # Probability for class 1
            preds = preds.flatten().cpu().numpy()  # Flatten to 1D array

            # Flatten targets to 1D array
            targs = targs.flatten().cpu().numpy()

            # Accumulate for the entire batch
            self.preds.extend(preds)
            self.targets.extend(targs)

        def value(self):
            try:
                # Compute ROC AUC using accumulated predictions and targets
                score = roc_auc_score(self.targets, self.preds)
                print(f"Calculated ROC-AUC score: {score}")  # Debugging line
                return score
            except ValueError:
                print("ValueError in ROC calculation, returning NaN")  # Debugging line
                return float("nan")

        def __call__(self, preds, targs):
            self.reset()
            self.accumulate(preds, targs)
            return self.value() or 0.0  # Ensure it never returns None


    # Create fastai Learner with the custom metric for sequence ROC
    learn = Learner(
        dls,
        model,
        loss_func=loss_func,
        opt_func=partial(OptimWrapper, opt=torch.optim.AdamW),
        metrics=[SequenceRocAuc()],
        path=path,
    )

    # Callbacks for training
    cbs = [
        SaveModelCallback(monitor='valid_loss', fname=f'best_valid'),
        EarlyStoppingCallback(monitor='valid_loss', patience=patience),
        DebugCallback()
    ]
    print("Training with parameters:", {
    "path": path,
    "n_epoch": n_epoch,
    "metrics": learn.metrics
    })
    
    learn.fit_one_cycle(n_epoch=n_epoch, reset_opt=True, lr_max=1e-4, wd=1e-2, cbs=cbs)

    return learn


def deploy(
    test_df: pd.DataFrame, learn: Learner, *,
    target_label: Optional[str] = None,
    cat_labels: Optional[Sequence[str]] = None, cont_labels: Optional[Sequence[str]] = None,
    device: torch.device = torch.device('cpu')
) -> pd.DataFrame:
    assert test_df.PATIENT.nunique() == len(test_df), 'duplicate patients!'

    if target_label is None: target_label = learn.target_label
    if cat_labels is None: cat_labels = learn.cat_labels
    if cont_labels is None: cont_labels = learn.cont_labels

    target_enc = learn.dls.dataset._datasets[-1].encode
    categories = target_enc.categories_[0]
    add_features = []
    if cat_labels:
        cat_enc = learn.dls.dataset._datasets[-2]._datasets[0].encode
        add_features.append((cat_enc, test_df[cat_labels].values))
    if cont_labels:
        cont_enc = learn.dls.dataset._datasets[-2]._datasets[1].encode
        add_features.append((cont_enc, test_df[cont_labels].values))

    test_ds = make_dataset(
        bags=test_df.slide_path.values,
        targets=(target_enc, test_df[target_label].values),
        add_features=add_features,
        bag_size=None)

    test_dl = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=1,
        device=device, pin_memory=device.type == "cuda")

    # Sequential predictions
    patient_preds, patient_targs = learn.get_preds(dl=test_dl, act=nn.Softmax(dim=2))

    # Convert predictions to DataFrame with the ground truth
    patient_preds_df = pd.DataFrame.from_dict({
        'PATIENT': test_df.PATIENT.values,
        target_label: test_df[target_label].values,
        **{f'{target_label}_{cat}_step_{i}': patient_preds[:, i, idx]
            for i in range(patient_preds.size(1))
            for idx, cat in enumerate(categories)}
    })

    # Calculate loss across sequences
    patient_preds_flat = patient_preds.view(-1, patient_preds.size(-1))
    patient_targs_flat = target_enc.transform(
        test_df[target_label].values.reshape(-1, 1))
    patient_preds_df['loss'] = F.cross_entropy(
        torch.tensor(patient_preds_flat), torch.tensor(patient_targs_flat),
        reduction='none').view(patient_preds.size(0), -1).mean(1)

    # Add a column for final prediction (could be a custom post-processing)
    patient_preds_df['pred'] = categories[patient_preds[:, -1, :].argmax(1)]

    return patient_preds_df
