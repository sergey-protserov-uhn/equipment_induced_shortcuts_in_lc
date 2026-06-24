import numpy as np
import scipy as sp
import torch as t
from sklearn import metrics
from torchmetrics import Metric


class NumPyMetricAdapter(Metric):
    def __init__(self, metric_func, **kwargs):
        super().__init__(**kwargs)
        self.metric_func = metric_func
        self.add_state("values", default=[])

    def update(self, preds, target):
        y_proba = (
            t.nn.functional.softmax(preds.detach(), dim=1)
            .moveaxis(1, -1)
            .cpu()
            .numpy()
        )
        y_true = target.detach().argmax(dim=1).cpu().numpy()
        self.values.extend(
            t.tensor(self.metric_func(y_true_, y_proba_))
            for y_true_, y_proba_ in zip(y_true, y_proba, strict=True)
        )

    def compute(self):
        return t.mean(t.stack(self.values), dim=0)


def f1_score(y_true, y_pred_or_proba, labels):
    if np.isdtype(y_pred_or_proba.dtype, "real floating"):
        y_pred = np.argmax(y_pred_or_proba, axis=-1)
    else:
        y_pred = y_pred_or_proba
    return metrics.f1_score(
        y_true.flatten(),
        y_pred.flatten(),
        labels=labels,
        average="macro",
        zero_division=1.0,
    )


def symmetric_hausdorff(y_true, y_pred_or_proba, labels):
    if np.isdtype(y_pred_or_proba.dtype, "real floating"):
        y_pred = np.argmax(y_pred_or_proba, axis=-1)
    else:
        y_pred = y_pred_or_proba
    label_hausdorffs = np.zeros(len(labels))
    for i, label in enumerate(labels):
        y_true_ = np.array(np.where(y_true == label)).T
        y_pred_ = np.array(np.where(y_pred == label)).T
        label_hausdorffs[i] = (
            sp.spatial.distance.directed_hausdorff(y_true_, y_pred_)[0]
            + sp.spatial.distance.directed_hausdorff(y_pred_, y_true_)[0]
        ) / 2
    return np.mean(label_hausdorffs)
