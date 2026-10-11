"""Confidence-based label-free accuracy predictors.

ATC (Garg et al., ICLR 2022): choose the threshold t on in-domain validation
confidences so that the share of examples below t equals the validation error;
predicted target accuracy = share of target confidences >= t.

DoC (Guillory et al., ICCV 2021): predicted target accuracy = source accuracy
minus the drop in mean confidence from source to target.
"""
from __future__ import annotations

import numpy as np


def atc_predict(val_conf: np.ndarray, val_correct: np.ndarray, tgt_conf: np.ndarray) -> float:
    val_conf, tgt_conf = np.asarray(val_conf, float), np.asarray(tgt_conf, float)
    err = 1.0 - float(np.mean(val_correct))
    t = float(np.quantile(val_conf, err)) if 0 < err < 1 else (val_conf.min() if err == 0 else np.inf)
    return float(np.mean(tgt_conf >= t))


def doc_predict(val_conf: np.ndarray, val_correct: np.ndarray, tgt_conf: np.ndarray) -> float:
    return float(np.mean(val_correct) - (np.mean(val_conf) - np.mean(tgt_conf)))
