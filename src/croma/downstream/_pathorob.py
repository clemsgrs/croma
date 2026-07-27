"""VENDORED THIRD-PARTY CODE -- DO NOT EDIT.

Verbatim copies from PathoROB, the reference implementation of the downstream
shortcut-susceptibility protocol:

    https://github.com/bifold-pathomics/PathoROB
    revision 6583cf0b0d902c8cc032308262fa3a3befdc0687 (2026-04-02)

Three things are borrowed, and between them they are the protocol: the APD reduction,
the split-mapping helper that says how many training rows each (confounder, class) cell
contributes at a given split, and the logistic-probe trainer that is fitted on those
rows (with the module-level helper it maps over).

What is here is **frozen**. It is not croma's code and it is not croma's metric: the
paper reports APD as the faithful PathoROB reference, and that claim rests on running
PathoROB's own protocol rather than a re-derivation of it. The reduction is only the
last step of it -- a probe trained on a different schedule, or with a different
regularisation search, produces different accuracies, and APD read off those accuracies
is a different number under the same name. Editing anything below -- reformatting it,
adding validation, renaming a local, "fixing" the reduction order -- voids that claim
silently, because the value still computes.

Consequences that follow from freezing, and are intended (ADR-0011):

- Upstream fixes deliberately do **not** propagate. There is no sync process.
- Argument validation, type hints and croma's docstring conventions belong in the
  public wrappers next door (``croma.downstream.apd``, ``croma.downstream.probe``),
  never here.
- This file is excluded from ``black`` in ``pyproject.toml`` so that a formatter
  version bump cannot rewrite it.

An intentional re-vendor is a deliberate act: replace the copies below from the
upstream revision you are moving to, update the revision above, and re-capture both
parity fixtures as ``tests/fixtures/pathorob_apd_parity.json`` and
``tests/fixtures/pathorob_schedule_parity.json`` document.

PathoROB is distributed under the BSD 3-Clause License, reproduced in full below and
credited in croma's ``NOTICE``.

---------------------------------------------------------------------------------------
BSD 3-Clause License

Copyright (c) 2025, BIFOLD Pathomics

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
---------------------------------------------------------------------------------------
"""

# --- BEGIN VERBATIM: pathorob/apd/utils.py -------------------------------------------

import numpy as np


def compute_apd(accuracies):
    scores = np.asarray(accuracies)  # Shape: (num_splits, iterations)
    scores = (scores[1:] / scores[0]).mean(axis=0) - 1
    return scores


def get_patches_map_to_split(dataset, split, num_patches_per_slide):
    """
    Calculate number of training patches per category (med_center-bio_class-combination) for a given split.

    Args:
        dataset (str): The selected dataset; either `camelyon`, `tcga`, or `tolkach_esca`.
        split (int: [0, ..., splits-1]): The split for which the numbers are calculated.
        num_patches_per_slide (int): Number of patches per slide for downstream experiment.

    Returns:
        list of tuples (i, j, num_paches): List of numbers of training patches (num_patches) per category: med_center(i)-bio_class(j)-combination.
        int: Maximum number of training slides per category.
    """
    if dataset == "camelyon":
        tss0_pairs = [(0, 0, (7 - split) * num_patches_per_slide), (0, 1, (7 + split) * num_patches_per_slide)]
        tss1_pairs = [(1, 0, (7 + split) * num_patches_per_slide), (1, 1, (7 - split) * num_patches_per_slide)]
        return sorted(tss0_pairs + tss1_pairs), 14

    elif dataset == "tcga":
        diag_pairs = [(i, j, (split + 2) * num_patches_per_slide) for i in range(4) for j in range(4) if i == j]
        inv_diag_pairs = [(i, j, (1 if split % 2 == 1 else (2 if split < 3 else 0)) * num_patches_per_slide) for i in
                          range(4) for j in range(4) if i + j == 3]
        rest_pairs = [(i, j, (2 if split < 2 else (1 if split < 5 else 0)) * num_patches_per_slide) for i in range(4)
                      for j in range(4) if i != j and i + j != 3]
        return sorted(diag_pairs + inv_diag_pairs + rest_pairs), 8

    elif dataset == "tolkach_esca":
        tss0_pairs = [(0, j, (3 - split) * num_patches_per_slide) for j in range(3)] + [
            (0, j, (3 + split) * num_patches_per_slide) for j in range(3, 6)]
        tss1_pairs = [(1, j, (3 + split) * num_patches_per_slide) for j in range(3)] + [
            (1, j, (3 - split) * num_patches_per_slide) for j in range(3, 6)]
        return sorted(tss0_pairs + tss1_pairs), 6

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# --- END VERBATIM ---------------------------------------------------------------------

# --- BEGIN VERBATIM: pathorob/apd/train_model.py -------------------------------------
# numpy is already imported by the block above, where upstream imports it too.

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


def fit_and_evaluate(args):
    """
    Trains LR model for given c and evaluates performance on validation set.

    Args: 
        args tuple: c, train_x, train_y, val_x, val_y, eval_func.

    Returns:
        c: Specified regularization parameter.
        val_score: Performance score for validation set.
    """
    c, train_x, train_y, val_x, val_y, eval_func = args
    model = LogisticRegression(C=c, random_state=0)
    model.fit(train_x, train_y)
    val_score = eval_func(val_y, model.predict(val_x))
    return c, val_score


def train_logistic_regression(train_x, train_y, val_x, val_y, test_xs, test_ys, eval_func=balanced_accuracy_score):
    """
    Trains LR model using the optimal regularization parameter c selected from the validation set.

    Args:
        train_x (tuple with patch features), train_y (tuple with integer labels): Training set.
        val_x (tuple with patch features), val_y (tuple with integer labels): Validation set.
        test_xs (list with feature tuples), test_ys (list with label tuples): Test sets. 
        eval_func: Evaluation metric (default=balanced_accuracy_score).

    Returns:
        final_model: Trained LR model.
        best_c: Optimal regularization parameter c selected from the validation set.
        test_scores (list): Performance scores for each test set.
    """
    # Generate C values
    C_POWER_RANGE = np.linspace(-8, 4, 15)
    Cs = 10**C_POWER_RANGE
    
    # Grid search
    args_list = [(c, train_x, train_y, val_x, val_y, eval_func) for c in Cs]
    results = list(map(fit_and_evaluate, args_list))
    
    # Find best C
    best_c, best_score = max(results, key=lambda x: x[1])
    
    # Train final model with best C and evaluate on test sets
    final_model = LogisticRegression(C=best_c)
    final_model.fit(train_x, train_y)
    test_scores = [eval_func(test_y, final_model.predict(test_x)) for test_x, test_y in zip(test_xs, test_ys)]
    
    return final_model, best_c, test_scores


# --- END VERBATIM ---------------------------------------------------------------------
