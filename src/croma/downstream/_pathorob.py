"""VENDORED THIRD-PARTY CODE -- DO NOT EDIT.

Verbatim copies from PathoROB, the reference implementation of the downstream
shortcut-susceptibility protocol:

    https://github.com/bifold-pathomics/PathoROB
    revision 6583cf0b0d902c8cc032308262fa3a3befdc0687 (2026-04-02)

So far that is the APD reduction alone. The probe protocol borrows two more functions
from the same source -- the split-mapping helper and the logistic-probe trainer -- and
they land in this file, under the same rules, when that slice ships.

What is here is **frozen**. It is not croma's code and it is not croma's metric: the
paper reports APD as the faithful PathoROB reference, and that claim rests on running
PathoROB's own reduction rather than a re-derivation of it. Editing anything below --
reformatting it, adding validation, renaming a local, "fixing" the reduction order --
voids that claim silently, because the value still computes.

Consequences that follow from freezing, and are intended (ADR-0011):

- Upstream fixes deliberately do **not** propagate. There is no sync process.
- Argument validation, type hints and croma's docstring conventions belong in the
  public wrapper next door (``croma.downstream.apd``), never here.
- This file is excluded from ``black`` in ``pyproject.toml`` so that a formatter
  version bump cannot rewrite it.

An intentional re-vendor is a deliberate act: replace the copies below from the
upstream revision you are moving to, update the revision above, and re-capture the
parity fixture as ``tests/fixtures/pathorob_apd_parity.json`` documents.

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


# --- END VERBATIM ---------------------------------------------------------------------
