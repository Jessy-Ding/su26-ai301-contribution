# su26-ai301-contribution
github-contribution-log for AI301 course at codepath
# Contribution [TorchIO-project/torchio #1187]: [Image intensity augmentations]

**Contribution Number:** [1]  
**Student:** [Mengyuan Ding]  
**Issue:** [[GitHub issue link](https://github.com/TorchIO-project/torchio/issues/1187)]  
**Status:** [Phase II] [In Progress]

---

## Problem Summary

TorchIO issue #1187 requests new intensity-domain augmentation transforms for medical images — specifically intensity inversion (mapping each voxel x to max − x + min over the image's own range), along with related operations like cluster-remapping and contrast jitter. The motivation comes from a concrete clinical problem the issue author raised: models trained on a single MRI pulse sequence tend to latch onto intensity values as features and generalize poorly to other modalities with different intensity distributions. Intensity-shifting augmentations push a model to learn shape and contour instead of absolute intensity, improving cross-modality robustness. TorchIO currently ships only RandomGamma as an intensity-perturbing transform, and the maintainer (@romainVala) confirmed that inversion is "worth to add" because it makes a radical, physically meaningful change to contrast. My scope for this contribution is to implement the intensity-inversion transform cleanly — computing parameters per channel, applying it as a probabilistic on/off transform, leaving label maps untouched — with documentation and tests following the existing intensity-transform pattern, and to propose the remaining transforms as follow-up work.

---

## Why I Chose This Issue

This issue sits at the intersection of what I already know and what I want to learn next. I come from computational neuroimaging and have worked across several projects with BIDS-formatted MRI data and PyTorch-based preprocessing and augmentation pipelines, so I'm comfortable with the data and tooling this transform lives in. I haven't worked specifically on intensity-based augmentation before, though — that's a deliberate part of the appeal. I understand the cross-modality generalization problem the issue describes, but implementing a transform to address it is new ground for me, and I expect to be learning the intensity-augmentation literature and TorchIO's internals as I build. What draws me most is the chance to contribute at the methods level to a recognized medical-imaging library rather than only applying existing tools: writing a transform other researchers will use is a different kind of work than running a pipeline, and it's the muscle I most want to develop. Concretely, I'm hoping to learn how a mature library structures its transform API and testing conventions, how to write code that meets the standards of maintainers I don't know, and how to scope and defend a design decision through real review — the exchange with the maintainer about whether this is a "random" versus a binary-state transform has already pushed me to think harder about the semantics of what I'm building, not just whether it runs.

---

## Understanding the Issue

### Problem Description

TorchIO has no transform for **intensity inversion** (mapping a voxel `x` to `max - x + min`, per channel). The use case (issue #1187) is cross-modality MRI training: when a model trains on one pulse sequence but must generalize to others, it tends to latch onto modality-specific intensity distributions instead of shape/contour. An inversion augmentation forces the model to stop treating absolute intensity as a feature.

This is an **enhancement, not a bug** — nothing is broken; a capability is missing.

### Expected Behavior

A transform `IntensityInversion` should exist so that:

```python
import torchio as tio
subject = tio.IntensityInversion()(subject)        # always invert
subject = tio.IntensityInversion(p=0.5)(subject)   # invert ~half the time
```

Per channel, the brightest voxel becomes the darkest and vice versa (`max - x + min`); intensity range is preserved; label maps are untouched; and the op is invertible.

### Current Behavior

No such transform exists. `tio.IntensityInversion` and `tio.RandomIntensityInversion` both raise `AttributeError`. The only related names in the public API are `Flip` (spatial, not intensity) and `apply_inverse_transform` (the invertibility framework — unrelated). Users must hand-roll the operation outside the transform pipeline.

### Affected Components

- [src/torchio/transforms/intensity/](vscode-webview://0ss1bd828g61d8leh94u90erkrrt42iidvgp5s4q94mauifrlnul/src/torchio/transforms/intensity/) — new transform lives here (alongside `gamma.py`)
- [src/torchio/transforms/**init**.py](vscode-webview://0ss1bd828g61d8leh94u90erkrrt42iidvgp5s4q94mauifrlnul/src/torchio/transforms/__init__.py) — export
- [src/torchio/**init**.py](vscode-webview://0ss1bd828g61d8leh94u90erkrrt42iidvgp5s4q94mauifrlnul/src/torchio/__init__.py) — top-level export + `__all__`
- `tests/` — new test file
- `docs/reference/transforms/` — new doc stub

---

## Reproduction Process

### Environment Setup

- TorchIO 2.0.0a1, editable install from the repo.
- The v2 transform API differs from the v1 prototype shown in the issue: a transform subclasses `IntensityTransform` and implements `make_params(batch)` / `apply_transform(batch, params)` operating on a `SubjectsBatch` (see [gamma.py](vscode-webview://0ss1bd828g61d8leh94u90erkrrt42iidvgp5s4q94mauifrlnul/src/torchio/transforms/intensity/gamma.py)), rather than the v1 `__call__(subject)`.
- Construct test images the way [tests/test_clamp.py](vscode-webview://0ss1bd828g61d8leh94u90erkrrt42iidvgp5s4q94mauifrlnul/tests/test_clamp.py) does: `tio.ScalarImage(data)` (tensor passed positionally).

### Steps to Reproduce

1. `python -c "import torchio as tio; print(hasattr(tio, 'IntensityInversion'))"` → **`False`**.
2. Run [reproduce_1187.py](vscode-webview://0ss1bd828g61d8leh94u90erkrrt42iidvgp5s4q94mauifrlnul/reproduce_1187.py), which (a) confirms the transform is absent and (b) establishes the intended behavior with a manual per-channel `max - x + min` reference on a 2-channel image with mismatched ranges (`[0,100]` and `[-5,5]`).
3. [Observed result]

### Reproduction Evidence

- **Commit showing reproduction:** add `reproduce_1187.py` on branch `1187-IntensityInversion-transform` (pushed to `origin`).
- **Screenshots/logs:** - **Logs:** `reproduce_1187.py` output — section 1 prints `hasattr(...) = False`; section 2 prints the per-channel swaps; section 3 contrasts global vs per-channel ranges.
- **My findings:** 
    - The operation is **deterministic** (no continuous parameter to sample), so it is _not_ parametric like `Gamma`.
    - The only stochastic aspect — "apply or not with probability `p`" — is **already provided by the base `Transform` class** ([transform.py:82](vscode-webview://0ss1bd828g61d8leh94u90erkrrt42iidvgp5s4q94mauifrlnul/src/torchio/transforms/transform.py#L82), [transform.py:187](vscode-webview://0ss1bd828g61d8leh94u90erkrrt42iidvgp5s4q94mauifrlnul/src/torchio/transforms/transform.py#L187)). So a separate `RandomIntensityInversion` wrapper is redundant in v2 (which already merged `RandomGamma` into `Gamma`).
    - Min/max must be **per-channel**; the on/off decision is **per-image**.

---

## Solution Approach

### Analysis

The root "cause" is simply an unimplemented feature. The design question — is this a parametric `Random*` transform or a deterministic one? — resolves to **deterministic**: there's no value to sample, and probability-of-application is a base-class concern (`p`). This matches the maintainer's conclusion in the thread ("call it `IntensityInversion` and put it in the intensity transforms").

### Proposed Solution

Add a single deterministic `IntensityInversion(IntensityTransform)` in `transforms/intensity/`, computing `max - x + min` per channel, leaving label maps untouched, and declaring itself invertible (it is its own inverse). Stochastic use is achieved via the inherited `p`. No `Random*` wrapper.

### Implementation Plan

Using UMPIRE framework (adapted):

**Understand:** Add an intensity-inversion augmentation (`max - x + min`, per channel) to encourage modality-invariant learning. Deterministic op; stochastic only via `p`.

**Match:** [gamma.py](vscode-webview://0ss1bd828g61d8leh94u90erkrrt42iidvgp5s4q94mauifrlnul/src/torchio/transforms/intensity/gamma.py) is the closest pattern: subclasses `IntensityTransform`, iterates `self._get_images(batch)`, mutates `img_batch.data`, and implements `invertible`/`inverse`. `Gamma` also models per-channel-style ops and the invertibility hooks. Export/test/doc registration mirrors `Gamma` exactly.

**Plan:** 
1. **Add** `src/torchio/transforms/intensity/inversion.py` with `class IntensityInversion(IntensityTransform)`:
    - `make_params` → `{}` (nothing to sample).
    - `apply_transform` → for each intensity image, per channel `c`: `data[c] = data[c].max() - data[c] + data[c].min()`.
    - `invertible` → `True`; `inverse(params)` → returns `IntensityInversion(copy=False)` (self-inverse).
    - Docstring documents per-channel min/max and per-image on/off via `p`.
2. **Export** in [src/torchio/transforms/**init**.py](vscode-webview://0ss1bd828g61d8leh94u90erkrrt42iidvgp5s4q94mauifrlnul/src/torchio/transforms/__init__.py) and [src/torchio/**init**.py](vscode-webview://0ss1bd828g61d8leh94u90erkrrt42iidvgp5s4q94mauifrlnul/src/torchio/__init__.py) (`from .intensity.inversion import IntensityInversion` + add to `__all__`).
3. **Add tests** `tests/test_inversion.py` mirroring [test_gamma.py](vscode-webview://0ss1bd828g61d8leh94u90erkrrt42iidvgp5s4q94mauifrlnul/tests/test_gamma.py): changes data; double-application is identity; `apply_inverse_transform` restores original; labels unchanged; per-channel correctness on mismatched-range channels; `p=0` is a no-op.
4. **Add docs** `docs/reference/transforms/inversion.md` (`::: torchio.IntensityInversion`) and link it in the transforms nav next to `gamma.md`.

**Implement:** [GitHub fork on issue#1187](https://github.com/Jessy-Ding/torchio/tree/1187-IntensityInversion-transform) Branch `1187-IntensityInversion-transform` 

**Review:** Self-checklist: follows `CONTRIBUTING.rst`; mirrors `Gamma`'s structure and naming; type hints + docstring with `Examples`; no `Random` prefix; labels left untouched; `ruff`/formatting clean; per the maintainer's note ("I'd like to see some evidence that it improves results in an application"), keep the PR a clean, minimal building block.

**Evaluate**: split into two layers. Layer 1 (unit tests) proves the transform **behaves correctly**; that is the prerequisite for merging. Layer 2 (an empirical experiment) provides the **evidence of usefulness** the maintainer explicitly asked for before adding it to the library. These are decoupled: code correctness does not depend on the experiment, but the merge decision does.

*Layer 1 — Correctness.* Unit tests prove correct behavior: double-apply is identity, `apply_inverse_transform` restores the original, labels are untouched, min/max is per-channel. Necessary for merge, but **not** what the maintainer is asking for.

*Layer 2 — Efficacy (evidence the maintainer requested).*

The maintainer asked for a **basic experiment on a toy problem** ("run training
with vs without and compare"). We deliberately do **not** use BraTS or any other
real multi-sequence dataset: it requires downloads/registration/GPU (no reviewer
will reproduce it) and it mixes confounds (resolution, noise, anatomy contrast)
into the comparison, making any gain hard to attribute to inversion alone. A
**self-contained synthetic toy** meets the maintainer's minimum, runs on CPU in
minutes, is fully reproducible, and isolates the single variable we care about
(intensity polarity). We are not required to go beyond this; we add only the two
robustness checks below so the result cannot be dismissed.

**Hypothesis (pre-registered, to avoid post-hoc rationalizing):**
> When the train and test intensity distributions differ in polarity,
> training with `IntensityInversion(p=0.5)` **improves generalization to the
> unseen (inverted-contrast) distribution** (higher Dice), while **not
> significantly harming** in-domain (same-polarity) performance.

This maps directly onto the issue's original motivation (single-distribution
training → cross-distribution generalization) and onto the maintainer's "is it
useful" question.

**Experiment design — synthetic polarity holdout (CPU, no downloads):**

| Element | Setting |
|---|---|
| Data | Procedurally generated 2D images: random shapes (circles/squares) as foreground, **foreground brighter than background** |
| Label | The shape mask (segmentation task) |
| Train | **Bright-foreground polarity only** |
| Test (OOD) | **Inverted-contrast** (foreground darker than background) = simulated distribution shift |
| Test (in-domain) | Held-out **bright-foreground** images (guard against degradation) |
| Arm A (baseline) | Standard augmentation: flip / affine / noise |
| Arm B | A + `IntensityInversion(p=0.5)` — the **only** changed variable |
| Metric | Dice on the inverted test set (primary); Dice on the in-domain test set (guard) |
| Stats | 3 random seeds; report mean ± std; paired (Arm A vs B at the same seed) |

**Success criterion (set in advance):** Arm B's inverted-test Dice is clearly
higher than Arm A's (consistent across seeds), and in-domain Dice drops by < 1–2
points. If met → evidence supports addition. If not met → honestly report "no
benefit for this task" — itself a valuable result.

**Methodological discipline (or the evidence gets rejected):**
1. **Single variable:** A and B share data / model / LR / seed / steps; only the
   added augmentation differs.
2. **Seed-paired:** Arm A seed 0 vs Arm B seed 0, so the difference is attributed
   to the augmentation, not initialization noise.
3. **Report in-domain too:** reporting only the OOD gain while hiding whether
   in-domain degraded is a buried risk; report both.
4. **Honest about CT:** the maintainer specifically flagged CT. CT intensities
   (Hounsfield units) are physically meaningful and standardized, so inversion may
   be **harmful** there. We **narrow the scope** — claim this is an MRI-style
   cross-contrast augmentation and do **not** claim benefit for CT.

**Two robustness checks (so the toy result can't be dismissed):**
1. **Test perturbation ≠ training augmentation.** The training augmentation is the
   exact linear inversion `max - x + min`. If the OOD test set were built with the
   *same* formula, Arm B would essentially be tested on the one transform it was
   trained on — that proves memorization, not generalization. So the test set uses
   a **different monotonic intensity-decreasing map** (e.g. gamma-then-invert, or a
   random piecewise-linear inversion curve): bright↔dark order is flipped, but it
   is **not** that exact line. If B still wins, it has learned the more general
   invariance "don't rely on absolute brightness polarity," not a memorized curve.
2. **Report in-domain Dice as a no-side-effect guard.** Besides the inverted test
   set, evaluate a same-polarity (in-distribution) test set. If B matches A there
   (drop < 1–2 points), the cross-polarity generalization came **for free**,
   without sacrificing the original task.

**Deliverables (to post back on the issue/PR):**
- A reproducible `experiments/intensity_inversion_eval.py` (fixed seeds, CPU);
- A table: `{Arm A Dice, Arm B Dice, Δ}` for both the inverted and in-domain test
  sets, across seeds;
- One or two qualitative figures: the same inverted test image, A vs B
  segmentation overlays;
- A conclusion plus the CT scope statement.

---

## Testing Strategy

### Unit Tests

`tests/test_inversion.py` — 9 tests, all passing, robust under `pytest-randomly`:

- **Test case 1 — `test_changes_data`**: applying the transform actually modifies intensity data.
- **Test case 2 — `test_double_application_is_identity`**: applying inversion twice returns the original (self-inverse), within float32 tolerance.
- **Test case 3 — `test_p_zero_is_noop`**: `p=0.0` leaves data unchanged (stochastic gate works).
- `test_range_is_preserved`: per-channel min/max preserved after inversion.
- `test_dark_and_bright_are_swapped`: the brightest voxel becomes the darkest and vice versa.
- `test_per_channel`: two channels with mismatched ranges (`[0,100]` and `[-5,5]`) each invert within their own range — proves per-channel scope.
- `test_inverse`: `apply_inverse_transform()` restores the original.
- `test_inverse_respects_include`: with `include=["t1"]`, the inverse does **not** corrupt the un-inverted `t2` (regression guard for the framework bug I found).
- `test_leaves_labels_unchanged`: `LabelMap` data is untouched.

### Integration Tests

- **Integration scenario 1 — invertibility framework**: ran the transform through TorchIO's history/`apply_inverse_transform` machinery (not just the method in isolation), confirming registration in `_TRANSFORM_REGISTRY` and correct round-trip.
- **Integration scenario 2 — full suite regression**: ran the entire `tests/` suite (**1257 passed, 4 skipped**) to confirm the new transform + exports + nav change broke nothing.

### Manual Testing

- **Docs build**: `zensical build` → "No issues found" (verified the griffe `**kwargs` warning was resolved and the new nav entry renders).
- **Efficacy experiment** (`experiments/intensity_inversion_eval.py`): synthetic polarity-holdout segmentation, 3 seeds, CPU. Result: OOD Dice **0.04 → 0.84**, in-domain **−0.02**, consistent across seeds; produced an overlay figure showing the baseline segmenting the inverted background while the inversion-trained model segments correctly.
- **Flakiness hunt**: ran the suite 10× under randomized ordering + a 500-seed brute-force to confirm test tolerances were not RNG-fragile.

---

## Implementation Notes

### Week 1 Progress：Reproduce & design

Confirmed the feature was missing (`reproduce_1187.py`), pinned the intended behavior (`max − x + min`, per channel), and resolved the key design question: **deterministic transform, not a `Random*` wrapper** — probability-of-application is already a base-class concern (`p`), matching v2's merge of `RandomGamma` into `Gamma`.

### Week 2 Progress：Implement & test

Built the transform, exports, docs, and tests. Key challenge: the plan assumed a single-image (C, …) tensor, but in v2 apply_transform operates on a (B, C, I, J, K) batch — so min/max had to be computed per channel over the spatial dims (vectorized amax/amin), not via the naive per-channel loop. A naive loop would have treated the batch dim as channels.

### Week 3 Progress：Review hardening

Three issues surfaced and were fixed:

Found a pre-existing framework bug (filed as #1480): the transform history stores only name+params, dropping include/exclude, so the inverse corrupts un-transformed images — Gamma is affected too. Worked around it locally by recording processed image names in make_params and scoping the inverse via include.
Test flakiness: pytest-randomly reseeds the global RNG; default assert_close float32 tolerance (1e-5) was too tight for ×100 data round-trips (worst-case error ~1.5e-5). Applied the codebase's atol=1e-4 convention.
Docs warning: added a no-op __init__(**kwargs) (mirroring Transpose) so the documented **kwargs resolves to a real signature.

### Code Changes

- **Files modified:**
  
- `src/torchio/transforms/intensity/inversion.py` (new — the transform)
- `src/torchio/transforms/__init__.py` (export)
- `src/torchio/__init__.py` (top-level export + `__all__`)
- `tests/test_inversion.py` (new — 9 tests)
- `docs/reference/transforms/inversion.md` (new — doc stub)
- `zensical.toml` (nav entry)
  
- **Key commits:** 

- [d84ba50](https://github.com/TorchIO-project/torchio/pull/1482/commits/d84ba50e) — Add IntensityInversion transform
- [ad36e2b](https://github.com/TorchIO-project/torchio/pull/1482/commits/ad36e2b0) — Sort inversion.md alphabetically in docs nav (review feedback)

- **Approach decisions:** 

- Deterministic transform + base-class `p` instead of a `Random*` class (less API surface, matches v2 conventions).
- Per-channel (not global) min/max — a global reflection would leak one channel's scale into another.
- Local include/exclude-safe inverse rather than a framework rewrite — keeps PR scope tight; the root-cause fix is tracked separately in #1480.
- Synthetic CPU toy for efficacy evidence rather than a real multi-sequence dataset — reproducible by any reviewer in minutes and isolates the single variable (intensity polarity).

---

## Pull Request

**PR Link:** [GitHub PR URL submitted](https://github.com/TorchIO-project/torchio/pull/1482)

**PR Description:** _(as submitted — see `pr_body.md`)_ Adds `IntensityInversion`, a deterministic per-channel intensity inversion (`max − x + min`); stochastic via the base-class `p`; self-inverse with an `include`/`exclude`-safe inverse; CT scope caveat in the docstring. References (does not close) the umbrella issue #1187, and notes the #1480 framework bug it works around. Includes efficacy evidence (synthetic toy, 3 seeds).

**Maintainer Feedback:**
- **2026-06-18**: Copilot (AI reviewer, Low severity) flagged that `zensical.toml`'s Intensity list is alphabetical and `inversion.md` was inserted out of order.
- **2026-06-18**: Addressed in commit `ad36e2b` (moved it after `histogram_standardization.md`); replied and resolved the conversation.

**Status:** [Awaiting review]

---

## Learnings & Reflections

### Technical Skills Gained

- The TorchIO v2 transform API: `IntensityTransform`, `make_params`/`apply_transform` over a `SubjectsBatch`, and the invertibility/history framework (`AppliedTransform`, `get_inverse_transform`).
- Vectorized per-channel tensor ops with `amax`/`amin` + `keepdim` broadcasting over batch tensors.
- Writing RNG-robust tests: understanding `pytest-randomly`, and why float32 round-trips need explicit tolerances.
- The fork-based open-source workflow: fork/upstream remotes, `commit --amend` + `--force-with-lease` (before a PR exists) vs. additive commits (after), responding to review comments.

### Challenges Overcome

- **Batch vs single-image semantics**: the plan's reference implementation was per-image, but the real API is batched — caught it by inspecting `ImagesBatch` shape rather than trusting the plan.
- **Distinguishing my bug from the library's bug**: when the include/exclude inverse failed, I reproduced it on `Gamma` too, proving it was a pre-existing framework issue, not my code — which changed the fix from "patch my transform" to "work around locally + report upstream."
- **Flaky test triage**: a test that passed alone but failed in the suite — traced to global-RNG reseeding rather than a logic error.

### What I'd Do Differently Next Time

- **Check the actual data shape/API contract before trusting a plan** — the per-channel batch dimension would have saved a rework.
- **Set test tolerances deliberately from the start** when arithmetic + randomized data are involved, instead of relying on `assert_close` defaults.
- **Verify ordering/style conventions** (alphabetical nav lists) before submitting, to pre-empt trivial review nits.

---

## Resources Used

- [Link to helpful documentation]
  
- [Tutorial or Stack Overflow post that helped]

- [GitHub issues or discussions that helped]
- [issue 1187](https://github.com/TorchIO-project/torchio/issues/1187)
