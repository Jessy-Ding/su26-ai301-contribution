# su26-ai301-contribution
github-contribution-log for AI301 course at codepath
# Contribution [TorchIO-project/torchio #1187]: [Image intensity augmentations]

**Contribution Number:** [1]  
**Student:** [Mengyuan Ding]  
**Issue:** [[GitHub issue link](https://github.com/TorchIO-project/torchio/issues/1187)]  
**Status:** [Phase I] [In Progress]

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

**Hypothesis (pre-registered, to avoid post-hoc rationalizing):**
- When the train and test intensity distributions differ (across MRI sequences),
- training with `IntensityInversion(p=0.5)` augmentation **improves generalization to unseen intensity distributions** (higher Dice on a held-out sequence), while **not significantly harming** in-domain performance.
This maps directly onto the issue's original motivation (single-sequence training → multi-sequence generalization).

**Experiment design — cross-modality holdout:**

| Element          | Setting                                                                                                                                                                         |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data             | A public **multi-contrast MRI** dataset, e.g. BraTS (each case has T1/T1ce/T2/FLAIR with a shared segmentation label) — natural fit: large intensity differences, shared labels |
| Task             | 2D/3D U-Net segmentation (slice-based is fine, lowers compute)                                                                                                                  |
| Train            | **One sequence only** (e.g. T1)                                                                                                                                                 |
| Test             | A sequence **unseen at train time** (e.g. FLAIR) = simulated modality shift                                                                                                     |
| Arm A (baseline) | Standard augmentation: flip / affine / noise                                                                                                                                    |
| Arm B            | A + `IntensityInversion(p=0.5)` — the **only** changed variable                                                                                                                 |
| Metric           | Dice on the held-out sequence (primary); Dice in-domain (guard against degradation)                                                                                             |
| Stats            | ≥3 random seeds; report mean ± std; paired test (Arm A vs B at the same seed)                                                                                                   |

**Success criterion (set in advance):** Arm B's held-out Dice is significantly higher than Arm A's (paired test, p < 0.05), and in-domain Dice drops by < 1–2 points. If met → evidence supports addition. If not met → honestly report "no benefit for this task" — itself a valuable result.

**Methodological discipline (or the evidence gets rejected):**
1. **Single variable:** A and B share data / model / LR / seed / steps; only the added augmentation differs.
2. **Seed-paired:** Arm A seed 0 vs Arm B seed 0, so the difference is attributed to the augmentation, not initialization noise.
3. **Report in-domain too:** reporting only the cross-modality gain while hiding whether in-domain degraded is a buried risk; report both.
4. **Honest about CT:** the maintainer specifically flagged CT. CT intensities (Hounsfield units) are physically meaningful and standardized, so inversion may be **harmful** there. Therefore:
	- **Narrow the scope:** claim this is an **MRI cross-sequence** augmentation; do **not** claim benefit for CT.
	- Optional bonus: run a small CT experiment; if it shows neutral/harmful, document "not recommended for CT" — that honesty makes the maintainer more likely to merge.

**Minimum viable version (limited compute):**
- A BraTS subset, 2D axial slices, a small U-Net, a few hundred steps; or
- A **mechanism-isolation synthetic experiment** first: on a single-sequence dataset, construct a test-time intensity transform the model never saw during training (simulated modality shift), and check whether A/B narrows the generalization gap. This cheaply validates the mechanism before scaling to real multi-sequence data.

**Deliverables (to post back on the issue/PR):**
- A reproducible `experiments/intensity_inversion_eval.py` (fixed seeds);
- A table: `{sequence pair, Arm A Dice, Arm B Dice, Δ, p-value}` across seeds;
- One or two qualitative figures: the same held-out slice, A vs B segmentation overlays;
- A conclusion plus the CT scope statement.


---

## Testing Strategy

### Unit Tests

- [ ] Test case 1: [Description]
- [ ] Test case 2: [Description]
- [ ] Test case 3: [Description]

### Integration Tests

- [ ] Integration scenario 1
- [ ] Integration scenario 2

### Manual Testing

[What you tested manually and results]

---

## Implementation Notes

### Week [X] Progress

[What you built this week, challenges faced, decisions made]

### Week [Y] Progress

[Continue documenting as you work]

### Code Changes

- **Files modified:** [List]
- **Key commits:** [Links to important commits]
- **Approach decisions:** [Why you chose certain approaches]

---

## Pull Request

**PR Link:** [GitHub PR URL when submitted]

**PR Description:** [Draft or final PR description - much of the content above can be adapted]

**Maintainer Feedback:**
- [Date]: [Summary of feedback received]
- [Date]: [How you addressed it]

**Status:** [Awaiting review / Iterating / Approved / Merged]

---

## Learnings & Reflections

### Technical Skills Gained

[What you learned technically]

### Challenges Overcome

[What was hard and how you solved it]

### What I'd Do Differently Next Time

[Reflection on your process]

---

## Resources Used

- [Link to helpful documentation]
- [Tutorial or Stack Overflow post that helped]
- [GitHub issues or discussions that helped]
