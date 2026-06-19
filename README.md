# su26-ai301-contribution

Open-source contribution log for the **AI301** course (CodePath, Summer 2026).
Each contribution lives in its own folder with a detailed, self-contained log
(problem, design, implementation, testing, evaluation, and maintainer feedback).

## Contributions

| # | Project | Issue / PR | Contribution | Status |
|---|---------|------------|--------------|--------|
| 01 | [TorchIO](https://github.com/TorchIO-project/torchio) | [#1187](https://github.com/TorchIO-project/torchio/issues/1187) · [PR #1482](https://github.com/TorchIO-project/torchio/pull/1482) | `IntensityInversion` transform — per-channel `max − x + min` intensity augmentation, with a three-stage efficacy study (synthetic + real-data) | Implementation complete & tested; real-data efficacy inconclusive — awaiting maintainer's merge decision |

_More contributions will be added as `contribution-02-…`, `contribution-03-…`, etc._

## Repository structure

```
su26-ai301-contribution/
├── README.md                                       # this overview / index
└── contribution-01-torchio-1187-intensity-inversion/
    └── README.md                                   # full log for contribution 01
```

Each contribution folder is self-contained: open its `README.md` to read the
whole story for that issue — the problem, the design decisions, the code
changes, the tests, the efficacy evaluation, and the back-and-forth with the
maintainers.

## Contribution 01 — TorchIO `IntensityInversion` (highlights)

- **What:** added a deterministic per-channel intensity-inversion transform
  (`x → max − x + min`) to TorchIO, with docs, 9 unit tests, and an
  `include`/`exclude`-safe inverse. Submitted as
  [PR #1482](https://github.com/TorchIO-project/torchio/pull/1482); also surfaced
  a pre-existing framework bug ([#1480](https://github.com/TorchIO-project/torchio/issues/1480)).
- **Efficacy (the hard part):** the maintainer asked for evidence on a real
  dataset. A synthetic toy showed a large gain under a clean polarity flip
  (+0.79 Dice), but a properly use-case-matched real-data study (CHAOS liver,
  T1→T2, 8 seeds) showed **no reliable benefit** (+0.017, within noise). The log
  reports this **honest negative result** in full rather than the inflated 3-seed
  figure — including how I distinguished a *broken* experiment from a genuine
  negative one.
- **Takeaway:** intensity inversion reliably helps only when the distribution
  shift is (close to) a global polarity flip; real cross-modality MRI shifts are
  more complex, so it does not by itself solve the cross-modality problem that
  motivated the issue.
