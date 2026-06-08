# su26-ai301-contribution
github-contribution-log for AI301 course at codepath
# Contribution [TorchIO-project/torchio #1187]: [Image intensity augmentations]

**Contribution Number:** [1]  
**Student:** [Mengyuan Ding]  
**Issue:** [GitHub issue link](https://github.com/TorchIO-project/torchio/issues/1187)
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

[In your own words, what's broken or missing?]

### Expected Behavior

[What should happen?]

### Current Behavior

[What actually happens?]

### Affected Components

[Which parts of the codebase are involved?]

---

## Reproduction Process

### Environment Setup

[Notes on setting up your local development environment - challenges you faced, how you solved them]

### Steps to Reproduce

1. [Step 1]
2. [Step 2]
3. [Observed result]

### Reproduction Evidence

- **Commit showing reproduction:** [Link to commit in your fork]
- **Screenshots/logs:** [If applicable]
- **My findings:** [What you discovered during reproduction]

---

## Solution Approach

### Analysis

[Your analysis of the root cause - what's causing the issue?]

### Proposed Solution

[High-level description of your fix approach]

### Implementation Plan

Using UMPIRE framework (adapted):

**Understand:** [Restate the problem]

**Match:** [What similar patterns/solutions exist in the codebase?]

**Plan:** [Step-by-step implementation plan]
1. [Modify file X to do Y]
2. [Add function Z]
3. [Update tests]

**Implement:** [Link to your branch/commits as you work]

**Review:** [Self-review checklist - does it follow the project's contribution guidelines?]

**Evaluate:** [How will you verify it works?]

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
