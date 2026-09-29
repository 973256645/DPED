# DPED: Distilling Preference Evolution Dynamics for Sequential Recommendation

This repository accompanies the manuscript **“DPED: Distilling Preference Evolution Dynamics for Sequential Recommendation.”** It currently contains a **partial release** of the research code. The available files illustrate selected preparation steps and a SASRec-based implementation of DPED; they do not constitute the complete pipeline used for all experiments in the paper.

## What is currently available

- Selected code for extracting a teacher's Top-100 item candidates.
- Selected code for constructing four chronological prefixes of a user's observed interaction history.
- A DPED implementation using SASRec as the student backbone.

The SASRec-based implementation is one instance of the proposed distillation method. DPED is a training approach that can be applied to different teacher–student configurations; the current repository does not contain the complete implementation for every configuration evaluated in the manuscript.

## Method overview

DPED extends final-state recommendation distillation with supervision at selected sequence prefixes. It aligns teacher and student representations at valid prefixes and aligns their item-ranking distributions at auxiliary prefixes. For distribution alignment, a global candidate pool is formed from the teacher's Top-$K$ items across the selected prefixes. Items already observed at a given prefix are masked, and the teacher and student are compared over the same remaining candidates. The final prefix contributes to the candidate pool but does not receive an additional auxiliary prefix-level ranking loss. These auxiliary objectives are used during training; the deployed student does not require extra inference components.

## Release scope and reproducibility

This is **not yet an end-to-end reproduction package**. Some source files and experimental configurations used for the manuscript are not included, so the current repository alone should not be expected to reproduce every reported result. The public datasets used in the paper are not redistributed here; please obtain them from their original providers and follow their terms of use.

**We will release the complete source code after the manuscript is accepted.** Until then, this repository is intended to document the available components and clarify their relationship to the method described in the paper.

## Contact

For questions about the manuscript or the current code release, contact Shuping Zhao (corresponding author): [zhaoshuping1753@hfut.edu.cn](mailto:zhaoshuping1753@hfut.edu.cn).
