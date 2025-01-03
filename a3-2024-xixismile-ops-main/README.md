[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/00EWmpN0)
## **Assignment Description**
- In this assignment, you will be working on natural language generation (NLG). You will be exploring various ways to generate text and demonstrate your understanding of decoding algorithms, the effect of their parameters, and NLG evaluation.
    
- In Part 1, you will implement two decoding algorithms (greedy and beam search), as well as two sampling algorithms (top-p and top-k) to replicate (to some extent) what one would get when using Huggingface's `generate` function that you've played with during the Week 6's exercise session.

- For Part 2, you will be implementing Contrastive Decoding, a method that combines the logits of two models at generation time.
    
- For Part 3, you will be evaluating NLG metrics for machine translation.

### Table of Contents
- **[Setup](#setup)**
    - [1) Google Setup](#1-google-colab-setup)
    - [2) Local Setup](#2-local-setup)
    - [3) Rest of the Setup](#3-rest-of-the-setup-colab-and-local)

- **[PART 1: NLG Decoding and Sampling Algorithms](#part-1-nlg-decoding-and-sampling-algorithms)**
    - [1.1) Implement decoding and sampling algorithms](#11-implement-decoding-and-sampling-algorithms)
    - [1.2) Test your implementations](#12-testing-your-implementation)
    
- **[PART 2: Constractive Decoding](#part-2-contrastive-decoding)**
    - [2.1) Implement the contrastive decoding method with adaptive plausibility constraint](#21-implement-contrastive-decoding-with-adaptive-plausibility-constraint)
    - [2.2) Evaluate your generations using the MAUVE metric](#22-evaluate-your-generations-using-the-MAUVE-metric)

- **[PART 3: MT Evaluation](#part-3-mt-evaluation)**
    - [3.1) Dataset and metrics analysis](#31-dataset-and-metrics-analysis)
    - [3.2) NLG metric calculation](#32-nlg-metric-calculation)
    - [3.3) Correlation calculation](#33-correlation-calculation)
    - [3.4) Correlation analysis](#34-correlation-analysis)

- **[PART 4: Checklist](#part-4-checklist)**
    
### Deliverables

To give us the deliverables you will have to commit the following files to your github classroom repository:

- ✅ The jupyter notebook: `a3_notebook.ipynb`

- ✅ The python files:
    - [ ] `a3_utils.py`, if you added any helper functions
    - [ ] `a3_decoding.py`
    - [ ] `a3_sampling.py`
    - [ ] `a3_contrastive_decoding.py`
    - [ ] `a3_contrastive_main.py`
    - [ ] `a3_mt_eval.py`

- ✅ The Part 3 open answer MD file: `a3_mt_qa.md`

- ✅ The JSON files generated in Parts 2 & 3: 
    - [ ] `part2_contrastive_generations.json`
    - [ ] `part2_greedy_generations.json`
    - [ ] `part3_metrics.json` 
    - [ ] `part3_corr.json`

### Expected Workload & Resources

We expect the first part of the assignment, notably Beam search, to be the longest part of the assignment. You can plan your workload according to that. Keep in mind that this is just our expectation, the same may not apply to everyone! It would be helpful to finish Part 1 to do Part 2. Part 3 can be done independently.

This assignment does not necessarily require a GPU to be completed (*i.e.*, there is no training -- only inference), although some processes such as decoding and model-based metric calculation can be sped up on GPU. Therefore for Part 1's beam search, all of Part 2, and Part 3's metric calculation it may be a good idea to use a GPU.

### Grade Breakdown
Here is the general grade breakdown per section, to help with prioritization:

- Part 1: Decoding & Sampling algorithms (100)
    - Greedy decoding: 15
    - Beam decoding: 40
    - Top-k sampling: 20
    - Top-p sampling: 25
- Part 2: Contrastive Decoding (60)
- Part 3: MT Evaluation (60)