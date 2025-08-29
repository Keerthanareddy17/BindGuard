# BindGuard : AI Safety & Efficiency Layer for Pesticide Discovery 🌿

![inspired-by-bindwell](https://github.com/user-attachments/assets/cb7dc35d-cfb6-459c-8e3a-408e646f5dd9)
<svg xmlns="http://www.w3.org/2000/svg" width="213.390625" height="35" viewBox="0 0 213.390625 35"><rect width="116.046875" height="35" fill="#91e57e"/><rect x="116.046875" width="97.34375" height="35" fill="#11be3b"/></svg>

## Purpose & Motivation

The discovery of new pesticides is a complex, time-consuming, and expensive process. Traditional pipelines involve chemists designing candidate molecules and performing wet-lab experiments to test their efficacy and safety. This process is not only slow but also prone to high costs and regulatory hurdles. Moreover, even molecules that show strong binding to target pests can be unsafe for humans, bees, or the environment!!🐝

**Bindwell**, a pioneering startup in AI-driven pesticide discovery, has already revolutionized the process with their suite of tools. You can check them out [here](https://www.ycombinator.com/companies/bindwell) :)

These tools by **BindWell** allow for rapid in-silico evaluation of candidate molecules, drastically accelerating discovery. Despite their impressive capabilities, there are still two major challenges that I kinda observed :  

1. **Safety** – Ensuring candidate molecules are non-toxic to humans, beneficial organisms, and the environment.  
2. **Efficiency** – Selecting the most informative molecules to test from millions of possibilities without wasting compute resources.

---

## My Thoughts & Contribution: BindGuard 🌱

BindGuard is designed as a complementary layer, addressing the two gaps above. This project focuses on building a **prototype system** that demonstrates the following:

1. **Toxicity Prediction Module** – A toxicity filter that screens molecules for potential toxicity before they enter the main discovery pipeline.  
2. **Active Learning Agent** – An intelligent agent that selects the most informative molecules to test based on model uncertainty, reducing computational load while improving efficiency.

By combining these two components, BindGuard provides a practical, research-oriented solution that makes AI-driven pesticide discovery **safer, faster, and smarter**.  

This README will walk you through each component, explain its purpose, showcase outcomes, and provide guidance on usage and potential improvements.

---

## Component 1: Safety Prediction Module 🧪

### Overview
The first part of BindGuard is a **toxicity prediction module**. Its goal is to act as an early filter for candidate pesticide molecules, predicting potential toxicity to beneficial organisms. This helps prevent unsafe molecules from entering the downstream pipeline, saving time, computation, and regulatory headaches.  

 I've built a **prototype** using the publicly available **[Tox21 dataset](https://tripod.nih.gov/tox21/challenge/data.jsp)**, which provides bioassay results for 12 toxicity-related pathways. The module takes a `SMILES` string as input and outputs a probability for each assay indicating potential toxicity.

<img width="597" height="542" alt="Screenshot 2025-08-29 at 18 00 33" src="https://github.com/user-attachments/assets/2407c9e5-ce6c-42dc-bdab-48c49e189dfd" />

### How It Works
1. **Data Preparation**
   - Raw Tox21 SDF files are cleaned and converted into a usable format.
   - Missing labels are handled, and molecules without SMILES are dropped.
   - Molecules are converted into **Morgan fingerprints** (2048 bits, radius 2) using `RDKit`.

2. **Model Training**
   - A **multi-task `MLP`** is trained to predict the 12 toxicity endpoints simultaneously.
   - `BCEWithLogitsLoss` with class-specific weights is used to handle label imbalance.
   - Early stopping is applied to prevent overfitting.
   - The model is exported as a **TorchScript** module for easy inference.

3. **Inference**
   - Any new `SMILES` string can be fed into the module.
   - It outputs **toxicity probabilities** for all 12 assays.
   - Probabilities can be interpreted to flag molecules likely to be unsafe.

So, basically it demonstrates how a safety filter could plug into Bindwell’s AI-driven pesticide pipeline.

Here's a sample of how the prediction looks like for ethanol 

<img width="549" height="276" alt="Screenshot 2025-08-28 at 17 49 01" src="https://github.com/user-attachments/assets/137c0589-62cf-421c-a92e-04c244b8af61" />


### Limitations & Future Improvements
- **Dataset**: Only Tox21 is used, which covers human-related pathways; environmental/bee toxicity is not fully captured.  
- **Prototype Scale**: Currently trained on a subset. Scaling to millions of candidates may require more compute and optimized pipelines.  
- **Model**: `MLP` works well for prototyping, but `GNNs` could improve molecular feature extraction and prediction accuracy.  
- **Integration**: Currently standalone; further work could integrate it directly into active learning and multi-objective screening loops.


### File Structure 📂

BindGuard/
-  data :  Raw & cleaned Tox21 datasets
- models : Trained MLP and TorchScript model
- prepare_tox21.py : Prepares raw `CSV` from `SDF`
- preprocess_tox21.py : Handles SDF -> CSV conversion
- prepare_data_for_mlp.py : Prepares `PyTorch` tensors & DataLoaders
- train.py : Trains multi-task MLP
- inference.py : Predicts toxicity from SMILES

---

## Component 2: Active Learning Module 🚀

The second part of BindGuard is an **active learning loop** designed to efficiently explore the chemical space for promising pesticide candidates. Instead of blindly testing all molecules, the system **iteratively selects molecules where the surrogate model is most uncertain**, simulates labeling, and retrains the model. This approach accelerates learning and improves the predictions over time.

This module complements the **toxicity filter** from Component 1 by focusing on **binding optimization** while minimizing the number of molecules that need evaluation.

<img width="516" height="677" alt="Screenshot 2025-08-29 at 18 10 45" src="https://github.com/user-attachments/assets/f5e0487b-e34c-48de-881a-e049a23b35ff" />


### How It Works
1. **Surrogate Model**
   - An **MLP regression model** predicts a binding score from Morgan fingerprints (2048-bit).
   - Trained on an initial labeled subset of molecules.
   - Dropout layers are kept active during inference for **MC-Dropout uncertainty estimation**.

2. **Active Learning Loop**
   - Loads the **unlabeled pool** of molecules and computes fingerprints.
   - Uses the surrogate to predict binding scores and **estimate uncertainty** via `MC-Dropout`.
   - Selects the **top-K most uncertain molecules** for labeling (simulated here via existing data).
   - Updates labeled and unlabeled pools and **retrains the surrogate**.
   - Repeats for a configurable number of iterations to gradually reduce uncertainty and improve model accuracy.

3. **Metrics & Visualization**
   - Tracks **validation loss** and **average selected uncertainty** per iteration.
   - Provides plots to visualize surrogate improvement and convergence.
   
<img width="556" height="440" alt="Screenshot 2025-08-29 at 16 21 01" src="https://github.com/user-attachments/assets/279fef84-f042-4596-ab6a-ad33bb276144" />

<img width="800" height="500" alt="Figure_2" src="https://github.com/user-attachments/assets/d8d28821-d385-462e-8135-87cce0894950" />

<img width="800" height="500" alt="Figure_3" src="https://github.com/user-attachments/assets/7fe3816d-62e7-44c3-9f32-56e6af32b20b" />


### Outcome 📊
- Demonstrates an **iterative, data-efficient approach** for chemical exploration.
- Shows reduction in uncertainty over successive iterations, indicating better surrogate predictions.
- Provides a **framework** for integrating human-in-the-loop or experimental binding assays in the future.


### Limitations & Future Improvements
- **Simulated Labeling**: Currently uses existing binding scores; real experimental measurements may introduce noise or delay.  
- **Scale**: Prototype runs on a small, simulated dataset. Scaling to millions of molecules requires optimized data pipelines.  
- **Model Choice**: MLP is a baseline; graph-based or attention models could better capture molecular structure.  
- **Integration**: Could be combined with Component 1 to filter out toxic molecules during the active learning loop, enabling a multi-objective optimization framework.


### File Structure 📂

BindGuard / agent/
- data
  - molecules.csv : Full molecule dataset with binding scores
  - labeled_fp.csv : Labeled molecules with fingerprints
  - unlabeled_fp.csv : Unlabeled molecules with fingerprints
  - selected_for_labeling.csv : Top uncertain molecules per iteration
    
- model : Surrogate MLP state_dict
- active_learning.py : MC-Dropout selection logic
- auto_active_learning.py : Full active learning loop with retraining
- dataset.py : Labeled & unlabeled dataset classes
- model.py : Surrogate MLP definition
- prepare_fingerprints.py : Fingerprint generation
- update_pools.py : Updates labeled & unlabeled pools
- utils.py : Initial pool preparation
- train_surrogate.py : Initial surrogate training
- retrain_surrogate.py : Surrogate retraining during active learning

---

## So....... 
BindGuard shows a practical way to accelerate pesticide discovery while keeping safety front and center. 
- The toxicity module prevents risky molecules from progressing, and the active learning loop focuses testing on the most informative candidates, saving time, compute, and resources.
- Together, they provide a ready-to-integrate framework for faster, safer, and smarter molecule selection.
  
(ps: We'd need a lot more of actual data samples, that what I used in this repo, to make this work a lot better 👀)

---
![made-in-python](https://github.com/user-attachments/assets/40dc0f7b-e22a-49f0-b530-7dfe0adea2de)

For questions, feedback, or collaboration ideas, reach out at katasanikeerthanareddy@gmail.com

Here's my [LinkedIn](https://www.linkedin.com/in/keerthana-reddy-katasani-b07238268/) ✌️
