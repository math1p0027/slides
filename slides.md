# A Review of Causality for Learning Algorithms in Medical Image Analysis

 Athanasios Vlontzos, Daniel Rueckert, Bernhard Kainz

Sofia Zervaki, 2025

---
# Introduction

- Medical imaging covers MRI, CT, X-ray, and Ultrasound → crucial for diagnosis & monitoring.

- ML has shown success in lesion detection, segmentation, and scan alignment.

- Many AI methods fail in clinical practice due to poor robustness & skipped readiness stages

--


<img src="figures/Figure1(2).png" alt="Histogram" width="1000">

--

- TRLs describe the stages from research -> deployment.

- Many ML systems in medical imaging jump from TRL 4 (PoC) → TRL 7 (Integration).

- This skips TRL 5–6, where algorithms are made robust and production-ready.
    
---

#  Why Causality Matters in Medical Imaging

- Current AI/ML often confuses correlation and causation

  - Example: COVID-19 X-rays ->  models learned hospital IDs/ethnicity, not disease (DeGrave et al., 2021).

- Domain shifts reduce robustness:

  - Population shifts – disease prevalence differs across regions.

  - Acquisition/annotation shifts – scanner settings, radiologist biases.

  - Data selection bias – limited datasets in medical domains.

- Causal analysis can mitigate these biases → safer & more adaptable ML.

--
# Background 

- *Structural Causal Models (SCM)*

  - Represent cause–effect relations with variables and functions.

  - Use Directed Acyclic Graphs (DAGs) to show dependencies.

- *Do-Operator (do(x))*

  - Simulates interventions: "What happens if we force X = x?"

  - Lets us estimate causal effects, not just correlations.


--

  
# Counterfactual Inference & Twin Networks


- Counterfactuals “What would Y have been if X had been different?”

- Computed using SCM and latent variables (U).

- Two main methods

  - Abduction–Action–Prediction: infer latent U → intervene → predict outcome.

  - Twin Network: duplicate model for factual & counterfactual worlds → jointly compute effects.



--
<img src="figures/Figure3(2).png" alt="Histogram" width="1000">

---

# Potential Outcomes & Average Treatment Effect (ATE)


- Potential Outcomes: Predict what would happen with vs. without a treatment.

 - \Y_1,i = outcome for unit i receiving the treatment
 - \Y_0,i = outcome for unit i not receiving the treatment

- Causal Effect (Individual): Difference between outcomes:
 - \tau = Y_1,i - Y_0,i


---
**Key Concepts**
- Causal Markov Condition: $X_i$ is independent of non-descendants given $PA_i$
- Causal Factorization:
    $$ P(X_1, X_2,...,X_n) = \prod_{i=1}^n  P(X_i|PA_i)$$

    Which reflects causal structure, unlike arbitrary factorizations
- Latent Variables: some variables may be unobserved (latent), which can confuse causal relationships
- Learning: Observational data + independence tests -> partial graph structure


--

 **Interventions in SCMs**


- No intervention: Passive observation

- Hard (perfect): Set variable to fixed value

- Soft (imperfect): Modify function or noise 

- Uncertain: Don’t know which variable or function was changed.

Knowing which interventions are possible/allowed helps guide causal discovery and ensures the model is useful for real tasks.


 

--
# D. Difference Between Statistical Models, Causal Graphical Models, and SCMs

- **Statistical models** capture correlations but can't distinguish cause from effect or handle interventions.

- **Causal graphical models** add directionality (cause -> effect) and allow us to model interventions.

- **Structural Causal Models (SCMs)**:
  - Define variables as functions of their causes and noise: $X_i = f_i(PA_i,U_i)$
  - Enable reasoning about interventions (changing the system) and counterfactuals (what would have happened).



---

#  IV. INDEPENDENT CAUSAL MECHANISMS

- A system's generative process consists of independent modules (mechanisms) that do not inform or influence one another.

- This implies:

  - Modularity: Each causal mechanism operates independently.

  - Separate intervenability: You can intervene on one mechanism without affecting others.

  - Invariance: Mechanisms stay stable across settings or interventions.



--
**Sparse Mechanism Shift (SMS) Hypothesis**

- When distribution shifts occur (e.g., across domains), they typically affect only a few mechanisms in the causal model.

- Contrast:
  - Causal factorization: Shift is localized, easier to generalize across settings.
  - Statistical (entangled) factorization: Shift spreads across many terms, harder to adapt.

- Supports robust generalization, transfer learning, and domain adaptation

--
**Algorithmic View of Independent Causal Mechanisms (ICM)**

- Mechanism independence can be formalized via algorithmic information theory:

  - A mechanism = bit string (shortest program encoding it).

  - Two mechanisms are independent if compressing them together doesn’t save space.

- Suggests causality is not limited to statistics.

  - Leads to algorithmic causal models using Kolmogorov complexity.

  - Independent programs (noise terms) ⟺ Independent mechanisms


--


<img src="figures/Fig1.png" alt="Histogram" width="1000">



---

#  V. CAUSAL DISCOVERY AND MACHINE LEARNING

- Learn causal structure from data 
- Challenges:

  - Conditional independence testing is hard with finite, high-dimensional data.

  - Doesn’t work well in the 2-variable case.

- Solution: Use assumptions about function classes (common in ML).

--

**Additive Noise & Distribution Shifts**

- Additive Noise Models (ANMs):

$$Y = f(X) + noise$$

  - Helps identify causal direction (X→Y fits, Y→X doesn't).

- Distribution Shifts Help:

  - Causal structure is invariant across environments.

  - Use data from different contexts/tasks (e.g., interventions, time, views).


---



# VI. LEARNING CAUSAL VARIABLES


--




<img src="figures/Fig2.png" alt="Histogram" width="1000">


--

# The goal of causal representation learning

- Observations $X = G(S_1 ... S_n)$. High-dimensional data is a nonlinear mixture of underlying causal variables.

- Goal: Learn a representation that reveals these causal variables and their relations.

- Neural networks map raw data to meaningful high-level variables to support downstream tasks.

- Causal inductive biases like Sparse Mechanism Shift (SMS) hypothesis help learning.

--

**Challenges in Learning Causal Variables**

- Causal variables are not given, they must be learned from data.

- Variables depend on the granularity of the data and available interventions or distribution shifts.

- Representation learning aims for robustness, interpretability, and fairness.

- Need to embed causal models inside larger ML models with high-dimensional inputs/outputs.

--

# Problem 1 — Learning Disentangled Representations

- Disentanglement means representing data as independent factors corresponding to causal variables.

- The Independent Causal Mechanisms (ICM) principle implies noise variables should be independent.

- Encoder-decoder frameworks (autoencoders) can learn latent variables that ideally correspond to causal factors.

- Object-centric learning as a special case of disentanglement.

--
# Using the ICM Principle for Causal Learning

- Make noise terms $U_i$ statistically independent.

- Mechanisms should be independently manipulable and invariant across problems.

- SMS hypothesis: sparse changes in mechanisms help identify causal factors.

- Different supervision signals or interventions affect which variables can be disentangled.

--
# Problem 2 — Learning Transferable Mechanisms

- Real-world agents face limited data and computational resources.

- Modular structures reflecting the modularity of the world help reuse knowledge across tasks.

- Example: Visual system factors out lighting variations rather than relearning face recognition.

- Bias towards models with structural similarity (homomorphism) to the real world's modular structure.

--

# Problem 3 — Learning Interventional World Models and Reasoning

- Deep learning captures statistical dependencies but ignores causal intervention properties.

- Causal models enable reasoning, planning, and imagining alternative scenarios (thinking as acting in imagined space).

- Importance of representing oneself and “free will” for social and cultural learning (future research frontier).

--

<img src="figures/Fig3.png" alt="Histogram" width="600">

---

# VII. IMPLICATIONS FOR MACHINE LEARNING
##  A. Semi-Supervised Learning (SSL)



- Traditional ML assumes data is i.i.d. (same distribution in train & test)

- Causal perspective: distributions may change, but causal mechanisms stay mostly stable

- This challenges how we use unlabeled data in SSL

--

### SSL and Causal Direction

Assume causal graph: X -> Y (X causes Y)

Joint distribution factorizes as  $ P(X, Y) = P(X) \times P(Y | X) $

ICM Principle:  $P(X)$ and $P(Y | X)$ are independent

Implication: Knowing $P(X)$ (unlabeled data) does not help improve $P(Y | X)$

Conclusion ->  SSL is expected to be ineffective in this causal direction

--

**SSL works better in the Anticausal Direction**

- Predicting cause from effect ( Y -> X)
- Marginal $P(X)$ contains info about conditional $P(Y|X)$
- Unlabeled data helps improve learning



--

**Relation to SSL Assumptions**

- Cluster assumption: labels stable within clusters of $P(X)$

- Low-density separation: decision boundaries lie in low-density regions

- Smoothness: nearby points in $P(X)$ have similar outputs

- These assumptions align with SSL success in anticausal settings




---
# VII. IMPLICATIONS FOR MACHINE LEARNING
## B. Adversarial Vulnerability

- Adversarial attacks violate i.i.d. -> exploit non-causal features.

- Causal models (true generative direction) may be more robust.

- Robustness improves when classifiers align with causal structure.

- Defenses:
  - Analysis by synthesis (model label -> input)
  - Autoencoder preprocessing to remove spurious perturbations


---
# VII. IMPLICATIONS FOR MACHINE LEARNING
## C. Robustness and Strong Generalization


- Causal models enable robustness to interventions & distribution shifts.

- Use causal features (not just correlations) for better generalization.

- Goal: Minimize worst-case risk across environments (e.g., different settings).

- Requires modeling interventions and training on diverse data.




---
# VII. IMPLICATIONS FOR MACHINE LEARNING
## D. Pre-training, Data Augmentation, and Self-Supervision

- Pre-training: Use large, diverse datasets to improve generalization.

- Augmentation: Apply synthetic changes (e.g. flips, crops) to build invariance.

- Self-Supervision: Use unlabeled data + pretext tasks to learn transferable features.

- Goal: Robust learning across varied environments with minimal labeled data.





---
# VII. IMPLICATIONS FOR MACHINE LEARNING
## E. Reinforcement Learning


- Causal Perspectives in RL:

  - On-policy RL estimates do-probabilities directly.

  - Off-policy / batch RL faces causal inference challenges due to observational data.


--

- a. World Models
  - Model-based RL = learn causal effects of actions.

  - Generative models simulate environments for safe agent training.

  - Structured models can capture entities + causal relations.


--
- b. Generalization & Transfer
  - RL agents struggle with generalization & data efficiency.

  - Solution: learn invariant causal mechanisms to adapt to changes.

  - Exploration (interventions) helps uncover causal structure.


--

- c. Counterfactuals
  - Improve learning & decision-making by reasoning about “what could have happened.”

  - Supports data efficiency, multi-agent communication, and planning.


--

- d. Offline RL
  - Learn from fixed datasets (no interaction).

  - Requires counterfactual reasoning to infer unseen actions’ outcomes.

  - Causal modeling (e.g., invariances) can reduce distribution shift issues.





 
---
# VII. IMPLICATIONS FOR MACHINE LEARNING
## F. Scientific Applications



Goal -> Use ML to complement (not replace) scientific understanding, with causality playing a key role.

--

- Physics Simulations
  - Neural networks boost efficiency of simulators.

  - Works well in controlled environments, but needs retraining if conditions change.


--

- Healthcare & Medicine
  - Personal health models (e.g. from EHR, genetics) require causal understanding to ensure reliable treatment recommendations.

  - Training on doctors' decisions alone may fail in real-world settings.

  - Causal models improve personalized medicine and pandemic analysis (e.g. Simpson’s paradox in COVID-19).



--

- Astronomy

  - Causal models help remove measurement confounding in exoplanet detection.

  - Enabled recovery of hidden signals -> discovery of 36 candidates, 21 validated.

  - Led to finding water on exoplanet K2-18b, in the habitable zone.





---
# VII. IMPLICATIONS FOR MACHINE LEARNING
##  G. Multi-Task Learning and Continual Learning


- The Problem:
  - Current AI is narrow - strong at specific tasks, weak at generalizing across diverse environments.

- Humans excel due to:

  - Discovering high-level abstractions

  - Recognizing causal relationships

  - Adapting to out-of-distribution (OOD) settings


--

**Multi-Task Learning**

- Aim: Solve multiple tasks across varying environments.

- Core idea: Share representations across tasks.

- Causal models can help by learning shared data-generating processes with components satisfying the Sparse Mechanism Shift (SMS) hypothesis.

- Evidence shows causal models adapt faster to sparse distribution shifts.


--

- Why Not Just Scale Models?
  - Big models (e.g. large language or vision transformers) do generalize surprisingly well across interventions if:

  - Data is sufficiently diverse (which is not always testable).

  - Assumptions match the real-world dynamics.

- But:

  - Worst-case generalization errors can still be high under distribution shifts.

  - Purely scaling doesn't explicitly capture environment structure or causal factors.


--

**Why Causality Still Matters**
- Causality makes assumptions explicit and interpretable.

- The Independent Causal Mechanisms (ICM) principle supports decomposing environments into modular, reusable parts.

- Useful for:

  - Generalizing across related environments

  - Designing modular ML systems

  - Enforcing structure aligned with physical and cognitive insights


--

Causality complements deep learning. 

Combining both may be essential to build robust, general-purpose, and adaptable AI.



---

# VIII. CONCLUSION



We explored the intersection of causality and machine learning, covering:

- Fundamentals of causal inference

- Independent Causal Mechanisms (ICM) and invariance as useful inductive biases

- Learning from observational & interventional data when causal variables are known

- The open challenge of causal representation learning

- Applications to open ML problems: semi-supervised learning, domain generalization, and robustness

--

## a. Learning Non-Linear Causal Relations at Scale

- Use modern ML to model complex, scalable causal relations

- Research goals:

  - When can we reliably learn non-linear causal structure?

  - Which ML frameworks are best suited?

  - Show causal models outperform statistical ones in generalization & reuse



--

## b. Learning Causal Variables

- Current deep models output entangled vector-based representations

- Need:

  - Modular, flexible representations (e.g., variable number of objects)

  - Causal representations that adapt to the task and interventions

  - Study how and when these variables can be reliably extracted



--

## c. Understanding Biases in Deep Learning

- Current robustness gains stem from data scale, augmentation, and pre-training, but:

  - Which elements really help? Why?

  - We need a taxonomy of inductive biases aligned with causality

  - Pre-training choices should be examined for their causal generalization impact


--

## d. Causal Models for World & Agent (in RL)

- In many RL settings, no abstract state space is given

- Goal: learn causal variables from raw data (e.g., pixels)

- Needed for:

  - Causal induction in real-world tasks

  - Building causal world models that support planning, robustness, and generalization








---

### Thank you!



