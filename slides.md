# Towards causal Representation Learning

 Bernhard Schölkopf , Francesco Locatello , Stefan Bauer , Nan Rosemary Ke , Nal Kalchbrenner
 Anirudh Goyal, Yoshua Bengio

Sofia Zervaki, 2025

---
# I. Introduction

- ML struggles with strong generalization.

- ML systems often ignore:
  - Interventions
  - Domain shifts
  - Temporal structure

- Most ML success comes from recognizing patterns in i.i.d. data.

The paper outlines some key ML challenges where causality matters.

--
# Issue 1: Robustness

- Deep learning models often fail under small changes in input data.
- Benchmarks evaluate models under such shifts, with solutions like
  - Data augmentation
  - Pre-training
  - Self-supervision
  - Architectures with inductive biases

- The authors argue that those fixes may not be sufficient 
- To generalize beyond the i.i.d setting we need **Causal Models** which go beyond just spotting patterns


--

# Issue 2: Learning Reusable Mechanisms

- Humans learn intuitively

- This lets them reuse knowledge to learn new tasks quickly

- For ML Systems, Agents should build modular representations of the world

  - Each module = a causal mechanism (like gravity, friction, etc.)

As a result 

- In new tasks/environments only some modules need updating

- Most knowledge can be reused without retraining


--

# A Causality Perspective: 

- Conditional probabilities don’t explain what happens when we intervene

- Intervention
  - Causality requires reasoning about actions and changes
  - Goes beyond what’s observed and includes
    - Hypothetical scenarios
    - Counterfactuals
    - Deliberate reasoning
    
---

#  II. LEVELS OF CAUSAL MODELING

## Physical Models

Gold standard: A set of differential equations describe how physical systems evolve over time.

These equations:  - Predict future behavior
                  - Describe interventions and causal structure
                  - Provide physical insight


Example:

$ \frac{dx}{dt} = f(x)$, with $x \in R^d$ and initial value $x(t_0)=x_0$

By Picard–Lindelöf, this has a unique solution if $f$ is Lipschitz


--
# Summary of different modeling approaches


| Model Type           | Predict in i.i.d. | Predict under shift/intervention | Answer counterfactuals | Obtain physical insight | Learn from data |
|----------------------|------------------|----------------------------------|-------------------------|--------------------------|------------------|
| Mechanistic/Physical | Yes              | Yes                              | Yes                     | Yes                      | ?                |
| Structural Causal    | Yes              | Yes                              | Yes                     | ?                        | ?                |
| Causal Graphical     | Yes              | Yes                              | No                      | ?                        | ?                |
| Statistical          | Yes              | No                               | No                      | No                       | Yes              |



--

#  A. Predicting in the i.i.d. setting

- Statistical models are trained to approximate probabilities 

- For a given set of input examples X and target labels Y, we want approximating $P(Y|X)$ 

- This works well for standard prediction tasks, but 

**Accurate predictions is not equal to causal understanding**


- they can fail when the distribution of the data changes (intervention, real-world actions etc.)



--

#  B. Predicting Under Distribution Shifts

- Interventions change variable values or their relationships, violating i.i.d. assumptions.

- Classical ML models often fail when deployed in real-world settings where distributions shift

- Causal models help build systems that remain accurate under such changes.

- Robust prediction requires more than test set accuracy. We need to trust that
    - **the predictions of the algorithm will remain valid if the conditions change**



--
#  C. Answering Counterfactual Questions

- Counterfactuals imagine what could have happened if actions were different.

- Harder than interventional questions but important for intelligent reasoning.

- Example:
  - "Would a patient have avoided heart failure if they exercised earlier?"

- Important for AI to reflect on past decisions and learn from alternatives.

- Critical in reinforcement learning for hypothesis testing and improving policies.


--

#  D. Nature of Data: Observational, Interventional, (Un)structured

- Data types matter for causal inference:
  - **Observational** (i.i.d. or with unknown shifts) vs. **Interventional** (known changes)
  - **Hand-engineered** (structured features) vs. **Raw** (images, audio)

- Statistical models work with observational/raw data but can't reveal causality

- Causal learning often needs
  - Data from multiple environments or known interventions
  - Assumptions like causal sufficiency

- Future direction
  - Replace expert-designed inputs with inductive biases, meta-learning, and self-supervision.


---

#  III. CAUSAL MODELS AND INFERENCE

- This section explores the difference between statistical and causal models.

- It also introduces a formal framework for reasoning about interventions and distribution shifts.

--

# A. Methods driven by i.i.d. data

- Most ML successes rely on

  - Large labeled datasets (human-made or simulated)

  - High-capacity models (neural nets etc.)

  - Powerful computing resources.

  - **i.i.d. assumption**

--

- Limitations of i.i.d.
  - Fails under distribution shifts:
    - Different real-world conditions ( different clinics or countries)
    - Adversarial examples break models with small image changes
    - Models confuse cause and effect (e.g., recommending a laptop after a laptop bag)
  - i.i.d. systems lack causal understanding

    - Cannot reason about interventions
    -Struggle in dynamic or changing environments


--

# B. The Reichenbach Principle: From Statistics to Causality


<div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px; background-color: #f9f9f9;">
  <strong>Reichenbach’s Common Cause Principle::</strong>If X and Y are statistically dependent, there's a variable Z causing both, such that X ⟶ Z ⟶ Y explains the dependence.
</div>


- Limitation of Observational Data:

  - Can't distinguish between causal structures (X ⟶ Y vs Y ⟶ X vs Z ⟶ both)

  - All produce the same observed dependence

- More Variables Help:
  - Conditional independence between variables can reveal causal direction

  - Leads to causal graphs and structural causal models (SCMs)

--

#  C. Structural causal models (SCMs)

- **SCM Definition**: Each variable $ X_i := f_i (PA_i, U_i) $
  - $PA_i$ is the parent variables (causes)
  - $U_i$ is the independent noise (captures randomness)
  - It is represented as a Directed Acyclic Graph (DAG)

--
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

  -Modularity: Each causal mechanism operates independently.

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
**Algorithmic View of ICM**

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

--

**Modern Approaches**

- Neural networks: Learn causal graphs via continuous optimization.

- Reinforcement learning: Agents explore to discover causality.

- Causal models are expected to generalize better under changes than purely predictive ones.



---



# VI. LEARNING CAUSAL VARIABLES


--




<img src="figures/Fig2.png" alt="Histogram" width="1000">


--



<img src="figures/Fig3.png" alt="Histogram" width="600">



---

# VII. IMPLICATIONS FOR MACHINE LEARNING

--

#  A. Semi-Supervised Learning (SSL)

--

# B. Adversarial Vulnerability


--

# C. Robustness and Strong Generalization


--

# D. Pre-training, Data Augmentation, and Self-Supervision

--

 # E. Reinforcement Learning

--

# F. Scientific Applications

--

#  G. Multi-Task Learning and Continual Learning




---

# VIII. CONCLUSION



























---
# Inline Math

This is Einstein's formula: $E = mc^2$



---

# Block Math

$$
\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
$$

---

# Vertical slides

This is the parent of vertical slides.

--
## Vertical slide 1

This is a vertical slide under the parent slide.

--
## Vertical slide 2

Another vertical slide under the parent slide.

---

# Add figures

Add a figure with Markdown code

```markdown
    ![Histogram of the solution of a bistable ODE](figures/demo.png)
```

![Histogram](figures/demo.png)

--

or with HTML code for more control

```html
<img src="figures/demo.png" alt="Histogram" width="400">
```

<img src="figures/demo.png" alt="Histogram" width="400">

--

or with percentage

```html
<img src="figures/demo.png" alt="Histogram" style="width:40%">
```

<img src="figures/demo.png" alt="Histogram" style="width:40%">

--

You can add a caption like this
```html
<figure>
  <img src="figures/demo.png" alt="Time series" style="width:70%">
  <figcaption>Figure 1: Histogram of the solution of a bistable ODE</figcaption>
</figure>
```

<figure>
  <img src="figures/demo.png" alt="Time series" style="width:70%">
  <figcaption>Figure 1: Histogram of the solution of a bistable ODE</figcaption>
</figure>

---

# Show a video

```html
<video src="media/video.mp4" autoplay muted loop style="width: 60%"></video>
```

<video src="media/video.mp4" autoplay muted loop style="width: 60%"></video>


---

# Code blocks

<pre><code class="language-python" data-trim>
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
</code></pre>


--

# Code blocks with highlighting

<pre><code class="language-python" data-trim data-line-numbers="3,5-6,10">
import numpy as np
import matplotlib.pyplot as plt

def simulate_ode(f, y0, t):
    """Simple forward Euler ODE solver."""
    y = np.zeros_like(t)
    y[0] = y0
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        y[i] = y[i-1] + dt * f(t[i-1], y[i-1])
    return y

# Example usage
f = lambda t, y: -0.5 * y
t = np.linspace(0, 10, 100)
y = simulate_ode(f, 1.0, t)

plt.plot(t, y)
plt.title("Exponential Decay")
plt.xlabel("Time")
plt.ylabel("y(t)")
plt.grid()
plt.show()
</code></pre>


--

<section>
  <h3>Code blocks with animations</h3>

  <div class="fragment">
    <pre><code class="language-python" data-trim data-line-numbers>
import numpy as np
import matplotlib.pyplot as plt
    </code></pre>
  </div>

  <div class="fragment">
    <pre><code class="language-python" data-trim data-line-numbers>
def simulate_ode(f, y0, t):
    """Simple forward Euler ODE solver."""
    y = np.zeros_like(t)
    y[0] = y0
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        y[i] = y[i-1] + dt * f(t[i-1], y[i-1])
    return y
    </code></pre>
  </div>

  <div class="fragment">
    <pre><code class="language-python" data-trim data-line-numbers>
f = lambda t, y: -0.5 * y
t = np.linspace(0, 10, 100)
y = simulate_ode(f, 1.0, t)
    </code></pre>
  </div>

  <div class="fragment">
    <pre><code class="language-python" data-trim data-line-numbers>
plt.plot(t, y)
plt.title("Exponential Decay")
plt.xlabel("Time")
plt.ylabel("y(t)")
plt.grid()
plt.show()
    </code></pre>
  </div>
</section>



---

### 🦧 That is all 🦧



