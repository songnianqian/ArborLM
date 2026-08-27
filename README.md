# ArborLM

### A Context-Indexed, HyperNetwork-Compressed, Branching Language Model

**ArborLM scales language-model capacity by branching rather than by continuously enlarging one monolithic network.**

![ArborLM Architecture](https://github.com/songnianqian/ArborLM/raw/main/MultiplePath.png)

A shared Transformer trunk learns broad language and common computation. A separate **Context Index** routes each input to a specialist branch. Inside the model, large conventional dense FFNs are replaced by a much smaller combination of **heterogeneous experts and shared HyperNetworks**.

As branches become more specialized, ArborLM can also reduce their hidden width. Because many Transformer parameter terms scale approximately with the square of hidden width, progressively narrower branches provide another strong reduction in model size.

The current implementation demonstrates these ideas at GPT-2 scale.

```text
                     Context Index
                          │
                          ▼
                  Shared d=768 trunk
                     8 layers
                          │
          ┌───────────────┼───────────────┐
          │               │               │
         P0              P1              ...
          │               │
      768 → 576       768 → 576
          │               │
      4 layers         4 layers
      HyperNet         HyperNet
      experts          experts
          │               │
      576 → 768       576 → 768
          │               │
          └────── selected path ──────┘
                          │
                    shared LM head
```

Only **one branch is executed for an input**, so adding specialist branches increases total stored capacity without requiring active inference capacity to grow at the same rate.

---

## Current Results

The single-path precursor to ArborLM is called **MultiExpertsHyper**. It combines heterogeneous compact experts with a shared HyperNetwork.

| Model | Training Step | PPL | Next-Token Accuracy |
|---|---:|---:|---:|
| MultiExpertsHyper | 8,200 | 36.96 | 40.04% |
| MultiExpertsHyper | 9,200 | 34.90 | 40.80% |
| MultiExpertsHyper | 9,800 | 34.19 | 40.89% |
| **MultiExpertsHyper** | **10,000** | **33.87** | **40.87%** |

The model was trained on `DKYoon/SlimPajama-6B` with 1024-token sequences.

The MultiExpertsHyper FFN subsystem is roughly **15× smaller than the conventional Dense/Baseline FFN design it replaces**, while maintaining strong language-model performance.

The six-path ArborLM model is currently undergoing long-run training.

---

# 1. Why ArborLM?

Most Transformer scaling follows one basic pattern:

> Make the same model deeper, wider, or both.

This increases capability, but every request still carries the cost of the increasingly large monolithic network.

ArborLM explores a different scaling direction.

It separates the model into:

1. a **shared trunk** for broadly useful computation;
2. **specialist branches** for narrower distributions of knowledge and behavior;
3. a **Context Index** that decides which branch should process a sequence.

The total model capacity is approximately

$$
P_{\text{total}} = P_{\text{trunk}} + \sum_{i=1}^{N} P_{\text{branch},i},
$$

while the active model for one routed inference is approximately

$$
P_{\text{active}} = P_{\text{trunk}} + P_{\text{selected branch}}.
$$

Therefore, adding branches can increase specialized stored capacity without increasing active inference parameters proportionally.

This is the basic reason for the name **ArborLM**: the model can grow like a tree rather than only by enlarging a single trunk.

---

# 2. Dense/Baseline FFN as a Black Box

A standard Transformer block contains a residual transformation of the form

$$
x' = x + F(x).
$$

The rest of the Transformer only sees the input and output of $F$:

$$
F:\mathbb{R}^{d}\rightarrow\mathbb{R}^{d}.
$$

A conventional Dense/Baseline Transformer usually implements $F$ as a large dense MLP:

$$
F(x)=W_2\,\sigma(W_1x),
$$

with the familiar expansion

$$
d\rightarrow4d\rightarrow d.
$$

Ignoring biases, this requires approximately

$$
8d^2
$$

weights for each FFN.

But the Transformer architecture does **not** require the internal function $F$ to have this particular form.

The FFN can instead be treated as a black-box function approximator:

```text
hidden state x
      │
      ▼
┌───────────────────────┐
│       F(x)            │
│                       │
│ internal architecture │
│ is replaceable        │
└───────────────────────┘
      │
      ▼
hidden state y
```

The relevant question is empirical:

> Can a much smaller structured function family learn a sufficiently useful transformation $F(x)$?

The MultiExpertsHyper experiments suggest that it can.

---

# 3. MultiExpertsHyper: Heterogeneous Compact Experts

A single aggressively compressed FFN may have too limited a function family.

Instead, MultiExpertsHyper uses several **small experts with different internal structures**.

The current expert family is:

```text
perc2
perc4
perc4
reglu1d
film
```

Each expert is itself a complete learned transformation from the Transformer hidden state back to the same hidden space:

$$
F_i:\mathbb{R}^{E}\rightarrow\mathbb{R}^{E}.
$$

Here, $E$ is the Transformer hidden width. The internal representation of an expert can be extremely small even though its input and output both remain $E$-dimensional.

## 3.1 Exact Example: `ExpertC_RankN`

For an input hidden vector

$$
x\in\mathbb{R}^{E},
$$

`ExpertC_RankN` first projects the hidden state into an $N$-dimensional latent space:

$$
s = Wx+b,
$$

where

$$
W\in\mathbb{R}^{N\times E},
\qquad
b\in\mathbb{R}^{N}.
$$

The latent state is passed through GELU and scaled elementwise by a learned vector $\alpha$:

$$
g=\mathrm{GELU}(s)\odot\alpha,
\qquad
\alpha\in\mathbb{R}^{N}.
$$

It is then projected back to the Transformer hidden width:

$$
y=V^{\top}g+b_{\text{out}},
$$

with

$$
V\in\mathbb{R}^{N\times E},
\qquad
b_{\text{out}}\in\mathbb{R}^{E}.
$$

Therefore the complete expert is the nonlinear map

$$
\boxed{\,F_N(x) = V^{\top}\left[\mathrm{GELU}(Wx+b)\odot\alpha\right] + b_{\text{out}}\,}
$$

with

$$
F_N:\mathbb{R}^{E}\rightarrow\mathbb{R}^{E}.
$$

For `ExpertC_Rank4`, $N=4$:

$$
x\in\mathbb{R}^{E}
\rightarrow
s\in\mathbb{R}^{4}
\rightarrow
g\in\mathbb{R}^{4}
\rightarrow
y\in\mathbb{R}^{E}.
$$

Thus the expert learns a full hidden-state-to-hidden-state function even though the nonlinear bottleneck contains only four latent values.

If all parameters shown above are trainable, the parameter count is

$$
P_{\text{RankN}}
=NE+N+N+NE+E
=2NE+2N+E.
$$

For $N=4$,

$$
P_{\text{Rank4}}=9E+8.
$$

At $E=768$, this is

$$
P_{\text{Rank4}}
=9(768)+8
=6,920
$$

parameters.

For comparison, a conventional Dense/Baseline GPT-2-style FFN uses the expansion

$$
E\rightarrow4E\rightarrow E,
$$

whose two weight matrices alone contain

$$
8E^2
$$

weights. At $E=768$, that is

$$
8(768)^2=4,718,592
$$

weights before biases.

## 3.2 Soft Mixing Creates a Richer Effective Function

Every expert produces its own learned mapping

$$
y_i=F_i(x).
$$

The soft gate produces input-dependent mixture weights. If the gate logits are $a_i(x)$, then

$$
p_i(x) = \frac{\exp(a_i(x))}{\sum_j \exp(a_j(x))}.
$$

The combined MultiExpertsHyper transformation is

$$
\boxed{\,F_{\text{mix}}(x) = \sum_{i=1}^{M} p_i(x)\,F_i(x)\,}
$$

where $M$ is the number of experts.

This is not simply one fixed low-rank matrix. Both the expert outputs and their mixture weights depend on the input:

$$
p_i=p_i(x),
\qquad
F_i=F_i(x).
$$

For a collection of rank-style experts, the effective function can be written explicitly as

$$
F_{\text{mix}}(x) = \sum_{i=1}^{M} p_i(x) \left[ V_i^{\top} \left( \mathrm{GELU}(W_i x + b_i) \odot \alpha_i \right) + b_{\text{out},i} \right].
$$

MultiExpertsHyper goes further by using **heterogeneous** expert structures rather than only copies of one rank-$N$ form. The result is an input-conditioned soft combination of several different nonlinear functions.

**Each compact expert is a complete learned map**

$$
F_i:\mathbb{R}^{E}\rightarrow\mathbb{R}^{E}.
$$

**The soft mixture then constructs an input-dependent effective map**

$$
F_{\text{mix}}(x)=\sum_i p_i(x)\,F_i(x),
$$

**allowing a much richer function family than any one tiny expert alone.**

The important difference from a conventional large MoE is that MultiExpertsHyper is not simply making many copies of the same large FFN.

The experts are deliberately **small and structurally heterogeneous**.

The working hypothesis is that

$$
\boxed{\ \text{small } F_i \;+\; \text{heterogeneous } F_i \;+\; \text{soft mixing} \;+\; \text{HyperNet conditioning} \;\Rightarrow\; \text{a rich effective } F\ }
$$

while using far fewer FFN parameters than a conventional dense design.

Importantly, the mature MultiExpertsHyper model does not collapse onto one expert. All expert types remain active with substantial gate mass.

---

# 4. HyperNet: Reusing Computation Across Depth

![Hypernet Architecture](https://github.com/songnianqian/ArborLM/raw/main/hypernet_injection.svg)

The second major component is the **HyperNetwork**.

A HyperNet adds a small input-conditioned controller:

$$
H(h).
$$

It observes the current hidden representation and produces a learned signal that influences the main layer's computation.

The key ArborLM idea is that the **same HyperNet can be shared across several layers**.

Consider four successive layers:

$$
h_1=F_1(h_0,H(h_0))
$$

$$
h_2=F_2(h_1,H(h_1))
$$

$$
h_3=F_3(h_2,H(h_2))
$$

$$
h_4=F_4(h_3,H(h_3)).
$$

The parameters of $H$ are shared, but its input is not.

Because the hidden state changes at every layer, the same HyperNet can produce different input-conditioned outputs at different depths.

## Depth-wise Parameter Reuse

```text
Layer 1 ─┐
Layer 2 ─┤
Layer 3 ─┼── shared HyperNet
Layer 4 ─┘
```

This may help explain why a large reduction in FFN parameters does not produce a proportional loss in model quality.

That interpretation is currently a design hypothesis rather than a completed component-level proof.

---

# 5. Learned Centroids as a Coordinate System

The HyperNet uses a learned centroid bank to describe the current hidden state relative to learned reference vectors.

A small number of centroids does **not** imply that the model has only that many possible concepts.

If the controller computes relationships

$$
s_i=\mathrm{sim}(h,c_i),
$$

then the complete vector

$$
s(h)=[s_1,s_2,\ldots,s_M]
$$

describes where the current representation lies relative to all learned references.

A useful analogy is to think of the centroids as **lighthouses**.

A few lighthouses do not restrict a ship to a few locations. Their relative signals provide a coordinate system from which many positions can be distinguished.

The current implementation also incorporates phase-derived information from the hidden state, giving the controller richer information than a simple nearest-centroid lookup.

---

# 6. Training Data

ArborLM separates the data used to train the language model from the fixed corpus used to construct the Context Index.

## Language-Model Training

Current MultiExpertsHyper and ArborLM experiments use:

```text
Dataset               DKYoon/SlimPajama-6B
Mode                  streaming
Tokenizer             GPT-2
Sequence length       1024
Micro batch           2
Gradient accumulation 32
Effective batch       64 sequences
Tokens / step         ≈65,536
Base learning rate    3e-4
Warmup                500 steps
```

The completed 10,000-step MultiExpertsHyper run therefore processed roughly **655 million language-model tokens**.

## Context Index Corpus

The current Context Index uses a frozen, source-balanced local corpus:

| Source | Sequences |
|---|---:|
| ArXiv | 3,000 |
| Book | 3,000 |
| C4 | 3,000 |
| CommonCrawl | 3,000 |
| GitHub | 3,000 |
| StackExchange | 3,000 |
| Wikipedia | 3,000 |
| **Total** | **21,000** |

Equal source sampling prevents large raw sources from dominating the initial index merely because more data are available.

The seven data sources are **not seven predefined model paths**. They provide a balanced and diverse corpus from which the initial semantic partition is learned.

---

# 7. Context Index: Routing Outside the Language Model

ArborLM does not use the LM itself as a token-by-token branch router.

Instead it has a separate **Context Index**.

```text
prompt / document prefix
          │
          ▼
     context encoder
          │
          ▼
    semantic embedding
          │
          ▼
      Context Index
          │
          ▼
    ranked branches
   P3, P5, P1, ...
```

**Language-model gradients do not backpropagate through the Context Index.**

The current implementation uses **top-1 routing** for ordinary multipath training and inference.

---

# 8. Building the Initial Context Index

```text
7 balanced sources
       │
       ▼
21,000 fixed sequences
       │
       ▼
MiniLM encoder
       │
       ▼
mean pooling
       │
       ▼
balanced k-means
       │
       ▼
6 semantic regions
       │
       ▼
P0 P1 P2 P3 P4 P5
```

The current model uses $K=6$.

Source-balanced data are used during construction so the initial clustering is driven by semantic structure rather than source volume.

The paths are therefore **not simply source IDs**.

---

# 9. PPL Feedback: Letting the LM Improve Its Own Index

The initial semantic Context Index answers:

> *Which contexts look similar?*

But once specialist paths have trained, the language model can answer a different question:

> *Which path actually models this context best?*

For an input $x$, candidate paths can be evaluated by negative log-likelihood:

$$
L_i(x) = -\frac{1}{T} \sum_t \log p_i(x_t \mid x_{\lt t}).
$$

The corresponding perplexity is

$$
PPL_i(x)=e^{L_i(x)}.
$$

This gives an LM-derived preference over paths.

```text
semantic Context Index
          │
          ▼
     initial routing
          │
          ▼
       LM training
          │
          ▼
 evaluate path PPL
          │
          ▼
    path preferences
          │
          ▼
    refit the Index
```

Conceptually:

$$
\boxed{\ \text{Index} \rightarrow \text{LM} \rightarrow \text{PPL feedback} \rightarrow \text{Index refit}\ }
$$

The refitted index is then frozen again while LM training continues.

In the current PPL-refitted index, the approximate fitted routing distribution is:

```text
P0 ≈ 11%
P1 ≈ 10%
P2 ≈  6%
P3 ≈ 18%
P4 ≈  7%
P5 ≈ 47%
```

This imbalance is not automatically an error. ArborLM does not require equal path traffic if the learned language distribution naturally assigns more contexts to one region than another.

---

# 10. Multiple Prefix Views

Routing should not depend on one artificially fixed prefix length.

During Context Index construction and PPL feedback, ArborLM uses multiple prefix lengths, including short and longer views spanning roughly 32 to 256 tokens.

---

# 11. Multiple Specialist Paths

The current ArborLM model starts from a mature MultiExpertsHyper model.

Its shared components are frozen:

- embeddings;
- 8-layer d768 Transformer trunk;
- trunk HyperNet;
- final normalization;
- LM output head.

Six new independent specialist paths are then created.

Each path uses:

```text
h_trunk @ d768
      │
      ▼
   768 → 576
      │
      ▼
4 Transformer blocks @ d576
      │
      ├── compact heterogeneous experts
      └── path HyperNet
      │
      ▼
   576 → 768
      │
      ▼
shared final LN + LM head
```

Only one selected path is executed for an input.

Active depth remains 12 Transformer layers.

For the present implementation:

```text
shared frozen base                ≈64.02M parameters
one narrow d576 path              ≈ 8.80M parameters
all six narrow paths              ≈52.78M parameters
active inference: trunk + 1 path  ≈72.82M parameters
```

---

# 12. Specialization Enables Width Reduction

A specialist branch models a narrower conditional distribution than the shared trunk.

ArborLM therefore hypothesizes that progressively specialized branches can often operate with progressively smaller hidden widths.

Many Transformer weight matrices scale approximately as

$$
P\propto d^2.
$$

If a branch width is reduced from $d$ to $rd$, many parameter terms fall approximately as

$$
r^2.
$$

The current ArborLM model already applies the first level of this idea:

$$
768\rightarrow576.
$$

Thus,

$$
\left(\frac{576}{768}\right)^2=0.5625.
$$

Width-squared components at d576 require only about **56%** of the parameters of equivalent d768 components.

Approximately **44% is removed simply by narrowing the branch**, before accounting for the compact MultiExpertsHyper FFN architecture.

| Hidden Width | Approx. d² Cost vs. d768 |
|---:|---:|
| 768 | 100% |
| 640 | 69.4% |
| 576 | 56.3% |
| 512 | 44.4% |
| 384 | 25.0% |
| 256 | 11.1% |

---

# 13. Tree Scaling

The current implementation has one branching level, but the same architecture can naturally extend to a hierarchy:

```text
                       Root
                        │
             ┌──────────┼──────────┐
             │          │          │
           Code       Science     Prose
             │          │
          ┌──┴──┐    ┌──┴──┐
        Python C++ Physics Biology
```

Deeper specialization may also be accompanied by decreasing hidden width.

Tree growth can therefore combine two effects:

1. **inactive sibling branches are not executed**;
2. **deeper specialist branches may themselves be smaller**.

---

# 14. Three Scaling Directions

## Local Compression

```text
Dense/Baseline FFN
       ↓
heterogeneous compact experts
```

## Vertical Reuse

```text
many layers
   ↓
shared HyperNet
```

## Horizontal Scaling

```text
shared trunk
   ↓
Context-Indexed tree
   ↓
specialized, progressively narrower branches
```

> **ArborLM compresses computation vertically through shared HyperNetworks and compact heterogeneous experts, then scales capacity horizontally through Context-Indexed specialist branches.**

---

# 15. Knowledge Distillation

The narrow specialist paths begin from fresh parameters while the shared trunk is already mature.

During early multipath training, ArborLM can use the mature MultiExpertsHyper specialist as a frozen teacher.

The teacher is used only during training and is **not part of ArborLM inference**.

Knowledge distillation is a training mechanism rather than a central architectural contribution of this project.

---

# 16. Dense/Baseline Comparison

A controlled Dense/Baseline model was trained using the same training pipeline.

| Step | Dense/Baseline PPL | MultiExpertsHyper PPL |
|---:|---:|---:|
| 200 | ~1327 | ~1695 |
| 400 | ~726 | ~737 |
| 600 | ~554 | ~536 |
| 2000 | ~232 | ~216 |

The Dense/Baseline model learned faster initially, while MultiExpertsHyper caught up and moved ahead during the early training run.

The Dense/Baseline model was not trained to the same 10,000-step horizon because of available compute, so no claim is made about final asymptotic superiority.

---

# 17. Future Context Intelligence

The current Context Index is simple and frozen during normal LM training.

A future ArborLM system can extend it into a lightweight intelligence layer that manages uncertainty, conversation state, and future branch selection.

## 17.1 Small Feedback LM for Low-Confidence Routing

At the beginning of a conversation, there may not be enough information for the Context Index to make a confident routing decision.

A future ArborLM system can use a small auxiliary language model only when Context Index confidence is low.

Possible confidence measures include the top-two score margin

$$
C=s_1-s_2
$$

or routing entropy

$$
H=-\sum_i p_i\log p_i.
$$

A possible cascade is:

```text
FAST
Context Index confident
        ↓
select branch

UNCERTAIN
Context Index uncertain
        ↓
small feedback LM
        ↓
select likely branch

VERY UNCERTAIN
Index and feedback LM disagree
        ↓
brief candidate-path PPL comparison
        ↓
select lower-PPL branch
```

---

# 18. Stateful Conversation Routing

A conversation has state.

A future smart Context Index can track:

```text
current branch
current confidence
conversation-topic representation
recent branch history
candidate alternate branch
likely next topic
```

Instead of recomputing routing independently on every turn, the Context Index can operate as a lightweight conversation state machine.

```text
START
  │
  ▼
DISCOVER
  │
  ▼
COMMITTED P3
  │
  ├── confidence stable → stay P3
  │
  └── topic shift → VERIFY → stay or switch
```

This can prevent route thrashing near branch boundaries.

---

# 19. Predicting the Next Topic

A smart Context Index can eventually predict what topic is likely to come next.

```text
current state = PyTorch
        │
        ▼
Context Index state model
        │
        ├── 0.55 CUDA
        ├── 0.30 training
        └── 0.10 data
```

If branches eventually live on different GPUs, CPU memory, or storage, likely next branches could be **pre-warmed or preloaded before they are requested**.

The Context Index therefore becomes not only a classifier but also a **predictive scheduler**.

Future routing can combine:

$$
\text{semantic context} + \text{conversation history} + \text{branch PPL feedback}
$$

---

# 20. Can ArborLM Scale to Large Models?

The current implementation is a GPT-2-scale research system.

Whether ArborLM scales to much larger language models remains an empirical question, but the architecture has several properties that become increasingly interesting as hidden width grows.

Many Transformer parameter terms scale approximately as $d^2$.

A conceptual large ArborLM might taper width down the tree:

```text
                    broad trunk
                     d = 8192
                         │
       ┌─────────────────┼─────────────────┐
       │                 │                 │
    science            code             language
    d = 6144          d = 6144          d = 6144
       │
   ┌───┴────┐
   │        │
physics   biology
d = 4096  d = 4096
```

These widths are examples, not proposed production settings.

The deeper hypothesis is:

> As the represented distribution becomes narrower through specialization, the minimum useful hidden width may also decrease.

Important open questions include:

- Does a large shared trunk become the dominant unavoidable cost?
- How narrow can a specialist become without losing general capability inherited from the trunk?
- How costly do routing errors become as branches differentiate more strongly?
- How much training data is required to support each specialist?
- At what point is another branch level more efficient than increasing an existing branch's width or depth?

---

# 21. Current Limitations

- The Dense/Baseline model has not been trained to the same maturity as MultiExpertsHyper.
- The exact contribution of HyperNet versus heterogeneous experts has not yet been isolated with a full long-run ablation matrix.
- Free autoregressive generation can still become repetitive even when teacher-forced perplexity is strong.
- The optimal timing and frequency of Context Index PPL refits remain research questions.
- Multi-level hierarchical branching has not yet been experimentally tested.
- ArborLM has not yet been validated at large-model scale.

---

# 22. Research Questions

- How closely can d576 specialist branches recover the quality of the mature d768 MultiExpertsHyper continuation?
- How much total specialized capacity can be added while keeping active inference size approximately constant?
- How far can hidden width decrease as branch specialization increases?
- Does sharing a HyperNet across multiple layers explain part of the observed FFN parameter efficiency?
- How much does heterogeneous expert structure contribute beyond a single compact expert?
- Can PPL-refitted Context Index routing significantly close the gap between routed and oracle path selection?
- Can a lightweight feedback LM resolve low-confidence routing efficiently?
- Can conversation state and next-topic prediction improve routing stability and branch preloading?
- At what point does adding another tree level become more efficient than increasing the width or depth of an existing branch?

---

# 23. Project Status

```text
✓ Dense/Baseline reference implementation
✓ MultiExpertsHyper compact FFN
✓ Heterogeneous expert mixture
✓ Shared HyperNet
✓ Mature 10,000-step MultiExpertsHyper training
✓ Context Index
✓ Source-balanced Context Index corpus
✓ Multi-prefix Context Index training
✓ PPL-feedback / refit mechanism
✓ Six-path d576 ArborLM conversion
✓ Frozen mature teacher
✓ Knowledge-distillation training
✓ 100-step multipath/KD validation
→ Long-run six-path training in progress
```

---

# 24. Repository Structure

```text
config.py
    Experiment and architecture configuration

model.py
    Shared trunk and core Transformer implementation

multipath_model.py
    ArborLM branch construction and multipath forward path

clustered_hypernet.py
    HyperNetwork and centroid mechanisms

content_index.py
    Context Index representation and routing

build_content_index.py
    Initial index construction and PPL-feedback refitting

train.py
    Language-model training

eval.py
    Perplexity and next-token evaluation

diagnostics.py
    Path, expert, HyperNet, routing, and prompt diagnostics

checkpoint.py
    Model checkpoint management
```

---

# 25. Name

**ArborLM**

*Arbor* means tree.

The name reflects the central design:

> Keep broadly useful computation in a strong shared trunk, then grow specialized capacity as branches rather than continuously enlarging one monolithic model.

---

## Short Version

ArborLM combines four core ideas:

1. **Treat the Dense/Baseline FFN as a replaceable black-box d→d transformation.**
2. **Use MultiExpertsHyper — small heterogeneous experts plus a shared HyperNet — instead of very large independent dense FFNs.**
3. **Use an external Context Index, improved through LM perplexity feedback, to route contexts to specialists.**
4. **Scale specialized capacity through progressively narrower tree branches while executing only one branch per inference.**

The broader research question is:

> **Can language-model capacity scale by organizing computation and knowledge into reusable shared structure and conditional branches, rather than requiring every parameter to participate in every request?**
