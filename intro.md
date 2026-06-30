# Preface

:::{admonition} ⚠️ Work in progress
:class: warning
These materials are a **living draft** — actively being written, revised, and expanded from my lecture transcripts and course notebooks. Expect rough edges, gaps, and changes between visits. This is a teaching companion, **not a final or official reference**. Spot something off? That's expected — it's a work in progress!
:::

Welcome to **Advanced Deep Learning**, the book edition of **OPIM 5516** at the University of Connecticut.

This is the sequel course. In OPIM 5509 you learned to build neural networks that make a prediction; here you learn to build networks that make a prediction **and tell you how much to trust it**, that reason over **relationships** instead of flat tables, and that wield the architecture behind modern AI — the **transformer** — across time series, text, images, and video. Three big ideas, three modules, taught the way I actually teach: casual in voice, serious in content, and relentlessly hands-on in **PyTorch and Keras**.

## Who this book is for

You've finished OPIM 5509 (or its equivalent). You can build a dense network, a ConvNet, and an RNN in Keras; you know what an embedding is; you can read a learning curve and an error metric without flinching. If that's you, you're ready. If it isn't, spend a few weeks on the fundamentals first — advanced deep learning without the basics is like advanced calculus without algebra.

## How the book is organized

The book follows the three modules of the course:

```{tableofcontents}
```

- **Module 1 — Bayesian(ish) Neural Networks.** A model that says "**\$420,000**" is less useful than one that says "**\$420,000 ± \$30,000**, and I'm *unusually* unsure about this one." We build that: the Functional API, SHAP for explainability, **MC Dropout** for *epistemic* (model) uncertainty, and a two-headed network for *aleatoric* (data) uncertainty — then check our intervals with coverage.
- **Module 2 — Graph Neural Networks.** Most interesting data isn't a flat table; it's a graph. We start with collaborative filtering and recommender systems (the gentle on-ramp), learn how **message passing and smoothing** turn a graph into features, and build **GCNs and GraphSAGE** in PyTorch.
- **Module 3 — Transformers.** The architecture that ate machine learning. We motivate it against the LSTM on time series, open the hood on **attention** and **parameter counting**, then apply transformers to text (a from-scratch GPT), images (the **Vision Transformer**), and video (spatiotemporal weather and energy forecasting).

## A note on uncertainty and explainability

If there's a single thread running through this entire course, it's this: **a prediction without a measure of trust is half a prediction.** SHAP and uncertainty quantification aren't a Module 1 topic you leave behind — they come back in the transformer module, where we stress-test a forecaster by perturbing the temperature ±10°F and watch the attention and the uncertainty respond. Keep that thread in mind throughout.

Let's build.
