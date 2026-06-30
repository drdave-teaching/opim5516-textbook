# Module 1 — Bayesian(ish) Neural Networks

A standard neural network is a confident liar. Ask it for the price of a house and it returns a single number — **\$420,000** — with exactly the same swagger whether the house is a textbook three-bedroom colonial or a one-of-a-kind oddity it has never seen anything like. In this module we fix that. We build networks that return a **prediction and a measure of how much to trust it**, and we learn to read which features drive both the prediction and the doubt.

This is also where the course's two recurring tools first appear: the **Keras Functional API** (because the models here have more than one output, so the simple `Sequential` stack won't do) and **SHAP** (because once a model is making consequential decisions, "trust me" is not an acceptable explanation).

## 1.0 — Webscraping and GCP (the data behind the project)

The module's project predicts used-car prices, and the data comes from *you*: a Craigslist scraper deployed on Google Cloud Platform via GitHub Actions, with a GenAI ETL step that turns messy listing text into clean structured rows. This is reused from OPIM 5512 — implement the scraper, let it run on a schedule, and you'll have a live, growing dataset of prices to model. We don't re-teach the deployment here; we use its output.

## 1.1 — Simple DNNs with SHAP

We "walk before we run." Start with an ordinary dense network for regression, but build it with the **Functional API** so that adding a second output head later is a one-line change rather than a rewrite.

Then we explain it. **SHAP** (SHapley Additive exPlanations) assigns each feature a value quantifying its contribution to a particular prediction, grounded in cooperative game theory. The three plots you'll live in:

- the **beeswarm**, a global summary of which features matter and in which direction;
- the **dependence plot**, showing how one feature's effect changes across its range (and how it interacts with another);
- the **waterfall**, which decomposes a *single* prediction into "this feature pushed the price up by \$X, that one pushed it down by \$Y."

```{admonition} The mental model
:class: tip
A dense net learns *what* to predict. SHAP tells you *why* it predicted that, one decision at a time. You will not deploy a model in this course that you cannot explain.
```

## 1.2 — Epistemic uncertainty with MC Dropout

Here is the first beautiful trick. Dropout is normally a *training-time* regularizer — randomly zero out neurons so the network can't over-rely on any one path. **Monte Carlo Dropout** leaves dropout *on at inference time* and runs the same input through the network many times (we use 100 passes). Each pass takes a slightly different random sub-network, so you get a *distribution* of predictions instead of one number.

The **mean** of those passes is your prediction. The **variance** is your **epistemic uncertainty** — the model's uncertainty about its own knowledge, high exactly where it has seen little similar data. In Keras, the key is calling the dropout layer with `training=True` during prediction.

We then check ourselves with **coverage**. For each test point, is the true value within $\pm 1.96$ standard deviations of our predicted mean? If our uncertainty is honest, that should happen about **95%** of the time. (Don't trust the bell curve? Use empirical percentiles instead — same logic, no Gaussian assumption.) When you plot coverage against the target, you'll often find it's excellent in the bulk of the data and **falls apart in the tails** — for us, the expensive houses — which is precisely the signal that you need more features or more data out there.

## 1.3 — Adding aleatoric uncertainty ("two heads, baby")

Epistemic uncertainty is reducible: collect more data and it shrinks. But some uncertainty is **irreducible noise in the data itself** — two identical-looking cars genuinely sell for different prices. That's **aleatoric uncertainty**, and MC Dropout alone won't capture it.

So we give the network **two heads**: one predicting the mean $\mu$, one predicting the variance $\sigma^2$ of the observation noise. Train it with a loss that rewards being both accurate *and* honestly calibrated about its noise. Now we can decompose total uncertainty into its two sources:

$$\underbrace{\text{Var}(\hat{y})}_{\text{total}} \;=\; \underbrace{\text{Var}(\mu \text{ across MC passes})}_{\text{epistemic (model)}} \;+\; \underbrace{\overline{\sigma^2}}_{\text{aleatoric (data)}}$$

The punchline, which genuinely changes the story: once you add the aleatoric head, you often discover that **data noise dominates**. That tells you something the point prediction never could — that the path to a better model is *better features and better data*, not a fancier architecture.

```{admonition} See it counted — and why the variance head blows up
:class: important
The companion notebook **`BayesianNN_Params_Epistemic_vs_Aleatoric.ipynb`** opens this model up parameter by parameter, and it answers the question that bites everyone the first time: *why does the aleatoric head sometimes explode into NaNs?* You train the two heads with the **Gaussian negative log-likelihood**, $\tfrac12\big(\log\sigma^2 + (y-\mu)^2/\sigma^2\big)$ — and that $(y-\mu)^2/\sigma^2$ term means a **tiny predicted $\sigma^2$ on a wrong point divides a real error by almost zero**, sending the loss to infinity. The notebook plots the loss as a U-shape in $\sigma^2$, **minimized exactly at $\sigma^2=(y-\mu)^2$** (so the model is gently forced to *learn* the noise level), and shows the standard cure: **predict $\log\sigma^2$, not $\sigma^2$** — then $\sigma^2=e^{s}>0$ always and the loss is smooth. It also makes the parameter split concrete: **epistemic uncertainty adds zero parameters** (it's the dropout you already have, measured across MC passes), while **aleatoric uncertainty is a whole second head**.
```

```{admonition} SHAP meets uncertainty
:class: note dropdown
You can run SHAP against **three** targets, not one: the mean (which features drive the prediction), the epistemic variance (which features make the model *unstable* because it lacks data there), and the aleatoric variance (which features are inherently *noisy*). A feature with high SHAP on aleatoric uncertainty is telling you the data for it is poor — no model can learn its way out of that.
```

## Project 1 — Probabilistic car prices

Bring it together: scrape (or pull) the car data, fit a Bayesian(ish) network, and report not just predictions but **intervals**, **coverage**, and a **SHAP** analysis of your best successes and biggest misses. Can your network beat a simple decision tree on both the point estimate *and* the honesty of its uncertainty? That's the bar.
