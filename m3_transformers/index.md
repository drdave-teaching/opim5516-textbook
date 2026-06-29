# Module 3 — Transformers

This is the architecture that ate machine learning. The same core idea — **attention** — now powers the models that write text, generate images, and forecast the weather. Rather than present it as magic, we earn it: we motivate the transformer against the LSTM you already know, open the hood on how data flows through it and how many parameters it actually has, and then apply it to four kinds of data in turn.

```{admonition} See the math — a "counted" notebook for every method
:class: important
The recurring theme here is **how data is transformed through the network** and **where every trainable parameter lives, and why**. There's a pure-numpy "counted" companion notebook for **every data modality** — each builds the model by hand, prints a table of *every weight and bias with a one-line reason it exists*, and traces the tensor shapes from input to output. The punchline across all of them: **it's the same transformer block every time — only how the input becomes tokens (and how tokens become the answer) changes.**

| Modality | Notebook | The trick at the ends |
|---|---|---|
| **(recurrent baseline)** | `LSTM_Params_and_ShapeFlow` | four gates; $4H(i{+}H{+}1)$, exactly Keras's number |
| **Time series** | `TimeSeries_Transformer_Params_and_ShapeFlow` | input embedding is a `Linear(F→d)`, not a token lookup; head forecasts the next value |
| **Text / GPT** | `Transformer_Params_and_ShapeFlow` | token-ID lookup + a (free) causal mask; scales to confirm **GPT-2 small = 124M** |
| **Images** | `ViT_Params_and_ShapeFlow` | an *image becomes patches → tokens* |
| **Video** | `VideoTransformer_Params_and_ShapeFlow` | frames become *space-time tokens*; a pixel decoder draws the next frame |

If you can read those tables and predict a `model.summary()`, you understand the model; if you can't, you're just calling `.fit()` and hoping.
```

## 3.1 — Time series: LSTM vs. Transformer

We start on familiar ground, forecasting an energy/weather series. First fit an **LSTM in PyTorch** — a recurrent network that walks the sequence one step at a time, carrying a hidden state. It works, but it's sequential and it forgets.

Then fit a **transformer** on the same data. Instead of marching through time, **self-attention** lets every time step look directly at every other time step and decide what's relevant — long-range dependencies in one hop, and fully parallel. We compare the two head to head, and — keeping the Module 1 thread alive — we put **epistemic uncertainty** on the transformer and run an **xAI stress test**: perturb the input temperature by ±10°F and watch how the attention weights and the predicted uncertainty respond. A model that doesn't flinch when you lie to it isn't paying attention.

## 3.2 — Text: building Dave-GPT

Language is the transformer's native habitat. We build up a small **GPT** from its pieces:

- **Causal masking** — the rule that a token may only attend to tokens *before* it, so the model genuinely *predicts* the next word rather than peeking at it.
- A **decoder-only** stack trained on the SEC filings dataset.
- **Fine-tuning** a pretrained `distilgpt2` instead of training from zero.
- Wiring it into **Dave-GPT**: a retrieval-augmented (**RAG**) assistant that answers from a stack of the course's own notebooks as its source material — and the discussion assignment asks you to build your *own* "Course-GPT" for another class you're taking.

## 3.3 — Images: the Vision Transformer

A transformer has no idea what an image is — so we make an image look like a sentence. Chop it into a grid of fixed-size **patches**, flatten and linearly embed each patch into a token, prepend a special **classification token**, and feed the whole sequence to a standard transformer encoder. That's the **Vision Transformer (ViT)**.

We fit a ConvNet baseline in PyTorch first, then the ViT on the same images (the EuroSAT satellite dataset), compare them, and apply xAI to see *where* each model looks. The lesson is conceptual as much as practical: convolution bakes in a strong prior (locality); attention learns its own — given enough data.

## 3.4 — Video: spatiotemporal forecasting

The capstone of the architecture tour. Treat a **video as a sequence of images** and you can forecast the future *frames* — and for us those frames are maps of the world. We download real high-resolution weather data (**HRRR**, 3 km over the continental US — temperature and wind) and coarse global forecasts (**GFS**), and build a **video transformer** that predicts upcoming temperature or energy-demand fields. We evaluate it not just with MAE but with **SSIM**, because a forecast map has to be *structurally* right, not merely close on average, and we fine-tune a **foundation model** (the IBM–NASA Prithvi family) rather than training from scratch.

## Project 3 — Spatiotemporal forecasting with transformers

Pick your path: **AI weather forecasting** (the full video-transformer treatment) or a more robust **energy-demand forecasting** project. Either way you'll build, evaluate, and explain a spatiotemporal transformer end to end — and, true to the spirit of the whole course, report not just how accurate it is but how much you can trust it.
