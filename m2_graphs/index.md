# Module 2 — Graph Neural Networks

A flat table is a convenient lie. It pretends every row is independent — that your movie taste has nothing to do with your friends', that one road sensor's traffic has nothing to do with the sensor upstream. But the interesting structure in the world is **relational**: who is connected to whom, what is near what. This module teaches networks that compute over that structure directly. We get there gently, starting from a method everyone has felt the effects of — recommender systems.

## 2.1 — Collaborative filtering and hybrid neural factorization

Every time YouTube, Amazon, or Netflix decides what to put in front of you, some version of this is running. We use the **MovieLens** dataset — users, movies, and the ratings between them — and the core idea is the **embedding**: a learned vector that describes each user and each movie. You don't hand-design these descriptions; you specify the *size* (say, 50 dimensions) and the model **learns the values through backpropagation**, much like the word embeddings you met in text analytics.

The simplest model, **collaborative filtering**, has no dense layers at all. You take the **dot product** of a user's embedding and a movie's embedding, squeeze it through a sigmoid, and that's your predicted rating:

$$\hat{r}_{u,m} = \sigma\big(\mathbf{e}_u \cdot \mathbf{e}_m + b\big)$$

To find similar movies, you don't even need ratings — you compare embedding directions with **cosine similarity**. Vectors that point the same way are alike.

Then we upgrade to **hybrid neural factorization**: concatenate a one-hot **genre** vector onto the learned embeddings, add a couple of dense layers with batch normalization and dropout, and let the network combine collaborative signal with content features. When it works, ask it for movies near *Toy Story* and it returns Monsters Inc., Moana, Frozen — it found the animated-Disney cluster on its own. (Woody smiles.)

```{admonition} Cold start
:class: note
What do you do for a brand-new user or a movie with no ratings — no embedding yet? Cluster by demographics, fall back to popularity, or **embed the content itself** (run the plot synopsis through a transformer to get a vector) and use similarity. The cold-start problem is the hardest and most practical question in the whole area.
```

```{admonition} The bridge — your recommender *is* a graph
:class: important
Here is the link to the rest of this module, and it's worth pausing on. What you just built in 2.1 is already a **graph**: a *bipartite* one, with users on one side, movies on the other, and an edge wherever someone rated something. The dot product $\mathbf{e}_u\cdot\mathbf{e}_m$ asks "do these two embeddings point the same way?" — it only ever looks **one hop** away, at movies you personally rated. The natural next question is: *what if we also used your friends' tastes, and their friends'?* That's exactly what a graph neural network does — it lets information flow **multiple hops** across the edges. Collaborative filtering is the one-hop special case. Now we generalize it.
```

## 2.2 — Adjacency, smoothing, and LightGCN

Now we make the graph explicit. Encode "who connects to whom" as an **adjacency matrix** $A$. The central move of a graph neural network is **smoothing**: a node's representation becomes a blend of its own features and those of its neighbors — the items that are "around you." Stack a few rounds of this and information propagates outward across the graph.

**LightGCN** strips the idea to its essence for recommendation: no feature transformations, no nonlinearities, just repeated neighborhood smoothing over the user–item graph. It's the cleanest demonstration that, on a dense enough graph, *the structure itself carries the signal.*

### See the algorithm — a GCN in pure numpy

Before you call a library, it's worth building one layer by hand, because the whole thing fits in three matrices. First let every node hear *itself* (self-loops, $\hat A = A + I$) and normalize so popular nodes don't drown everyone out: $\tilde A = \hat D^{-1/2}\hat A\hat D^{-1/2}$. Then **one GCN layer is three jobs**:

$$H^{(l+1)} \;=\; \sigma\big(\;\underbrace{\tilde A}_{\text{mix with neighbors}}\;\underbrace{H^{(l)}}_{\text{features}}\;\underbrace{W^{(l)}}_{\text{transform}}\;\big)$$

The companion notebook does this on **Zachary's Karate Club** (34 people, two factions) and shows three things you can *see*:

- **Propagation alone reveals the communities.** With *random, untrained* weights, just multiplying by $\tilde A$ a few times pulls the two factions apart. Structure does the work.
- **Oversmoothing is real.** Stack too many layers and every node averages with everyone until they're identical — the communities collapse into one blob. More layers is *not* more better.
- **Two labels are enough.** A 2-layer GCN, told only which faction nodes 0 and 33 belong to, labels the **other 32 people at ~97% accuracy** — because the graph carries the signal.

```{admonition} ✋ Do it yourself
:class: tip
Run `GNN_FromScratch_SeeTheAlgorithm.ipynb`, and first do the one-page **message-passing-by-hand worksheet**: build $\hat A$, $\tilde A$, and one smoothing step on a 5-node graph with a pencil. If you can do one step by hand, you understand the algorithm.
```

## 2.3 — GCNs and GraphSAGE

With the machinery in place, two workhorse architectures:

- **GCN (Graph Convolutional Network)** — smooth over the whole graph at once using the normalized adjacency matrix. Powerful, but it needs the entire graph in memory.
- **GraphSAGE** — instead of using all neighbors, *sample* a fixed number and aggregate. This makes it **inductive**: it generalizes to nodes (and whole graphs) it never saw in training, which is what you need in production.

In PyTorch (with PyTorch Geometric), swapping one for the other is essentially a one-line change — `GCNConv` → `SAGEConv` — and seeing that is the point. The abstraction is clean: define the message-passing layer, and the framework handles the graph plumbing.

```{admonition} See the sampling — why GraphSAGE exists
:class: tip
The companion notebook `GCN_vs_GraphSAGE_SeeTheSampling.ipynb` draws the difference. A GCN reads **all** of a node's neighbors every layer, so a node with 10,000 friends costs 10,000× more — and a *brand-new* node means retraining the whole model. GraphSAGE instead **samples $K$** neighbors (fixed cost, no matter how popular) and learns the *aggregator function* $h_v=\sigma\!\big(W\cdot[\,h_v \Vert \text{mean}(\text{sample}(N(v)))\,]\big)$ rather than per-node embeddings. The payoff you can watch: add a new student who befriends both faction leaders *after* training, and GraphSAGE embeds them on the spot — **inductively**, no retraining — which a plain GCN simply cannot do.
```

## Project 2 — Traffic prediction

The original plan used everyone's LinkedIn network as the graph, but those graphs turned out **too sparse** to be interesting — a recurring lesson about graph methods: they only shine when the graph is dense enough to have real signal. So the project pivots to **traffic**: a spatiotemporal sensor network where each node carries a time series of speeds or volumes, and a GNN predicts what happens next. Build it for Connecticut or a state of your choice; if you're feeling ambitious, fold in weather data.

```{admonition} Where this is heading
:class: tip dropdown
This module is also a runway for Module 3. The "embed the content, then compute similarity" move from cold-start, and the spatiotemporal framing of the traffic project, both reappear when we forecast weather and energy with transformers over sequences of images.
```
