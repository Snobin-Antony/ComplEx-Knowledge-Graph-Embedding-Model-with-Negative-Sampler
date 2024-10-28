# ComplEx-Knowledge-Graph-Embedding-Model-with-Negative-Sampler
A ComplEx knowledge graph embedding model from scratch using heterogeneous book graph dataset and negative sampler, implemented in PyTorch.

## Project Overview 📝
This project consists of two parts:

* Part 1: Build a ComplEx knowledge graph embedding model.
* Part 2: Implement a Negative Sampler

  
#### Download Data: [here](https://drive.google.com/file/d/1fzwUXMnDm_JbGYvAvgReipbVasTct8XQ/view?usp=sharing)

### Background KG Data 🌐

Graph data are represented by graph triples. A triple `(h,r,t)` consists of three parts: the `head (h)`, `relation(r)`, and `tail(t)`.

Here’s a simple example:

This graph:

<div style="text-align:center;">
<img src="https://drive.google.com/uc?export=view&id=1mKxZX0sTk584MUdyXsPT1zFKx1tLcUl9"   width="300"/>
</div>

will be represented as the following triplet

```
Head: "France"
Relation: "hasCapital"
Tail: "Paris"
```
i.e. `(France, hasCapital, Paris)`


## Dataset Overview 📊
The given dataset represents a heterogeneous graph in which the nodes represent: _authors, books, genres, publishers, awards, and readers_. The edges represent various types of relationships between these entities, such as:

* (Author, _wrote_, Book)
* (Book, _published_by_, Publisher)
* (Book, _belongs_to_genre_, Genre)
* (Book, _won_award_, Award)
* (Reader, _read_, Book)

#### Diagram
<div style="text-align:center;">
    <img src="https://drive.google.com/uc?export=view&id=1QOmxWVFYLmUwfKEHKHoePdbTLbV37nCO" width="600" />
</div>

## Part 1: ComplEx

Using the provided heterogeneous book graph dataset, implementd a [ComplEx](https://arxiv.org/pdf/1606.06357) knowledge graph embedding model from scratch using PyTorch. Tuned the parameters and optimized the model for improved performance and also made modely enhancements.

## Part 2: Negative Sampler

In this section, we will address the challenge the model faces under the Open World Assumption: distinguishing between false facts and missing ones. To assist our model in learning this distinction, we will employ a Negative Sampling strategy. This involves generating "corrupted" versions of existing facts, which we will use as negative samples.

### Example
Let's consider a simple example:

$$
\mathcal{E} = \{\text{Mike, George, Liverpool, Manchester, London}\}, \\
\mathcal{R} = \{\text{bornIn, friendsWith}\}, \\
f \in G = \{\text{Mike, bornIn, Liverpool}\}, \\
$$

where, $G$ is our knowledge graph with $\mathcal{E}$, the set of entities, $\mathcal{R}$ the set of relations  and $f$ is a true fact.

If we change the tail of the fact, we can generate the following synthetic negatives:

$$
\text{negatives} = \begin{bmatrix}
\text{(Mike, bornIn, Manchester)  } \\
\text{(Mike, bornIn, George)  } \\
\text{(Mike, bornIn, London )  }
\end{bmatrix}.
$$


Some Negative Samplers can be more edge-informed, producing more relevant negatives, such as:

$$
\text{negatives} = \begin{bmatrix}
\text{(Mike, bornIn, Manchester)  } \\
\text{(Mike, bornIn, London )  }
\end{bmatrix}.
$$


### Uniform Negative Sampler

1. Implemented a uniform negative sampler that corrupts the tail entity (created the data-informed version)
2. Integrated it into the training pipeline
