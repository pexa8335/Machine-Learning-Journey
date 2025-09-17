---
tags:
  - LinearRegression
  - ML
  - "#Math"
Date: 2025-09-17
Relevant:
  - "[[3.2 - Ordinary Least Square (OLS).]]"
  - "[[3.1 - Linear Regression.]]"
---

>[!important] Math convention
>By convention, vectors are typically treated as column vectors.

$$
y = \begin{bmatrix} 
y_{1} \\
y_{2} \\
\vdots \\
y_{M} 
\end{bmatrix} \in \mathbb{R}^{M \times 1} \quad \text{or} \quad \in \mathbb{R}^M
$$
When using $y^T$, it becomes a row vector.
$$
y^T = \begin{bmatrix}
y_{1}, y_{2}, \dots, y_{M}
\end{bmatrix} \in \mathbb{R}^{1 \times M}
$$
---
## RSS(w) formula.

$$
RSS(w) = \sum_{i=1}^M(y_{i} - w_{0} - w_{1}x_{i_{1}} - w_{2}x_{i_{2}} - \dots - w_{n}x_{i_{n}})^2
$$

Note:
-  The dataset $X$ is a collection of $M$ observation vectors.
	- $x_{i}$ is an observation vector in the dataset, each observation contains $n$ features.
	- After augmenting with a '1' for the bias term, $x_{i}$ becomes $x_{i} = (1, x_{i_{1}}, x_{i_{2}},\dots,x_{i_{n}})^T$, so $x_i \in \mathbb{R}^{n+1}$.
- $y_{i}$ is the **true label (scalar)** for each corresponding $x_{i}$. For instance, with observation $x_{1} = (1, 2, 3)$ we have $y_{1} = 5$.
- $w = (w_{0}, w_{1}, w_{2}, \dots, w_{n})^T$, so $w \in \mathbb{R}^{n+1}$.
	- This is the parameter vector we want to find.
- $M$ is **the number of observations/vectors** in the dataset.
- $n$ is **the number of features** in each original observation $x_{i}$ (excluding the bias term).

Define a matrix $A \in \mathbb{R}^{M \times (n+1)}$ where each **row** is an augmented observation vector $x_{i}^T$.

$$
A = \begin{pmatrix}
1 & x_{11} & x_{12} & \dots & x_{1n} \\
1 & x_{21} & x_{22} & \dots & x_{2n} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{M1} & x_{M2} & \dots & x_{Mn}
\end{pmatrix}
$$
Define a vector $y \in \mathbb{R}^{M \times 1}$ (or $\mathbb{R}^M$) which has each element as a $y_{i}$ scalar.
$$
y = \begin{bmatrix}
y_{1} \\
y_{2} \\
\vdots \\
y_{M}
\end{bmatrix}
$$
>[!question]
>Why do we have a column of '1's in matrix A?
>- To represent the bias term $w_{0} \times 1$.

Rewrite the function:
$$
RSS(w) = ||y - Aw||^2
$$
Review the [[Euclidean norm (L2 norm).]] to understand why we have this formula.

From this formula, we have:
$$
RSS(w) = (y-Aw)^T(y-Aw)
$$
$$
\Leftrightarrow y^Ty - y^TAw - (Aw)^Ty + (Aw)^T(Aw)
$$

Since $y^TAw$ is a scalar, $(y^TAw)^T = w^TA^Ty$. Also, $(Aw)^Ty = w^TA^Ty$.
Thus, $y^TAw + (Aw)^Ty = y^TAw + w^TA^Ty$.
Since $y^TAw$ is a scalar, $y^TAw = (y^TAw)^T = w^TA^Ty$.
So, $y^TAw + (Aw)^Ty = 2y^TAw$.

$$
\Leftrightarrow y^Ty - 2y^TAw + (Aw)^T(Aw)
$$
Our goal is to find the **minimum point** to **minimize** the $RSS(w)$ function, so we take the gradient with respect to $w$ and set it to zero:
$$
\nabla_w RSS(w) = 0
$$
$$
\Leftrightarrow -2A^Ty + 2A^TAw = 0
$$
$$
\Leftrightarrow A^TAw = A^Ty
$$
$$
\Leftrightarrow w^* = (A^TA)^{-1}A^Ty
$$

When **differentiating** with respect to $w$, the derivative of $-2y^TAw$ is $-2A^Ty$.
The derivative of $(Aw)^T(Aw) = w^TA^TAw$ is $2A^TAw$. (This is because $A^TA$ is a symmetric matrix).

We assume that $A^TA$ is invertible.

If the columns in $A$ are linearly dependent (i.e., $A$ does not have full column rank), then $A^TA$ will be non-invertible (singular).

>[!danger]
>High dimension space requires considerable computation.

