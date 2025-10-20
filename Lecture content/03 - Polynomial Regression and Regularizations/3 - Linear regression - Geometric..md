---
tags:
  - GradientDescent
  - LinearRegression
  - Math
  - ML
Date: 2025-10-17
Relevant:
  - "[[Gradient Descent.]]"
  - "[[Lecture content/02 - Linear Regression & Gradient Descent/Linear Regression.|Linear Regression.]]"
  - "[[Mathematical proof - Normal Equation.]]"
Video: https://www.youtube.com/watch?v=SGbelq087Qs
---
# 1. Geometric Intepretation.

![[Pasted image 20251018160900.png]]

Let's say we have 2 feature (d = 2) and 3 samples (n = 3).
Feature 1 ($F_{1}$) is height, feature 2 ($F_{2}$) is weight, we have 3 samples so $F_{1}, F_{2}$ are 3-dimensional vectors.

Và tương ứng theo đó, we have a 3-dimensional vector $y = [y_{1}, y_{2}, y_{3}]$.

Phác thảo theo không gian 3D ta được ảnh như trên.

![[Pasted image 20251018161100.png]]

Randomly draw a point = $3f_{1} + 2.5f_{2}$ and $-5f_{1} + 3f_{2}$ (randomly positioned, just for simulation) $\to$ this is the linear combination of 2 vectors $f_{1}, f_{2}$. 

These 2 $f_{1}, f_{2}$ vectors create a subspace (a 2-dimensional plane).

![[Pasted image 20251018161404.png]]

And we have to find the closest point from the label to the plane. The point is on the plane so it's the linear combination of $f_{1}, f_{2}$ - it can 100% be represented by using 2 vectors $f_{1}, f_{2}$.

![[Pasted image 20251018161515.png]]

## 2. How to get this point? 
Use the vector label ($y$ vector), project it onto the subspace spanned by the feature vectors $f_{1}, f_{2}$ (The 2-dimensional plane).

So basically we're looking for all linear combination of feature vectors result in a point that is closest to the label.

## How to understand this point?

Which one is the closest among all possible points?

Because this point is on the plane spanned by the feature vectors $f_{1}, f_{2}$, it surely can be represented by the linear combination of these two feature vectors.

This point will be interpreted as $w_{1}f_{1} + w_{2}f_{2}$.

$$
X^T = \begin{pmatrix}f_{1}  & f_{2}\end{pmatrix}
$$
$$
w = \begin{pmatrix}
w_{1}\\
w_{2}
\end{pmatrix}
$$
After using this notation, we have:
$$
X^Tw
$$
![[Pasted image 20251018163126.png]]

![[Pasted image 20251018163331.png]]

Basic addition rule in vector:
- Place the tail (the start) of the error vector ($y - X^Tw$), let's say this vector is $e$, at the end (arrowhead) of the first vector $X^Tw$.
- The sum, $X^Tw + (y - X^Tw)$ is the new vector goes from the tail of $X^Tw$ to the arrowhead of ($y-X^Tw$) - vector $y$ - the label vector.

Define:
- $X^Tw = p$
- $y - X^Tw = e$
- e + p = y.

The angel between e and p is $90^o$ so the dot product = 0. So we have, 
$$
(y-X^Tw)^T(X^Tw) = 0 \iff yX^Tw - X^Tw.Xw^T
$$

Recall from [[Mathematical proof - Normal Equation.]]: $w^* = (X^TX)^{-1}X^Ty$



----

There are 4 Kings in a deck of 52 cards. What is the probability of drawing 2 Kings.

P(Man|Pink) = P(Pink|man).P(man) = 5/40 x 0.4 /
chia P pink

There are 100 people at a party. You count how many people is wearing pink and how many men there are. You get these numbers:


(Man,Pink) = 5

(Man, NotPink) = 35

(NotMan, Pink) = 20

(NotMan, NotPink) = 40

  

What is the probability that a person wearing pink is a man P(Man|Pink) = ?

The overall success rate of a Better hospital is 50/100, and the overall success rate of a Normal hospital is 68/100. Can we conclude that the Normal hospital is more successful in treating patients?

![[Pasted image 20251018101938.png]]

C

![[Pasted image 20251018102250.png]]

infected = 1.8
not infected = 98.2

P(positive|infected ) = 0.95
P(pos | not infect) = 0.03

![[Pasted image 20251018102614.png]]
c

P(Pos|virus ) - 0.9
(pos | not) = 0.08

virus = 0.01

