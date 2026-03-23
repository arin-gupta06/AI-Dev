# 📐 Mathematics Complete Master Guide for AI & Machine Learning

## (Theory + Application + Tricks — From Foundations to Advanced)

> **Purpose:** This guide covers every mathematical concept essential for AI/ML — with theory, intuition, formulas, Python code, tricks, and real-world AI applications. Use it for university exams, interviews, research, and building AI systems.

> **📚 Learning Framework — Every topic in this guide is structured around three core questions:**
>
> | Question | What It Answers | Icon |
> |----------|-----------------|------|
> | **WHY?** | Why should I learn this? What breaks without it? | 🔍 |
> | **WHAT?** | What is this concept? Definitions, formulas, theory. | 📖 |
> | **HOW?** | How do I use it? Code, tricks, applied examples. | ⚙️ |
>
> Look for these markers throughout the guide.

------------------------------------------------------------------------

# 🔷 PART I — LINEAR ALGEBRA

------------------------------------------------------------------------

# 1. Scalars, Vectors, Matrices & Tensors

## 1.1 Definitions

| Object | Description | Example |
|--------|-------------|---------|
| **Scalar** | Single number (0D) | `x = 5` |
| **Vector** | 1D array of numbers | `v = [1, 2, 3]` |
| **Matrix** | 2D grid of numbers (rows × cols) | `A = [[1,2],[3,4]]` |
| **Tensor** | n-dimensional array (generalization) | 3D+ arrays in deep learning |

## 1.2 Why It Matters for AI

- **Scalars** → learning rate, bias terms
- **Vectors** → feature vectors, word embeddings, gradients
- **Matrices** → weight matrices in neural networks, datasets, transformations
- **Tensors** → image batches (4D), video data (5D), transformer attention

## 1.3 Code: Creating & Inspecting

``` python
import numpy as np

# Scalar
alpha = 0.01

# Vector (1D)
v = np.array([1, 2, 3])
print(v.shape)  # (3,)

# Matrix (2D)
A = np.array([[1, 2], [3, 4], [5, 6]])
print(A.shape)  # (3, 2)

# Tensor (3D) — e.g., batch of 2 grayscale 3x3 images
T = np.random.randn(2, 3, 3)
print(T.shape)  # (2, 3, 3)
```

## 1.4 Tricks & Tips

- Always check `.shape` before any operation — shape mismatches are the #1 bug in ML.
- Use `np.expand_dims()` or `reshape()` to add/remove dimensions for broadcasting.
- In PyTorch: `tensor.unsqueeze(0)` adds a batch dimension.

------------------------------------------------------------------------

# 2. Vector Operations

> 🔍 **WHY?** Vectors are the atoms of AI. Every data point is a vector. Every gradient is a vector. Similarity, direction, and magnitude — all vector operations. You literally cannot do ML without this.
>
> 📖 **WHAT?** Operations that combine, compare, and measure vectors — dot products, norms, projections.
>
> ⚙️ **HOW?** NumPy makes vector math trivial. Below are the operations you'll use daily.

## 2.1 Addition & Scalar Multiplication

```
u + v = [u₁+v₁, u₂+v₂, ..., uₙ+vₙ]
c · v = [c·v₁, c·v₂, ..., c·vₙ]
```

``` python
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

print(u + v)      # [5, 7, 9]
print(3 * u)      # [3, 6, 9]
```

## 2.2 Dot Product (Inner Product)

**Formula:**  `u · v = Σ uᵢvᵢ = |u||v|cos(θ)`

``` python
dot = np.dot(u, v)       # 32
dot = u @ v              # 32  (preferred syntax)
```

**AI Application:**
- Measures similarity between vectors
- Core of attention mechanisms: `score = Q · Kᵀ`
- Cosine similarity = `(u · v) / (|u| · |v|)`

## 2.3 Vector Norm (Magnitude)

| Norm | Formula | Use Case |
|------|---------|----------|
| **L1 (Manhattan)** | `Σ|xᵢ|` | Lasso regularization, sparse features |
| **L2 (Euclidean)** | `√(Σxᵢ²)` | Ridge regularization, distance metrics |
| **L∞ (Max)** | `max(|xᵢ|)` | Adversarial robustness |

``` python
x = np.array([3, 4])

l1 = np.linalg.norm(x, 1)    # 7
l2 = np.linalg.norm(x, 2)    # 5.0
linf = np.linalg.norm(x, np.inf)  # 4
```

## 2.4 Cross Product (3D only)

``` python
a = np.array([1, 0, 0])
b = np.array([0, 1, 0])
cross = np.cross(a, b)  # [0, 0, 1]
```

**Use:** 3D graphics & robotics, surface normals.

## 2.5 Trick: Unit Vectors

Normalize any vector to length 1 → direction without magnitude.

``` python
unit_v = v / np.linalg.norm(v)
```

**Why?** Word2Vec, GloVe embeddings are often normalized before cosine similarity.

------------------------------------------------------------------------

# 3. Matrix Operations

> 🔍 **WHY?** A neural network IS a chain of matrix multiplications + activations. Understanding matrix ops = understanding what happens inside every ML model.
>
> 📖 **WHAT?** How to multiply, transpose, invert, and decompose matrices — the building blocks of linear transformations.
>
> ⚙️ **HOW?** NumPy's `@` operator and `np.linalg` module handle everything. Key: shapes must align — `(m×n) @ (n×p) → (m×p)`.

## 3.1 Basic Operations

``` python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Element-wise
print(A + B)
print(A * B)          # Hadamard product (element-wise)

# Matrix multiplication
print(A @ B)          # [[19,22],[43,50]]
print(np.matmul(A, B))
```

## 3.2 Transpose

`Aᵀ[i,j] = A[j,i]` — flips rows ↔ columns.

``` python
print(A.T)
# [[1, 3],
#  [2, 4]]
```

**AI Rule:** `(AB)ᵀ = BᵀAᵀ` — order reverses!

## 3.3 Special Matrices

| Matrix | Property | AI Use |
|--------|----------|--------|
| **Identity (I)** | `AI = IA = A` | Initialization, skip connections |
| **Diagonal** | Non-zero only on diagonal | Scaling, variance matrices |
| **Symmetric** | `A = Aᵀ` | Covariance matrices, kernels |
| **Orthogonal** | `AᵀA = I`, `A⁻¹ = Aᵀ` | Stable training, rotations |
| **Sparse** | Mostly zeros | Graph neural networks, NLP |

``` python
I = np.eye(3)                    # 3x3 identity
D = np.diag([2, 3, 4])          # diagonal matrix
```

## 3.4 Inverse & Pseudoinverse

`A⁻¹A = AA⁻¹ = I`  — only exists for square, non-singular matrices.

``` python
A_inv = np.linalg.inv(A)

# Pseudoinverse (works for non-square / singular)
A_pinv = np.linalg.pinv(A)
```

**AI Application:** Solving `Ax = b` → `x = A⁻¹b` → used in linear regression closed-form: `θ = (XᵀX)⁻¹Xᵀy`

**⚠️ Trick:** Never compute inverse directly in practice — use `np.linalg.solve(A, b)` which is faster and numerically stable.

## 3.5 Determinant

`det(A)` measures how a matrix scales space.

``` python
det = np.linalg.det(A)  # -2.0
```

- `det = 0` → singular (no inverse), columns are linearly dependent
- `|det| > 1` → expansion, `|det| < 1` → contraction

## 3.6 Trace

`tr(A) = Σ Aᵢᵢ` — sum of diagonal elements.

``` python
trace = np.trace(A)  # 5
```

**Property:** `tr(AB) = tr(BA)` — cyclic permutation invariant. Used in Frobenius norm: `‖A‖_F = √(tr(AᵀA))`

------------------------------------------------------------------------

# 4. Systems of Linear Equations

> 🔍 **WHY?** Linear regression, least squares, and even neural network weight updates boil down to solving systems of equations. This is the mathematical backbone of fitting models to data.
>
> 📖 **WHAT?** A system `Ax = b` asks: given a transformation A and output b, what input x produced it?
>
> ⚙️ **HOW?** Use `np.linalg.solve()` (never manually invert). For overdetermined systems → least squares.

## 4.1 Matrix Form

```
2x + 3y = 8        →    Ax = b
4x + y  = 6             where A = [[2,3],[4,1]], x = [x,y], b = [8,6]
```

## 4.2 Solving

``` python
A = np.array([[2, 3], [4, 1]])
b = np.array([8, 6])

x = np.linalg.solve(A, b)  # [1.0, 2.0]
```

## 4.3 Types of Solutions

| Condition | Solutions | Geometrically |
|-----------|-----------|---------------|
| `det(A) ≠ 0` | Unique | Lines/planes intersect at one point |
| `det(A) = 0`, consistent | Infinite | Lines/planes overlap |
| `det(A) = 0`, inconsistent | None | Parallel (no intersection) |

**AI Connection:** Overdetermined systems (more equations than unknowns) → least squares solution → linear regression.

------------------------------------------------------------------------

# 5. Eigenvalues & Eigenvectors

> 🔍 **WHY?** PCA (the most common dimensionality reduction) is 100% eigenvalue math. Google's PageRank is an eigenvector. Stability of training, convergence analysis, spectral methods — all eigenvalues. If you skip this, you cannot understand half of unsupervised learning.
>
> 📖 **WHAT?** Special vectors that a matrix only scales (doesn't rotate). They reveal the "natural axes" of a transformation.
>
> ⚙️ **HOW?** `np.linalg.eig()` computes them. For symmetric matrices (covariance), use `np.linalg.eigh()` — guaranteed real values and faster.

## 5.1 Definition

For a square matrix A, if: `Av = λv`

Then `v` is an **eigenvector** and `λ` is the corresponding **eigenvalue**.

**Intuition:** Eigenvectors are directions that don't change under transformation A — they only get scaled by λ.

## 5.2 Computing

``` python
A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Values:", eigenvalues)    # [5., 2.]
print("Vectors:\n", eigenvectors)
```

## 5.3 Properties

- Sum of eigenvalues = `tr(A)`
- Product of eigenvalues = `det(A)`
- Symmetric matrices have real eigenvalues and orthogonal eigenvectors
- If all eigenvalues > 0 → positive definite matrix

## 5.4 AI Applications

| Application | How Eigenvalues/Eigenvectors Are Used |
|-------------|---------------------------------------|
| **PCA** | Eigenvectors of covariance matrix = principal components |
| **Google PageRank** | Dominant eigenvector of link matrix = page importance |
| **Spectral Clustering** | Eigenvectors of graph Laplacian |
| **Stability Analysis** | Eigenvalues determine convergence of iterative methods |
| **Covariance Matrices** | Eigenvalues = variance along principal axes |

## 5.5 Trick: Quick Eigenvalue Check

For a 2×2 matrix `[[a,b],[c,d]]`:
- `λ₁ + λ₂ = a + d` (trace)
- `λ₁ · λ₂ = ad - bc` (determinant)

------------------------------------------------------------------------

# 6. Matrix Decompositions

> 🔍 **WHY?** Decompositions break complex matrices into simpler, interpretable pieces. SVD alone powers recommender systems, image compression, NLP embeddings, and pseudoinverses. Without this, you're treating matrices as black boxes.
>
> 📖 **WHAT?** Factoring a matrix into products of simpler matrices (diagonal, orthogonal, triangular).
>
> ⚙️ **HOW?** `np.linalg.svd()`, `np.linalg.cholesky()`, `np.linalg.qr()` — each decomposition has specific use cases and stability properties.

## 6.1 Eigendecomposition

`A = QΛQ⁻¹` where Q = matrix of eigenvectors, Λ = diagonal of eigenvalues.

Only works for square matrices with n linearly independent eigenvectors.

## 6.2 Singular Value Decomposition (SVD)

`A = UΣVᵀ` — works for ANY matrix (m×n).

| Component | Shape | Meaning |
|-----------|-------|---------|
| **U** | m×m | Left singular vectors (row space) |
| **Σ** | m×n | Singular values (diagonal, sorted ↓) |
| **Vᵀ** | n×n | Right singular vectors (column space) |

``` python
A = np.array([[1, 2], [3, 4], [5, 6]])
U, S, Vt = np.linalg.svd(A)

# Low-rank approximation (keep top k singular values)
k = 1
A_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
```

**AI Applications:**
- **Dimensionality reduction** (truncated SVD)
- **Recommender systems** (Netflix Prize used SVD)
- **Image compression** (keep top-k singular values)
- **Latent Semantic Analysis** in NLP
- **Pseudoinverse:** `A⁺ = VΣ⁺Uᵀ`

## 6.3 Cholesky Decomposition

For positive definite matrices: `A = LLᵀ` where L is lower triangular.

``` python
A = np.array([[4, 2], [2, 3]])
L = np.linalg.cholesky(A)
```

**Use:** Faster solving of `Ax = b`, sampling from multivariate Gaussians.

## 6.4 QR Decomposition

`A = QR` where Q is orthogonal, R is upper triangular.

``` python
Q, R = np.linalg.qr(A)
```

**Use:** Numerically stable least squares, training large models.

------------------------------------------------------------------------

# 7. Vector Spaces & Subspaces

> 🔍 **WHY?** Understanding rank tells you if your features are redundant. Null space reveals what information a transformation destroys. Projection onto subspaces IS what linear regression does. These abstract concepts have very concrete AI consequences.
>
> 📖 **WHAT?** The structural theory of how vectors relate — independence, span, basis, and the four fundamental subspaces of a matrix.
>
> ⚙️ **HOW?** Check rank with `np.linalg.matrix_rank()`. If rank < number of features → you have multicollinearity → consider PCA or regularization.

## 7.1 Core Concepts

- **Span:** Set of all linear combinations of a set of vectors
- **Linear Independence:** No vector can be written as a combination of others
- **Basis:** Minimal set of independent vectors that span the space
- **Dimension:** Number of vectors in a basis
- **Rank:** Dimension of the column space = `rank(A)`

``` python
rank = np.linalg.matrix_rank(A)
```

## 7.2 Four Fundamental Subspaces

| Subspace | Symbol | Dimension |
|----------|--------|-----------|
| Column Space | C(A) | rank(A) = r |
| Row Space | C(Aᵀ) | r |
| Null Space | N(A) | n - r |
| Left Null Space | N(Aᵀ) | m - r |

**AI Insight:** Rank-deficient matrices mean redundant features → signals to reduce dimensionality.

## 7.3 Projections

Projecting vector `b` onto column space of A:

`proj = A(AᵀA)⁻¹Aᵀb`

**This is exactly what linear regression does** — projects y onto the column space of X.

------------------------------------------------------------------------

# 🔷 PART II — CALCULUS

------------------------------------------------------------------------

# 8. Limits & Continuity

> 🔍 **WHY?** Limits define derivatives, and derivatives are the engine of all neural network training. Without limits, there are no gradients. Continuity determines whether gradient-based optimization can work on a function at all.
>
> 📖 **WHAT?** Limits describe function behavior as inputs approach a point. Continuity means no sudden jumps.
>
> ⚙️ **HOW?** You rarely compute limits manually in AI, but you implicitly rely on them every time you use an activation function or compute a gradient.

## 8.1 Limits

`lim(x→a) f(x) = L` means f(x) → L as x → a.

**Key Limits for AI:**

| Limit | Value | Where Used |
|-------|-------|------------|
| `lim(x→0) sin(x)/x` | 1 | Signal processing |
| `lim(x→∞) (1 + 1/x)ˣ` | e ≈ 2.718 | Exponential growth, softmax |
| `lim(x→0) (eˣ - 1)/x` | 1 | Taylor expansions |

## 8.2 Continuity

A function is continuous at `a` if: `lim(x→a) f(x) = f(a)`

**AI Relevance:** Activation functions must be continuous (mostly) for gradient flow. ReLU is continuous but not differentiable at 0 — subgradients handle this.

------------------------------------------------------------------------

# 9. Derivatives (Single Variable)

> 🔍 **WHY?** The derivative IS the gradient in 1D. Every model you train uses derivatives to update weights. The chain rule IS backpropagation. If you understand derivatives deeply, you understand how every neural network learns.
>
> 📖 **WHAT?** The instantaneous rate of change of a function. Measures how output changes when input changes by a tiny amount.
>
> ⚙️ **HOW?** Modern frameworks (PyTorch, TensorFlow) compute derivatives automatically via autograd. But knowing the rules lets you debug, derive custom gradients, and understand training dynamics.

## 9.1 Definition & Rules

`f'(x) = lim(h→0) [f(x+h) - f(x)] / h`

| Rule | Formula |
|------|---------|
| Power | `d/dx(xⁿ) = nxⁿ⁻¹` |
| Product | `(fg)' = f'g + fg'` |
| Chain | `d/dx f(g(x)) = f'(g(x)) · g'(x)` |
| Quotient | `(f/g)' = (f'g - fg') / g²` |

## 9.2 Essential Derivatives for AI

| Function | Derivative | AI Context |
|----------|------------|------------|
| `eˣ` | `eˣ` | Softmax, exponential decay |
| `ln(x)` | `1/x` | Log-loss (cross-entropy) |
| `σ(x) = 1/(1+e⁻ˣ)` | `σ(x)(1-σ(x))` | Sigmoid activation |
| `tanh(x)` | `1 - tanh²(x)` | Tanh activation |
| `ReLU(x) = max(0,x)` | `0 if x<0, 1 if x>0` | Most common activation |
| `softmax(xᵢ)` | `sᵢ(δᵢⱼ - sⱼ)` | Output layer for classification |

## 9.3 Code: Numerical Derivatives

``` python
def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

# Example: derivative of x² at x=3
numerical_derivative(lambda x: x**2, 3)  # ≈ 6.0
```

## 9.4 Trick: Chain Rule is the Foundation of Backpropagation

Every layer in a neural network applies the chain rule:

```
dL/dw₁ = dL/dŷ · dŷ/dz₂ · dz₂/da₁ · da₁/dz₁ · dz₁/dw₁
```

**This IS backpropagation.** Understanding the chain rule = understanding all of deep learning training.

------------------------------------------------------------------------

# 10. Partial Derivatives & Gradients

> 🔍 **WHY?** Real models have millions of parameters. The gradient tells you how to adjust ALL of them simultaneously to reduce loss. This is the mathematical heart of gradient descent — the algorithm that trains every neural network.
>
> 📖 **WHAT?** Partial derivatives extend derivatives to functions of many variables. The gradient collects all of them into one vector pointing "uphill."
>
> ⚙️ **HOW?** PyTorch's `autograd` and JAX's `grad` compute gradients automatically. The Jacobian and Hessian extend this to vector outputs and second-order information.

## 10.1 Partial Derivatives

For `f(x, y)`: differentiate w.r.t. one variable, treating others as constants.

```
f(x, y) = x²y + 3y²
∂f/∂x = 2xy
∂f/∂y = x² + 6y
```

## 10.2 Gradient

The gradient is the vector of all partial derivatives:

`∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]`

**Key Insight:** The gradient points in the direction of steepest ascent. Negative gradient = steepest descent → this is gradient descent!

``` python
# Computing gradients with autograd (PyTorch)
import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x[0]**2 + x[1]**3
y.backward()
print(x.grad)  # tensor([4., 27.])  → [2*2, 3*9]
```

## 10.3 Jacobian Matrix

For a vector-valued function `f: Rⁿ → Rᵐ`:

```
J[i,j] = ∂fᵢ/∂xⱼ    (m×n matrix)
```

**AI Use:** Maps how input perturbations affect all outputs. Used in GANs, neural ODEs, and sensitivity analysis.

## 10.4 Hessian Matrix

Matrix of second-order partial derivatives:

```
H[i,j] = ∂²f/∂xᵢ∂xⱼ
```

- Hessian positive definite → local minimum (convex)
- Hessian negative definite → local maximum (concave)
- Mixed signs → saddle point

**AI Use:** Second-order optimization (Newton's method), understanding loss landscape curvature. Computing full Hessian is O(n²) — too expensive for large NNs, so we use approximations (BFGS, Adam).

------------------------------------------------------------------------

# 11. Gradient Descent & Optimization

> 🔍 **WHY?** This is THE algorithm. Every model you've heard of — GPT, DALL-E, AlphaFold — was trained by some variant of gradient descent. Understanding it deeply lets you debug training failures, tune hyperparameters, and choose the right optimizer.
>
> 📖 **WHAT?** An iterative algorithm that moves parameters in the direction that reduces the loss function, using the gradient as a compass.
>
> ⚙️ **HOW?** In practice: pick an optimizer (Adam is the default), set a learning rate (~3e-4), and use a scheduler. Below is the theory behind it all.

## 11.1 The Core Algorithm

```
θ ← θ - α · ∇L(θ)
```

Where: `θ` = parameters, `α` = learning rate, `L` = loss function.

## 11.2 Variants

| Variant | Update Rule | Pros/Cons |
|---------|-------------|-----------|
| **Batch GD** | Full dataset gradient | Stable but slow |
| **Stochastic GD** | One sample's gradient | Fast but noisy |
| **Mini-batch GD** | Batch of samples | Best of both worlds |
| **Momentum** | `v = βv + ∇L; θ -= αv` | Escapes local minima |
| **Adam** | Adaptive LR per parameter | Most popular in deep learning |

## 11.3 Code: Gradient Descent from Scratch

``` python
def gradient_descent(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)

    for _ in range(epochs):
        predictions = X @ theta
        error = predictions - y
        gradient = (2/m) * X.T @ error
        theta -= lr * gradient

    return theta
```

## 11.4 Learning Rate Tricks

- **Too high:** Loss explodes or oscillates
- **Too low:** Converges too slowly
- **Schedule:** Reduce LR over time (step, cosine, warmup)
- **Rule of thumb:** Start with `3e-4` for Adam, `1e-2` for SGD

------------------------------------------------------------------------

# 12. Integrals

> 🔍 **WHY?** Probabilities are computed via integrals. Expected values, normalizing probability distributions, computing areas under ROC curves, marginalizing over latent variables in Bayesian methods — all integrals. Many are intractable, which is why Monte Carlo methods exist.
>
> 📖 **WHAT?** The reverse of differentiation. Integrals accumulate quantities — area under a curve, total probability, cumulative distributions.
>
> ⚙️ **HOW?** `scipy.integrate.quad()` for numerical integration. In AI, most integrals are estimated via sampling (Monte Carlo) rather than solved analytically.

## 12.1 Fundamental Theorem

`∫ₐᵇ f(x)dx = F(b) - F(a)` where `F'(x) = f(x)`

## 12.2 Key Integrals for AI

| Integral | Result | Use |
|----------|--------|-----|
| `∫ eˣ dx` | `eˣ + C` | Probability, moment generating |
| `∫ 1/x dx` | `ln|x| + C` | Entropy, information theory |
| `∫₋∞^∞ e^(-x²) dx` | `√π` | Gaussian integral (normalization) |
| `∫ xⁿ dx` | `xⁿ⁺¹/(n+1) + C` | Polynomial models |

## 12.3 Numerical Integration

``` python
from scipy import integrate

# Definite integral of x² from 0 to 1
result, error = integrate.quad(lambda x: x**2, 0, 1)
print(result)  # 0.333...
```

## 12.4 AI Applications

- **Expected Value:** `E[X] = ∫ x·f(x)dx`
- **Evidence (marginal likelihood):** `p(x) = ∫ p(x|θ)p(θ)dθ` — intractable, hence variational inference
- **Area under curve (AUC-ROC):** Evaluated via numerical integration
- **KL Divergence:** `KL(P‖Q) = ∫ p(x) ln(p(x)/q(x)) dx`

------------------------------------------------------------------------

# 13. Multivariable Calculus

> 🔍 **WHY?** Real-world AI operates in high-dimensional spaces. Multiple integrals compute joint probabilities. Taylor expansions justify why quadratic loss approximations work in optimization. These tools connect single-variable intuition to the multi-dimensional reality of ML.
>
> 📖 **WHAT?** Extending integration and approximation techniques to functions of multiple variables.
>
> ⚙️ **HOW?** `scipy.integrate.dblquad()` for double integrals. Taylor expansions are used implicitly in Newton's method and natural gradient descent.

## 13.1 Multiple Integrals

``` python
# Double integral of x*y over [0,1]×[0,1]
result, _ = integrate.dblquad(lambda y, x: x*y, 0, 1, 0, 1)
# 0.25
```

## 13.2 Taylor Series Expansion

`f(x) ≈ f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + ...`

**AI Uses:**
- Approximating complex functions locally
- Second-order optimization uses quadratic approximation:
  `L(θ) ≈ L(θ₀) + ∇L·(θ-θ₀) + ½(θ-θ₀)ᵀH(θ-θ₀)`

------------------------------------------------------------------------

# 🔷 PART III — PROBABILITY & STATISTICS

------------------------------------------------------------------------

# 14. Probability Fundamentals

> 🔍 **WHY?** ML is fundamentally about making predictions under uncertainty. Probability gives you the language and tools to reason about uncertainty, quantify confidence, and make optimal decisions. Without it, you can't understand classification, Bayesian methods, or even what a loss function means.
>
> 📖 **WHAT?** The mathematical framework for quantifying uncertainty — axioms, conditional probability, and Bayes' theorem.
>
> ⚙️ **HOW?** Bayes' theorem is used directly in Naive Bayes classifiers. Conditional probability appears in every probabilistic model. The base rate fallacy is a common interview question.

## 14.1 Core Axioms

1. `0 ≤ P(A) ≤ 1`
2. `P(Ω) = 1` (total probability)
3. `P(A ∪ B) = P(A) + P(B) - P(A ∩ B)`

## 14.2 Conditional Probability & Bayes' Theorem

```
P(A|B) = P(B|A) · P(A) / P(B)
```

**Bayes' Theorem is the foundation of:**
- Naive Bayes classifier
- Bayesian neural networks
- Bayesian optimization (hyperparameter tuning)
- Spam filters, medical diagnosis

``` python
# Bayes example: Disease testing
# P(Disease) = 0.01, P(Positive|Disease) = 0.99, P(Positive|No Disease) = 0.05
p_disease = 0.01
p_pos_given_disease = 0.99
p_pos_given_no_disease = 0.05

p_positive = p_pos_given_disease * p_disease + p_pos_given_no_disease * (1 - p_disease)
p_disease_given_pos = (p_pos_given_disease * p_disease) / p_positive
print(f"P(Disease|Positive) = {p_disease_given_pos:.4f}")  # ≈ 0.1667
```

**⚠️ Insight:** Even with a 99% accurate test, a positive result only means ~17% chance of disease when disease prevalence is 1%. This is **base rate fallacy**.

## 14.3 Independence vs Conditional Independence

- **Independent:** `P(A∩B) = P(A)·P(B)`
- **Conditionally Independent:** `P(A∩B|C) = P(A|C)·P(B|C)` — this is what Naive Bayes assumes.

------------------------------------------------------------------------

# 15. Random Variables & Distributions

> 🔍 **WHY?** Every ML model assumes data comes from some distribution. Classification assumes Bernoulli/Categorical. Regression assumes Normal. GANs learn to generate distributions. If you don't know distributions, you can't choose the right model or loss function.
>
> 📖 **WHAT?** Distributions describe the possible values a random variable can take and their probabilities.
>
> ⚙️ **HOW?** `scipy.stats` has every distribution you need. The Normal distribution alone powers weight initialization, noise injection, Gaussian processes, and the Central Limit Theorem.

## 15.1 Discrete Distributions

| Distribution | PMF | Mean | Variance | AI Use |
|-------------|-----|------|----------|--------|
| **Bernoulli(p)** | `P(X=1)=p` | p | p(1-p) | Binary classification |
| **Binomial(n,p)** | `C(n,k)pᵏ(1-p)ⁿ⁻ᵏ` | np | np(1-p) | Multiple trials |
| **Poisson(λ)** | `e⁻λλᵏ/k!` | λ | λ | Event counting, anomaly detection |
| **Categorical(p)** | `P(X=k) = pₖ` | — | — | Multi-class output |

## 15.2 Continuous Distributions

| Distribution | PDF | Mean | Variance | AI Use |
|-------------|-----|------|----------|--------|
| **Uniform(a,b)** | `1/(b-a)` | (a+b)/2 | (b-a)²/12 | Random initialization |
| **Normal(μ,σ²)** | `(1/σ√2π)e^(-(x-μ)²/2σ²)` | μ | σ² | Everything in ML |
| **Exponential(λ)** | `λe^(-λx)` | 1/λ | 1/λ² | Survival analysis |
| **Beta(α,β)** | — | α/(α+β) | — | Bayesian priors |

## 15.3 The Normal (Gaussian) Distribution — Most Important

``` python
import scipy.stats as stats

# Standard Normal: μ=0, σ=1
x = np.linspace(-4, 4, 100)
pdf = stats.norm.pdf(x, 0, 1)
cdf = stats.norm.cdf(1.96)  # ≈ 0.975

# 68-95-99.7 Rule
# 68% of data within 1σ, 95% within 2σ, 99.7% within 3σ
```

**Why Normal Distribution Dominates AI:**
- Central Limit Theorem: sum of many random variables → Normal
- Weight initialization (Xavier, He) uses Gaussian
- Gaussian noise in VAEs, diffusion models
- GP (Gaussian Process) regression assumes Gaussian priors

------------------------------------------------------------------------

# 16. Expectation, Variance & Moments

> 🔍 **WHY?** Expected value IS what your model predicts. Variance measures how uncertain that prediction is. Bias-variance tradeoff, confidence intervals, batch normalization — all built on these concepts. Covariance drives PCA and feature correlation analysis.
>
> 📖 **WHAT?** Summary statistics that capture the center (mean), spread (variance), and relationships (covariance) of distributions.
>
> ⚙️ **HOW?** `np.mean()`, `np.var()`, `np.cov()` — use these constantly in EDA. The covariance matrix is the input to PCA.

## 16.1 Expected Value (Mean)

```
E[X] = Σ xᵢP(xᵢ)              (discrete)
E[X] = ∫ x·f(x)dx              (continuous)
```

**Properties:**
- `E[aX + b] = aE[X] + b` (linearity)
- `E[X + Y] = E[X] + E[Y]` (always, even if dependent)
- `E[XY] = E[X]·E[Y]` only if independent

## 16.2 Variance & Standard Deviation

```
Var(X) = E[(X - μ)²] = E[X²] - (E[X])²
SD(X) = √Var(X)
```

**Properties:**
- `Var(aX + b) = a²Var(X)`
- `Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)`

## 16.3 Covariance & Correlation

```
Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)] = E[XY] - E[X]E[Y]
Corr(X,Y) = Cov(X,Y) / (σₓ · σᵧ)    ∈ [-1, 1]
```

``` python
cov_matrix = np.cov(data, rowvar=False)
corr_matrix = np.corrcoef(data, rowvar=False)
```

## 16.4 Higher Moments

| Moment | Measures | AI Use |
|--------|----------|--------|
| **Skewness** | Asymmetry | Feature engineering, detecting outliers |
| **Kurtosis** | Tail heaviness | Risk assessment, distribution shape |

------------------------------------------------------------------------

# 17. Maximum Likelihood Estimation (MLE)

> 🔍 **WHY?** When you train a neural network, you're doing MLE without knowing it. MSE loss = MLE with Gaussian assumption. Cross-entropy = MLE with Bernoulli/Categorical. Understanding MLE reveals WHY your loss function works and when to choose a different one.
>
> 📖 **WHAT?** A method for finding the parameter values that make the observed data most probable.
>
> ⚙️ **HOW?** In practice, you maximize the log-likelihood (= minimize negative log-likelihood = minimize your loss function). This connection is fundamental.

## 17.1 Concept

Find parameters θ that maximize the probability of observed data:

```
θ_MLE = argmax_θ P(data | θ)
       = argmax_θ Σ log P(xᵢ | θ)    (log-likelihood)
```

## 17.2 Example: MLE for Normal Distribution

``` python
# MLE estimates for Normal distribution
data = np.random.normal(5, 2, 1000)
mu_mle = np.mean(data)       # ≈ 5.0
sigma_mle = np.std(data)     # ≈ 2.0 (biased estimate)
```

## 17.3 Connection to Loss Functions

| Loss Function | Equivalent MLE Assumption |
|---------------|---------------------------|
| **MSE** | Data follows Normal distribution |
| **Cross-Entropy** | Data follows Bernoulli/Categorical |
| **MAE** | Data follows Laplace distribution |

**Key Insight:** When you train a neural network with cross-entropy loss, you're doing MLE!

------------------------------------------------------------------------

# 18. Information Theory

> 🔍 **WHY?** Cross-entropy loss (the most common classification loss) comes directly from information theory. KL divergence powers VAEs and policy optimization (PPO/TRPO). Entropy measures how "surprised" your model is. This field bridges probability and optimization.
>
> 📖 **WHAT?** The mathematics of information, uncertainty, and the difference between distributions.
>
> ⚙️ **HOW?** `scipy.stats.entropy()` computes entropy and KL divergence. Cross-entropy is built into every framework: `torch.nn.CrossEntropyLoss()`, `tf.keras.losses.CategoricalCrossentropy()`.

## 18.1 Entropy

Measures uncertainty/information content:

```
H(X) = -Σ P(xᵢ) log₂ P(xᵢ)
```

- Max entropy = uniform distribution (most uncertain)
- Min entropy = 0 (completely certain)

## 18.2 Cross-Entropy

```
H(P, Q) = -Σ P(xᵢ) log Q(xᵢ)
```

**This is the cross-entropy loss function in classification!**

## 18.3 KL Divergence

```
KL(P ‖ Q) = Σ P(xᵢ) log(P(xᵢ)/Q(xᵢ))
           = H(P, Q) - H(P)
```

- Always ≥ 0 (Gibbs' inequality)
- Not symmetric: `KL(P‖Q) ≠ KL(Q‖P)`

**AI Uses:**
- VAE loss = Reconstruction loss + KL divergence
- Policy gradient methods (TRPO, PPO) use KL to constrain updates
- Knowledge distillation

## 18.4 Mutual Information

```
I(X; Y) = H(X) - H(X|Y) = KL(P(X,Y) ‖ P(X)P(Y))
```

Measures how much knowing Y reduces uncertainty about X.

**Used in:** Feature selection, InfoGAN, representation learning.

------------------------------------------------------------------------

# 🔷 PART IV — OPTIMIZATION THEORY

------------------------------------------------------------------------

# 19. Convexity

> 🔍 **WHY?** If your loss function is convex, gradient descent finds the global minimum guaranteed. If it's not (like neural networks), you need tricks (momentum, Adam, warm restarts). Understanding convexity tells you when optimization is easy vs. hard and guides model choice.
>
> 📖 **WHAT?** A function is convex if any line segment between two points on its graph lies above the graph — one global minimum, no local minima traps.
>
> ⚙️ **HOW?** Check convexity via the Hessian (positive semi-definite = convex). Linear regression loss is convex → guaranteed solution. Neural network loss is not → use adaptive optimizers.

## 19.1 Convex Sets & Functions

A function f is **convex** if: `f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)` for all λ ∈ [0,1]

**Why it matters:** Convex optimization problems have a single global minimum — no local minima traps.

## 19.2 Convex Functions in ML

| Function | Convex? | Used In |
|----------|---------|---------|
| MSE Loss | ✅ Yes | Linear regression |
| Cross-Entropy | ✅ Yes (w.r.t. outputs) | Logistic regression |
| L1/L2 regularization | ✅ Yes | Lasso/Ridge |
| Neural network loss | ❌ No (in general) | Deep learning |

## 19.3 Trick: Convexity Check

- Compute the Hessian H
- If H is positive semi-definite everywhere → convex
- For 1D: `f''(x) ≥ 0` → convex

------------------------------------------------------------------------

# 20. Constrained Optimization

> 🔍 **WHY?** Many ML problems have constraints: SVM maximizes margin subject to correct classification. Fairness constraints in AI, resource budgets, probability constraints (must sum to 1). Lagrange multipliers convert constrained problems into unconstrained ones that gradient descent can solve.
>
> 📖 **WHAT?** Techniques for optimizing functions with equality or inequality constraints — Lagrangians, dual problems, KKT conditions.
>
> ⚙️ **HOW?** SVM's entire derivation uses KKT conditions. The dual formulation enables the kernel trick. In practice, `scipy.optimize.minimize(constraints=...)` handles this.

## 20.1 Lagrange Multipliers

Optimize `f(x)` subject to `g(x) = 0`:

```
L(x, λ) = f(x) - λ·g(x)
∇L = 0  →  ∇f = λ∇g
```

## 20.2 KKT Conditions (Inequality Constraints)

For `min f(x)` s.t. `gᵢ(x) ≤ 0`:

1. Stationarity: `∇f + Σλᵢ∇gᵢ = 0`
2. Primal feasibility: `gᵢ(x) ≤ 0`
3. Dual feasibility: `λᵢ ≥ 0`
4. Complementary slackness: `λᵢgᵢ(x) = 0`

**AI Use:** SVM optimization uses KKT conditions — support vectors are points where constraints are active.

------------------------------------------------------------------------

# 🔷 PART V — ADVANCED TOPICS FOR DEEP LEARNING

------------------------------------------------------------------------

# 21. Matrix Calculus for Neural Networks

> 🔍 **WHY?** This is where linear algebra meets calculus — the literal math inside `loss.backward()`. If you want to implement a custom layer, debug gradient issues, or understand why a model isn't training, you need to know how to differentiate matrix expressions.
>
> 📖 **WHAT?** Rules for differentiating expressions involving matrices and vectors — the backbone of backpropagation.
>
> ⚙️ **HOW?** Memorize the key derivatives below. Use the Matrix Cookbook (free PDF) as a reference. PyTorch autograd handles this automatically, but knowing the math helps you debug.

## 21.1 Key Derivatives

| Expression | Derivative |
|------------|------------|
| `∂(Wx)/∂x` | `W` |
| `∂(Wx)/∂W` | `xᵀ` (Jacobian sense) |
| `∂(xᵀAx)/∂x` | `(A + Aᵀ)x` |
| `∂(‖Wx - y‖²)/∂W` | `2(Wx - y)xᵀ` |

## 21.2 Backpropagation Derivation (Single Layer)

```
Forward:  z = Wx + b  →  a = σ(z)  →  L = loss(a, y)

Backward:
  dL/da = loss'(a, y)
  dL/dz = dL/da ⊙ σ'(z)        # ⊙ = element-wise
  dL/dW = dL/dz · xᵀ
  dL/db = dL/dz
  dL/dx = Wᵀ · dL/dz            # passed to previous layer
```

------------------------------------------------------------------------

# 22. Dimensionality Reduction Mathematics

> 🔍 **WHY?** Real-world data is high-dimensional (thousands of features). Dimensionality reduction removes redundancy, speeds up training, enables visualization, and fights the curse of dimensionality. PCA is used in almost every data preprocessing pipeline.
>
> 📖 **WHAT?** Techniques that project data into lower dimensions while preserving the most important structure (variance, distances, or topology).
>
> ⚙️ **HOW?** `sklearn.decomposition.PCA` for linear reduction. t-SNE/UMAP for visualization. Choose n_components by checking `explained_variance_ratio_`.

## 22.1 PCA Step-by-Step

1. Center data: `X̃ = X - mean(X)`
2. Compute covariance: `C = (1/n)X̃ᵀX̃`
3. Eigendecompose: `C = QΛQᵀ`
4. Project: `Z = X̃Q_k` (keep top-k eigenvectors)

``` python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
```

## 22.2 t-SNE & UMAP (Nonlinear)

- Convert pairwise distances to probabilities
- Minimize KL divergence between high-D and low-D distributions
- Used for visualization, not feature extraction

------------------------------------------------------------------------

# 23. Attention & Transformer Mathematics

> 🔍 **WHY?** Transformers power GPT, BERT, DALL-E, and essentially all state-of-the-art AI. The attention mechanism is a mathematical operation — understanding it lets you modify architectures, debug attention patterns, and innovate on the core mechanism.
>
> 📖 **WHAT?** Attention computes weighted sums where the weights are determined by query-key similarity. It's a soft dictionary lookup.
>
> ⚙️ **HOW?** The formula `Attention(Q,K,V) = softmax(QKᵀ/√d)V` is the heart of every transformer. Multi-head attention runs multiple of these in parallel.

## 23.1 Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
```

- `Q, K, V` are linear projections of input embeddings
- `√dₖ` scaling prevents dot products from growing too large (keeps softmax gradients healthy)

## 23.2 Multi-Head Attention

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)W_O
where headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)
```

## 23.3 Positional Encoding

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

**Why sin/cos?** Allows the model to learn relative positions via linear transformations.

------------------------------------------------------------------------

# 24. Loss Functions — Mathematical Foundations

> 🔍 **WHY?** The loss function defines WHAT the model learns. Choose the wrong loss → model optimizes the wrong thing. Understanding the math behind each loss tells you when to use MSE vs. cross-entropy, how to handle imbalanced classes (Focal loss), and how to train similarity models (Triplet loss).
>
> 📖 **WHAT?** Functions that measure how wrong a model's predictions are, providing the signal that drives learning.
>
> ⚙️ **HOW?** Most are built into frameworks. But knowing the formulas helps you: (1) pick the right one, (2) create custom losses, (3) debug training.

| Loss | Formula | When to Use |
|------|---------|-------------|
| **MSE** | `(1/n)Σ(yᵢ - ŷᵢ)²` | Regression |
| **MAE** | `(1/n)Σ|yᵢ - ŷᵢ|` | Robust regression |
| **Binary CE** | `-Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]` | Binary classification |
| **Categorical CE** | `-Σ yᵢ·log(ŷᵢ)` | Multi-class |
| **Hinge** | `max(0, 1 - y·ŷ)` | SVM |
| **Huber** | MSE if small error, MAE if large | Robust regression |
| **Focal** | `-αₜ(1-pₜ)ᵞ log(pₜ)` | Class imbalance |
| **Contrastive** | — | Siamese networks |
| **Triplet** | `max(0, d(a,p) - d(a,n) + margin)` | Face recognition |

------------------------------------------------------------------------

# 25. Regularization Mathematics

> 🔍 **WHY?** Without regularization, complex models memorize training data (overfit). L1/L2/Dropout are mathematical techniques that constrain model complexity. Understanding the math lets you choose the right type and strength of regularization.
>
> 📖 **WHAT?** Extra terms added to the loss function (or modifications to training) that penalize overly complex models.
>
> ⚙️ **HOW?** L2 = `weight_decay` parameter in optimizers. L1 requires explicit implementation. Dropout = `nn.Dropout(p=0.5)`. Elastic Net combines both.

## 25.1 L1 Regularization (Lasso)

```
L_total = L_data + λΣ|wᵢ|
```

- Promotes sparsity (drives weights to exactly 0)
- Feature selection effect

## 25.2 L2 Regularization (Ridge / Weight Decay)

```
L_total = L_data + λΣwᵢ²
```

- Shrinks weights towards 0 (but not exactly 0)
- Equivalent to Gaussian prior on weights
- In Adam optimizer: "decoupled weight decay" (AdamW)

## 25.3 Elastic Net

```
L_total = L_data + λ₁Σ|wᵢ| + λ₂Σwᵢ²
```

Combines benefits of L1 and L2.

## 25.4 Dropout (Probabilistic Regularization)

- Randomly zero out neurons with probability p during training
- At test time: scale by (1-p) or use inverted dropout
- Mathematically ≈ ensemble of 2ⁿ sub-networks

------------------------------------------------------------------------

# 🔷 PART VI — NUMERICAL METHODS & PRACTICAL MATH

------------------------------------------------------------------------

# 26. Numerical Stability

> 🔍 **WHY?** Your math can be correct but your code produces NaN or Inf because computers use finite-precision arithmetic. This is the #1 reason models suddenly produce garbage outputs. These tricks are the difference between a model that trains and one that explodes.
>
> 📖 **WHAT?** Techniques to handle overflow, underflow, and cancellation errors in floating-point computation.
>
> ⚙️ **HOW?** Log-sum-exp trick for softmax, working in log-space for probabilities, gradient clipping for exploding gradients. These are non-negotiable in production.

## 26.1 Common Pitfalls

| Problem | Cause | Solution |
|---------|-------|----------|
| **Overflow** | `e^(1000)` | Log-sum-exp trick |
| **Underflow** | `e^(-1000) ≈ 0` | Work in log-space |
| **Catastrophic cancellation** | Subtracting similar numbers | Rearrange formulas |
| **Vanishing gradients** | Deep networks + sigmoid | ReLU, BatchNorm, ResNets |
| **Exploding gradients** | Large weight products | Gradient clipping |

## 26.2 Log-Sum-Exp Trick

``` python
# WRONG (overflow):
# result = np.log(np.sum(np.exp(x)))

# CORRECT:
def logsumexp(x):
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c)))
```

## 26.3 Softmax Stability

``` python
def stable_softmax(x):
    x_shifted = x - np.max(x)  # prevent overflow
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)
```

------------------------------------------------------------------------

# 27. Monte Carlo Methods

> 🔍 **WHY?** Many integrals in AI are impossible to solve analytically (e.g., Bayesian posteriors, partition functions). Monte Carlo methods estimate them by sampling. This powers Bayesian inference, reinforcement learning (policy evaluation), and generative models.
>
> 📖 **WHAT?** Approximation methods that use random sampling to estimate quantities that are deterministic but hard to compute.
>
> ⚙️ **HOW?** `np.random` for basic MC. PyMC3/Stan for MCMC in Bayesian models. RL frameworks use MC rollouts for policy gradient estimation.

## 27.1 Basic Monte Carlo Estimation

Estimate `E[f(X)]` by sampling:

```
E[f(X)] ≈ (1/N) Σ f(xᵢ)    where xᵢ ~ P(X)
```

``` python
# Estimate π using Monte Carlo
N = 1_000_000
points = np.random.uniform(-1, 1, (N, 2))
inside = np.sum(np.linalg.norm(points, axis=1) <= 1)
pi_estimate = 4 * inside / N  # ≈ 3.1416
```

## 27.2 MCMC (Markov Chain Monte Carlo)

- Sample from complex distributions by constructing a Markov chain
- **Metropolis-Hastings:** Accept/reject proposals
- **Gibbs Sampling:** Sample one variable at a time

**AI Use:** Bayesian inference, sampling from posteriors, energy-based models.

------------------------------------------------------------------------

# 28. Graph Theory for AI

> 🔍 **WHY?** Social networks, molecules, knowledge graphs, recommendation systems — all graph-structured data. Graph Neural Networks (GNNs) are one of the fastest-growing areas in AI. Understanding graph math lets you work with non-Euclidean data that CNNs and RNNs can't handle.
>
> 📖 **WHAT?** The mathematics of nodes, edges, adjacency, and how information propagates through connected structures.
>
> ⚙️ **HOW?** PyTorch Geometric and DGL are the go-to frameworks. The adjacency matrix is the core data structure. GCN layers use normalized adjacency for message passing.

## 28.1 Key Concepts

| Concept | Definition | AI Application |
|---------|------------|----------------|
| **Node/Edge** | Entities and relationships | Knowledge graphs, social networks |
| **Adjacency Matrix** | A[i,j] = 1 if edge (i,j) | GNN input |
| **Degree** | Number of connections | Node importance |
| **Laplacian** | L = D - A | Spectral clustering, GCN |
| **Random Walk** | Traverse randomly | Node2Vec, DeepWalk |

## 28.2 Graph Neural Networks Math

```
Message Passing:
h_v^(k+1) = UPDATE(h_v^(k), AGGREGATE({h_u^(k) : u ∈ N(v)}))

GCN Layer:
H^(l+1) = σ(D̃⁻½ÃD̃⁻½ H^(l) W^(l))
where Ã = A + I (self-loops), D̃ = degree matrix of Ã
```

------------------------------------------------------------------------

# 🔷 PART VII — QUICK REFERENCE & FORMULAS

------------------------------------------------------------------------

# 29. Essential Formula Sheet

## Linear Algebra
```
‖v‖₂ = √(Σvᵢ²)                    L2 norm
cos(θ) = (u·v)/(‖u‖‖v‖)           Cosine similarity
A⁻¹ exists iff det(A) ≠ 0          Invertibility
Av = λv                             Eigenvalue equation
A = UΣVᵀ                           SVD
θ = (XᵀX)⁻¹Xᵀy                    Normal equation
```

## Calculus
```
∇f = [∂f/∂x₁, ..., ∂f/∂xₙ]       Gradient
θ ← θ - α∇L(θ)                    Gradient descent
f(x) ≈ f(a) + f'(a)(x-a) + ...    Taylor expansion
```

## Probability
```
P(A|B) = P(B|A)P(A)/P(B)           Bayes' theorem
E[X] = Σ xᵢP(xᵢ)                  Expected value
Var(X) = E[X²] - (E[X])²          Variance
H(X) = -Σ P(xᵢ)log P(xᵢ)         Entropy
KL(P‖Q) = Σ P(x)log(P(x)/Q(x))   KL divergence
```

## Optimization
```
L(x,λ) = f(x) - λg(x)             Lagrangian
∇L = 0 at optimum                  Stationarity
```

------------------------------------------------------------------------

# 30. Common Tricks & Best Practices

| # | Trick | When to Use |
|---|-------|-------------|
| 1 | Check tensor shapes after every operation | Always |
| 2 | Use log-probabilities instead of probabilities | Avoid underflow |
| 3 | Normalize features (zero mean, unit variance) | Before training any model |
| 4 | Use Xavier/He initialization | Neural network weights |
| 5 | Gradient clipping at 1.0 or 5.0 | RNNs, Transformers |
| 6 | Use `np.linalg.solve` not `np.linalg.inv` | Solving linear systems |
| 7 | Exploit matrix symmetry for 2x speedup | Covariance, kernel matrices |
| 8 | Batch matrix multiplications | GPU efficiency |
| 9 | Use einsum for complex tensor ops | Multi-dimensional contractions |
| 10 | Verify gradients numerically before trusting | Custom loss functions |

------------------------------------------------------------------------

# Final Mastery Checklist

## Linear Algebra
- [ ] Scalars, Vectors, Matrices, Tensors
- [ ] Dot Product, Norms (L1, L2, L∞)
- [ ] Matrix Multiplication, Transpose, Inverse
- [ ] Determinant, Trace, Rank
- [ ] Eigenvalues & Eigenvectors
- [ ] SVD, Cholesky, QR Decomposition
- [ ] Vector Spaces, Basis, Projections

## Calculus
- [ ] Limits, Continuity, Derivatives
- [ ] Chain Rule (= Backpropagation)
- [ ] Partial Derivatives & Gradients
- [ ] Jacobian & Hessian
- [ ] Gradient Descent & Variants
- [ ] Integration & Numerical Methods
- [ ] Taylor Series

## Probability & Statistics
- [ ] Bayes' Theorem & Conditional Probability
- [ ] Discrete & Continuous Distributions
- [ ] Expectation, Variance, Covariance
- [ ] Maximum Likelihood Estimation
- [ ] Information Theory (Entropy, KL, Mutual Info)

## Optimization
- [ ] Convexity & Convex Functions
- [ ] Lagrange Multipliers & KKT Conditions
- [ ] Constrained Optimization

## Advanced / Deep Learning Math
- [ ] Matrix Calculus & Backprop Derivation
- [ ] PCA & Dimensionality Reduction
- [ ] Attention & Transformer Math
- [ ] Loss Functions & Regularization
- [ ] Numerical Stability Tricks
- [ ] Monte Carlo Methods
- [ ] Graph Theory & GNNs

------------------------------------------------------------------------

> **You can now confidently use this guide for:**
> - University exams & viva
> - Machine Learning & Deep Learning courses
> - AI/ML interviews
> - Research paper reading
> - Building AI systems from scratch
> - Understanding any model's mathematical foundation
