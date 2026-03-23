# Mathematics for AI & Data Science — Complete Learning Guide

> A comprehensive reference covering all essential math concepts for AI/ML — from vectors and matrices to eigenvalues, matrix decomposition, and their real-world applications. Written with beginner-friendly explanations, intuitive analogies, and runnable NumPy code.

---

## Table of Contents

1. [Vectors — The Building Blocks](#1-vectors--the-building-blocks)
2. [Matrices — The Transformation Machines](#2-matrices--the-transformation-machines)
3. [Matrix Arithmetic — Addition, Subtraction & Scalar Multiplication](#3-matrix-arithmetic--addition-subtraction--scalar-multiplication)
4. [Matrix Multiplication — Dot Product](#4-matrix-multiplication--dot-product)
5. [Matrix-Vector Multiplication](#5-matrix-vector-multiplication)
6. [Determinant — The "Volume Scale Factor"](#6-determinant--the-volume-scale-factor)
7. [Inverse Matrix — Undoing a Transformation](#7-inverse-matrix--undoing-a-transformation)
8. [Eigenvalues & Eigenvectors — The Special Directions](#8-eigenvalues--eigenvectors--the-special-directions)
9. [Matrix Decomposition — Breaking Matrices into Components](#9-matrix-decomposition--breaking-matrices-into-components)
10. [Singular Value Decomposition (SVD)](#10-singular-value-decomposition-svd)
11. [Other Important Decompositions](#11-other-important-decompositions)
12. [Norms — Measuring Size of Vectors & Matrices](#12-norms--measuring-size-of-vectors--matrices)
13. [Solving Linear Systems of Equations](#13-solving-linear-systems-of-equations)
14. [Special Matrices You Should Know](#14-special-matrices-you-should-know)
15. [How All of This Connects to AI/ML](#15-how-all-of-this-connects-to-aiml)
16. [Quick Reference — NumPy Cheat Sheet](#16-quick-reference--numpy-cheat-sheet)

---

# 1. Vectors — The Building Blocks

## What is a Vector?

A **vector** is simply an ordered list of numbers. Think of it as an arrow in space that has both **direction** and **magnitude** (length).

```
Real World Analogy:
   GPS coordinates [latitude, longitude] → a 2D vector
   RGB color [255, 128, 0]              → a 3D vector
   A student's scores [90, 85, 78, 92]  → a 4D vector
```

### Creating Vectors in NumPy

```python
import numpy as np

# 1D vector
v = np.array([1, 2, 3, 4])
print(v)          # [1 2 3 4]
print(v.shape)    # (4,)
print(v.ndim)     # 1
```

### Row Vector vs Column Vector

```python
# Row vector (1 × n)
row = np.array([[1, 2, 3]])
print(row.shape)   # (1, 3)

# Column vector (n × 1)
col = np.array([[1], [2], [3]])
print(col.shape)   # (3, 1)

# Convert between them
col = row.T        # Transpose: row → column
row = col.T        # Transpose: column → row
```

---

## Vector Operations

### Addition & Subtraction (element-wise)

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)   # [5, 7, 9]
print(a - b)   # [-3, -3, -3]
```

**Geometric meaning:** Adding vectors is like walking in one direction, then walking in another — the result is where you end up.

### Scalar Multiplication

```python
v = np.array([1, 2, 3])
print(2 * v)    # [2, 4, 6]   — stretched by 2
print(-1 * v)   # [-1, -2, -3] — flipped direction
print(0.5 * v)  # [0.5, 1.0, 1.5] — shrunk by half
```

**Geometric meaning:** Scalar multiplication stretches or shrinks the vector without changing its direction (unless the scalar is negative, which flips it).

### Dot Product (Inner Product)

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

result = np.dot(a, b)   # 1*4 + 2*5 + 3*6 = 32
print(result)            # 32
```

**What does the dot product tell you?**

| Dot Product Value | Meaning |
|---|---|
| Positive | Vectors point in roughly the **same** direction |
| Zero | Vectors are **perpendicular** (90° angle) |
| Negative | Vectors point in **opposite** directions |

**Real-world use:** In ML, the dot product measures **similarity** between two vectors (e.g., how similar are two documents, two user preferences, etc.).

### Magnitude (Length) of a Vector

```python
v = np.array([3, 4])

# Manual calculation: √(3² + 4²) = √(9+16) = √25 = 5
magnitude = np.linalg.norm(v)
print(magnitude)   # 5.0
```

### Unit Vector (Normalization)

A **unit vector** has magnitude = 1. Divide by the magnitude to normalize:

```python
v = np.array([3, 4])
unit_v = v / np.linalg.norm(v)
print(unit_v)                    # [0.6, 0.8]
print(np.linalg.norm(unit_v))   # 1.0  ← confirmed!
```

**Why normalize?** In ML, we often care about **direction** (what features matter) not **magnitude** (how big the numbers are). Normalization puts all vectors on equal footing.

### Cross Product (3D only)

```python
a = np.array([1, 0, 0])
b = np.array([0, 1, 0])

cross = np.cross(a, b)
print(cross)   # [0, 0, 1] — perpendicular to both a and b
```

**Geometric meaning:** The cross product gives a vector **perpendicular to both** input vectors. Its magnitude equals the area of the parallelogram they form.

---

# 2. Matrices — The Transformation Machines

## What is a Matrix?

A matrix is a **2D grid of numbers** arranged in rows and columns. Think of it as a spreadsheet, but more importantly — a matrix is a **transformation machine** that can rotate, scale, shear, or reflect vectors.

```
     Column 0  Column 1
Row 0 [  1        2   ]
Row 1 [  3        4   ]

This is a 2×2 matrix (2 rows, 2 columns)
```

### Creating Matrices in NumPy

```python
import numpy as np

# From a nested list
A = np.array([[1, 2],
              [3, 4]])
print(A.shape)   # (2, 2)

# Using helper functions
zeros = np.zeros((3, 3))       # 3×3 matrix of zeros
ones = np.ones((2, 4))         # 2×4 matrix of ones
identity = np.eye(3)           # 3×3 identity matrix
random = np.random.rand(3, 3)  # 3×3 random matrix
```

### Useful Creation Methods

```python
# From a range, reshaped
A = np.arange(1, 10).reshape(3, 3)
print(A)
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]

# Diagonal matrix
D = np.diag([1, 2, 3])
print(D)
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]
```

---

## Matrix Attributes

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

A.shape      # (2, 3) — 2 rows, 3 columns
A.ndim       # 2      — number of dimensions
A.size       # 6      — total number of elements
A.dtype      # int64  — data type
A.T          # Transpose (rows ↔ columns)
```

---

## Transpose — Flipping Rows and Columns

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(A.shape)   # (2, 3)

AT = A.T
print(AT)
# [[1 4]
#  [2 5]
#  [3 6]]
print(AT.shape)  # (3, 2) — rows and columns swapped
```

**Analogy:** Transpose is like rotating a spreadsheet 90° — first row becomes first column.

---

# 3. Matrix Arithmetic — Addition, Subtraction & Scalar Multiplication

## Addition & Subtraction

Matrices must have the **same shape** to be added or subtracted. The operation happens **element-by-element**.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Addition:\n", A + B)
# [[ 6  8]
#  [10 12]]

print("Subtraction:\n", A - B)
# [[-4 -4]
#  [-4 -4]]
```

### Visual Breakdown

```
    [1 2]     [5 6]     [1+5  2+6]     [6  8]
    [3 4]  +  [7 8]  =  [3+7  4+8]  =  [10 12]
```

---

## Scalar Multiplication

Multiply **every element** in the matrix by a single number (scalar):

```python
A = np.array([[1, 2], [3, 4]])
C = 2 * A
print(C)
# [[2 4]
#  [6 8]]
```

**Geometric meaning:** Scalar multiplication scales the entire transformation uniformly — like zooming in on a picture.

---

# 4. Matrix Multiplication — Dot Product

## The Rule

To multiply matrices $A$ and $B$:
- **Columns of A** must equal **rows of B**
- Result shape: (rows of A) × (columns of B)

```
A is (m × n) and B is (n × p) → Result is (m × p)
         ↑ these must match ↑
```

### NumPy Code

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])    # 2×2
B = np.array([[5, 6], [7, 8]])    # 2×2

# Three equivalent ways:
result = np.dot(A, B)
result = A @ B                     # Preferred (cleaner)
result = np.matmul(A, B)

print(result)
# [[19 22]
#  [43 50]]
```

### How It Works (Step by Step)

```
A @ B:

Row 0 of A dot Column 0 of B:  1*5 + 2*7 = 19
Row 0 of A dot Column 1 of B:  1*6 + 2*8 = 22
Row 1 of A dot Column 0 of B:  3*5 + 4*7 = 43
Row 1 of A dot Column 1 of B:  3*6 + 4*8 = 50

Result: [[19 22]
         [43 50]]
```

---

## Important: Matrix Multiplication is NOT Commutative

```python
A @ B  ≠  B @ A    # Order matters!
```

```python
print(A @ B)
# [[19 22]
#  [43 50]]

print(B @ A)
# [[23 34]
#  [31 46]]   ← Different!
```

**This is unlike scalar multiplication** where $3 \times 5 = 5 \times 3$.

---

## Element-wise Multiplication (Hadamard Product)

This is **different** from matrix multiplication — it simply multiplies corresponding elements:

```python
print(A * B)    # Element-wise (NOT matrix multiplication)
# [[ 5 12]
#  [21 32]]

# vs

print(A @ B)    # Matrix multiplication
# [[19 22]
#  [43 50]]
```

---

# 5. Matrix-Vector Multiplication

## What Happens When You Multiply a Matrix by a Vector?

The matrix **transforms** the vector into a new vector. This is the core idea of linear algebra — matrices are transformation functions.

```python
import numpy as np

M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
V = np.array([1, 0, 1])

result = np.dot(M, V)    # or M @ V
print(result)              # [ 4 10 16]
```

### Step by Step

```
Row 0 of M · V:  1*1 + 2*0 + 3*1 =  4
Row 1 of M · V:  4*1 + 5*0 + 6*1 = 10
Row 2 of M · V:  7*1 + 8*0 + 9*1 = 16

Result: [4, 10, 16]
```

### Geometric Interpretation

```
Before transformation:  V = [1, 0, 1]     (original position)
After transformation:   result = [4, 10, 16]  (new position)

The matrix M "moved" the vector to a new location in space.
```

---

# 6. Determinant — The "Volume Scale Factor"

## What is the Determinant?

The determinant is a **single number** that tells you how a matrix transformation **scales area (2D) or volume (3D)**.

```
Analogy:
   det = 2   → the transformation DOUBLES the area
   det = 0.5 → the transformation HALVES the area
   det = 0   → the transformation COLLAPSES everything to a lower dimension (flat!)
   det = -1  → the transformation FLIPS the orientation (mirror) with same area
```

---

## Computing the Determinant

### For a 2×2 Matrix

$$\text{det}\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$$

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])

det = np.linalg.det(A)
print(det)   # -2.0  (meaning: area is scaled by 2, and orientation is flipped)
```

### Manual Verification

```
det(A) = (1)(4) - (2)(3) = 4 - 6 = -2 ✓
```

### For Any Square Matrix

```python
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 0]])

det = np.linalg.det(M)
print(round(det))   # 27
```

> **Note:** `np.linalg.det()` returns a float due to floating-point arithmetic. Use `round()` or `int()` for clean results when you expect integers.

---

## What the Determinant Value Tells You

| Determinant Value | Meaning |
|---|---|
| `det > 0` | Transformation preserves orientation, scales area by `det` |
| `det < 0` | Transformation **flips** orientation (mirror), scales area by `|det|` |
| `det = 1` | Area is perfectly preserved (rotation only) |
| `det = 0` | Matrix is **singular** — it collapses space (no inverse exists!) |
| `|det|` large | Transformation greatly expands space |
| `|det|` small | Transformation greatly compresses space |

---

## Why Does det = 0 Matter?

A determinant of zero means:
- The matrix **squashes** everything into a lower dimension (e.g., 2D → 1D line, 3D → 2D plane)
- The matrix has **no inverse** (you can't undo the squashing)
- The matrix is called **singular** or **non-invertible**
- Multiple inputs map to the same output (information is lost)

```python
# Singular matrix — second row is double the first
singular = np.array([[1, 2],
                     [2, 4]])
print(np.linalg.det(singular))   # 0.0

# Trying to invert it will fail:
# np.linalg.inv(singular)  → raises LinAlgError!
```

---

# 7. Inverse Matrix — Undoing a Transformation

## What is the Inverse?

The inverse of a matrix $A$ is another matrix $A^{-1}$ such that:

$$A \cdot A^{-1} = A^{-1} \cdot A = I$$

where $I$ is the **identity matrix** (the "do nothing" matrix).

```
Analogy:
   If A represents "rotate 30° clockwise"
   Then A⁻¹ represents "rotate 30° counterclockwise"
   
   A × A⁻¹ = I (do nothing) — the rotation is perfectly undone
```

---

## Computing the Inverse

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])

inv = np.linalg.inv(A)
print(inv)
# [[-2.   1. ]
#  [ 1.5 -0.5]]
```

### Verification — Multiply A by its Inverse

```python
# Should produce the identity matrix
result = A @ inv
print(np.round(result))
# [[1. 0.]
#  [0. 1.]]   ← Identity matrix! ✓
```

---

## When Does the Inverse NOT Exist?

A matrix has **no inverse** when its determinant is zero (singular matrix):

```python
# Safe approach: always check first
A = np.array([[1, 2],
              [2, 4]])

det = np.linalg.det(A)
print(f"Determinant: {det}")   # 0.0

if det != 0:
    inv = np.linalg.inv(A)
else:
    print("Matrix is singular — no inverse exists!")
```

---

## For a 2×2 Matrix — The Formula

$$A^{-1} = \frac{1}{ad - bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$$

```python
# For A = [[1, 2], [3, 4]]:
# det = 1*4 - 2*3 = -2
# A⁻¹ = (1/-2) * [[4, -2], [-3, 1]]
#      = [[-2, 1], [1.5, -0.5]]  ✓
```

---

## Pseudo-Inverse (For Non-Square or Singular Matrices)

When a matrix is non-square or singular, use the **Moore-Penrose pseudo-inverse**:

```python
# Works for ANY matrix (even non-square)
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])    # 3×2 matrix — no regular inverse

pinv = np.linalg.pinv(A)
print(pinv.shape)   # (2, 3) — pseudo-inverse is transposed shape
```

**Used in:** Least-squares regression, when you have more equations than unknowns.

---

# 8. Eigenvalues & Eigenvectors — The Special Directions

## Start With What a Matrix Does

A matrix is a **transformation machine**. When you multiply a matrix by a vector, it **transforms** that vector — it can rotate, stretch, squish, or flip it.

```python
import numpy as np

A = np.array([[2, 1],
              [0, 3]])

v = np.array([1, 1])
result = A @ v
print(result)   # [3, 3]
```

The vector `[1, 1]` got transformed into `[3, 3]` — it changed both **direction and magnitude**.

---

## The Big Question

> **Are there special vectors that, when transformed by the matrix, DON'T change direction — they only get stretched or shrunk?**

**YES!** Those special vectors are called **eigenvectors**, and the stretching/shrinking factor is the **eigenvalue**.

---

## The Math

For a matrix $A$, if:

$$A \cdot v = \lambda \cdot v$$

Then:
- $v$ is an **eigenvector** (the special direction that doesn't rotate)
- $\lambda$ (lambda) is the **eigenvalue** (how much it stretches/shrinks)

```
Left side:  A @ v   → matrix transforms the vector (could rotate, scale, etc.)
Right side: λ * v   → just scales the vector (no rotation, only stretch/shrink)

If both sides are equal → the matrix's ONLY effect on this vector is scaling!
```

---

## Visual Intuition

Imagine you push a door:

```
Normal push (NOT an eigenvector):
   → Push at an angle → door ROTATES AND moves = direction changes ❌

Special push (IS an eigenvector):
   → Push along the hinge axis → door only SLIDES along that same line
   → Direction stays the same ✅, only the magnitude changes
```

---

## Concrete Example (No Code, Just Numbers)

```
Matrix A = [[2, 0],
            [0, 3]]

This matrix stretches:
   - x-direction by 2
   - y-direction by 3

Eigenvector 1: [1, 0]  (points along x-axis)
Eigenvalue 1:  2       (stretched by 2×)

   A @ [1, 0] = [2, 0] = 2 * [1, 0]  ✅ same direction, just scaled

Eigenvector 2: [0, 1]  (points along y-axis)
Eigenvalue 2:  3       (stretched by 3×)

   A @ [0, 1] = [0, 3] = 3 * [0, 1]  ✅ same direction, just scaled
```

---

## Computing with NumPy

```python
import numpy as np

A = np.array([[4, 2],
              [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

### Displaying Them Paired Together

```python
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]       # COLUMN i is the eigenvector
    lam = eigenvalues[i]         # corresponding eigenvalue
    print(f"λ{i+1} = {lam:.4f}  →  v{i+1} = {v}")
```

Output:
```
λ1 = 5.0000  →  v1 = [0.89442719 0.4472136 ]
λ2 = 2.0000  →  v2 = [-0.70710678  0.70710678]
```

---

## Verifying: A @ v = λ * v

```python
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    
    Av = A @ v          # Transform the vector
    lv = lam * v        # Just scale the vector
    
    print(f"A @ v{i+1}  = {Av}")
    print(f"λ{i+1} * v{i+1} = {lv}")
    print(f"Equal? {np.allclose(Av, lv)}")   # True!
    print()
```

---

## IMPORTANT: Eigenvectors Are Columns, Not Rows!

This is the #1 source of confusion:

```python
eigenvalues, eigenvectors = np.linalg.eig(A)

# ✅ CORRECT — use columns
first_eigenvector = eigenvectors[:, 0]

# ❌ WRONG — this is NOT an eigenvector
not_eigenvector = eigenvectors[0, :]
```

```
The eigenvectors matrix looks like:

     [v1_component1   v2_component1]
     [v1_component2   v2_component2]
       ↑ column 0       ↑ column 1
       eigenvector 1     eigenvector 2
```

---

## What Do Different Eigenvalues Mean?

| Eigenvalue ($\lambda$) | What happens to the eigenvector |
|---|---|
| $\lambda > 1$ | **Stretched** (gets longer) |
| $\lambda = 1$ | **Unchanged** (stays exactly the same) |
| $0 < \lambda < 1$ | **Shrunk** (gets shorter) |
| $\lambda = 0$ | **Collapsed to zero** (destroyed — that dimension is lost) |
| $\lambda < 0$ | **Flipped AND scaled** (reverses direction) |

```python
import numpy as np

# λ = 3 → stretches, λ = -2 → flips direction and stretches by 2x
A = np.array([[3, 0],
              [0, -2]])

eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)   # [3. -2.]
```

---

## For Symmetric Matrices — Use `eigh()`

If your matrix is symmetric ($A = A^T$), use `np.linalg.eigh()` instead:

```python
# Symmetric matrix
S = np.array([[4, 2],
              [2, 3]])

eigenvalues, eigenvectors = np.linalg.eigh(S)
print("Eigenvalues:", eigenvalues)    # Sorted in ascending order!
```

### Why `eigh()` is better for symmetric matrices:

| Feature | `eig()` | `eigh()` |
|---|---|---|
| Works on | Any square matrix | Symmetric matrices only |
| Speed | Slower | Faster |
| Eigenvalues | Unsorted, may be complex | Sorted, always real |
| Eigenvectors | May not be orthogonal | Guaranteed orthogonal |

---

## Real-World Analogy: Earthquake

```
Imagine shaking a building during an earthquake:

The building vibrates in many complex ways, BUT
there are certain natural directions it prefers to vibrate in.

Eigenvectors = the natural vibration directions
Eigenvalues  = how strongly it vibrates in each direction

Big eigenvalue   → dangerous vibration mode (resonance!)
Small eigenvalue → weak, safe vibration
```

---

## Where Eigenvalues & Eigenvectors Are Used

| Field | Eigenvector = | Eigenvalue = |
|---|---|---|
| **Google PageRank** | Web pages | Importance score |
| **PCA (Data Science)** | Principal directions in data | Variance captured |
| **Quantum Mechanics** | Possible states | Energy levels |
| **Structural Engineering** | Vibration modes | Vibration frequency |
| **Facial Recognition** | "Eigenfaces" (base patterns) | Importance of each pattern |
| **Stability Analysis** | System modes | Growth/decay rate |

---

# 9. Matrix Decomposition — Breaking Matrices into Components

## What is Matrix Decomposition?

Matrix decomposition (or factorization) means **breaking a matrix into simpler, meaningful component matrices** that, when multiplied back together, reconstruct the original matrix.

```
Analogy:
   Prime factorization:  60 = 2 × 2 × 3 × 5
   Matrix decomposition: A  = U × Σ × Vᵀ

   Just like prime factors reveal the "building blocks" of a number,
   matrix decomposition reveals the "building blocks" of a transformation.
```

---

## Why Decompose a Matrix?

| Reason | Explanation |
|---|---|
| **Simplification** | Easier to work with simpler components |
| **Reveal structure** | Discover hidden patterns in data |
| **Compression** | Keep only the important parts, discard the rest |
| **Numerical stability** | More reliable computation than working with the original |
| **Solving equations** | Faster and more efficient than direct methods |

---

## Decomposition vs Divide and Conquer

A common confusion — they look similar but are fundamentally different:

| | Divide & Conquer | Matrix Decomposition |
|---|---|---|
| **Purpose** | Solve a problem **faster** (efficiency) | Reveal **hidden structure** (understanding) |
| **Nature** | An **algorithm strategy** | A **mathematical factorization** |
| **Recursive?** | Yes, recursively splits | No, one-time factorization |
| **Example** | Merge sort, quicksort | SVD, LU, QR decomposition |
| **Output** | A solution (sorted array, etc.) | Component matrices with mathematical meaning |

> **Fun fact:** NumPy's internal implementation of SVD actually uses a divide and conquer algorithm to *compute* the decomposition efficiently. The algorithm (D&C) is different from the result (decomposition).

---

## Overview of Common Decompositions

| Decomposition | Formula | Matrix Requirement | Best For |
|---|---|---|---|
| **SVD** | $A = U\Sigma V^T$ | Any matrix | Most general, PCA, compression |
| **Eigendecomposition** | $A = PDP^{-1}$ | Square matrix | Understanding transformations |
| **LU** | $A = LU$ | Square matrix | Solving linear equations |
| **QR** | $A = QR$ | Any matrix | Least squares, numerical stability |
| **Cholesky** | $A = LL^T$ | Symmetric positive definite | Fast solving, simulations |

---

# 10. Singular Value Decomposition (SVD)

## The Most Important Decomposition

SVD breaks **any** matrix (even non-square!) into three matrices:

$$A = U \cdot \Sigma \cdot V^T$$

| Component | Name | Shape | What It Represents |
|---|---|---|---|
| $U$ | Left singular vectors | (m × m) | **Rotation** in output space |
| $\Sigma$ | Singular values | (m × n) diagonal | **Scaling** factors (importance) |
| $V^T$ | Right singular vectors | (n × n) | **Rotation** in input space |

```
The transformation A does this:
   1. Vᵀ rotates the input
   2. Σ stretches/shrinks along each axis
   3. U rotates the result into the output space

   A = Rotate → Scale → Rotate
```

---

## Computing SVD in NumPy

```python
import numpy as np

B = np.array([[5, 6],
              [7, 8]])

# Decompose
U, S, Vt = np.linalg.svd(B)

print("Left Singular Vectors (U):\n", U)
print("Singular Values (S):\n", S)
print("Right Singular Vectors (Vt):\n", Vt)
```

> **Note:** NumPy returns `S` as a 1D array of singular values (not a diagonal matrix). You need `np.diag(S)` to get the diagonal matrix form.

---

## Reconstructing the Original Matrix

```python
# Reconstruct B from its SVD components
B_reconstructed = U @ np.diag(S) @ Vt
print("Original:\n", B)
print("Reconstructed:\n", B_reconstructed)
# They are the same! ✓
```

---

## Singular Values — The Key Insight

The singular values in `S` are **sorted in descending order** — the first is the most important:

```python
print("Singular values:", S)
# e.g., [13.07, 0.17]
#        ↑ very important    ↑ barely matters
```

This ordering is what makes SVD powerful for **compression** and **dimensionality reduction** — you can throw away the small singular values and still approximate the original matrix.

---

## Low-Rank Approximation (Compression)

Keep only the top-$k$ singular values to approximate the matrix:

$$A \approx U_k \cdot \Sigma_k \cdot V_k^T$$

```python
import numpy as np

# Original matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

U, S, Vt = np.linalg.svd(A)

# Rank-1 approximation (keep only the largest singular value)
k = 1
A_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
print("Rank-1 approximation:\n", np.round(A_approx, 2))

# Rank-2 approximation (keep top 2 singular values)
k = 2
A_approx2 = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
print("Rank-2 approximation:\n", np.round(A_approx2, 2))
# Much closer to the original!
```

---

## Real-World Applications of SVD

### 1. Image Compression

An image is a matrix of pixel values. SVD lets you store only the most important components:

```python
# Conceptual example (for a grayscale image):
# image = large matrix of pixel values
# U, S, Vt = np.linalg.svd(image)
# 
# Keep top 50 singular values instead of all 1000:
# compressed = U[:, :50] @ np.diag(S[:50]) @ Vt[:50, :]
# 
# Result: Much smaller storage, image still looks good!
```

### 2. Recommendation Systems (Netflix, Spotify)

```
Users × Movies rating matrix:
   Decompose into: Users × Features × Features × Movies
   
   "Features" might represent: Action-loving, Comedy-fan, Drama-enthusiast
   SVD discovers these hidden preferences automatically!
```

### 3. PCA (Principal Component Analysis)

PCA uses SVD internally to find the directions of maximum variance in data. The eigenvectors of the covariance matrix (found via SVD) become the principal components.

### 4. Noise Reduction

Small singular values often represent noise. Discarding them "cleans" the data:

```python
# Original noisy data → SVD → keep top singular values → reconstruct
# Result: cleaner data with noise removed
```

### 5. Latent Semantic Analysis (NLP)

In text analysis, SVD finds hidden topics in a document-term matrix:

```
Documents × Words matrix → SVD → Topics discovered!
```

---

## SVD vs Eigendecomposition

| Feature | Eigendecomposition | SVD |
|---|---|---|
| Works on | Square matrices only | **Any** matrix (even non-square) |
| Formula | $A = PDP^{-1}$ | $A = U\Sigma V^T$ |
| Values | Eigenvalues (can be negative/complex) | Singular values (always non-negative real) |
| Vectors | May not be orthogonal | Always orthogonal |
| Existence | Not always (need n independent eigenvectors) | **Always** exists |

---

# 11. Other Important Decompositions

## LU Decomposition

Breaks a matrix into a **Lower** triangular and **Upper** triangular matrix:

$$A = L \cdot U$$

```python
from scipy.linalg import lu

A = np.array([[2, 1, 1],
              [4, 3, 3],
              [8, 7, 9]])

P, L, U = lu(A)   # P is a permutation matrix
print("Lower:\n", L)
print("Upper:\n", U)
print("Reconstructed:\n", P @ L @ U)   # Same as A
```

**Used for:** Efficiently solving systems of linear equations $Ax = b$, especially when you need to solve for many different $b$ values.

---

## QR Decomposition

Breaks a matrix into an **Orthogonal** matrix $Q$ and an **upper triangular** matrix $R$:

$$A = Q \cdot R$$

```python
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

Q, R = np.linalg.qr(A)
print("Q (orthogonal):\n", Q)
print("R (upper triangular):\n", R)
print("Reconstructed:\n", Q @ R)   # Same as A
```

**Used for:** Least-squares problems, numerical stability in computations.

---

## Cholesky Decomposition

For **symmetric positive definite** matrices only. Breaks into:

$$A = L \cdot L^T$$

```python
# Symmetric positive definite matrix
A = np.array([[4, 2],
              [2, 3]])

L = np.linalg.cholesky(A)
print("L:\n", L)
print("Reconstructed:\n", L @ L.T)   # Same as A
```

**Used for:** Machine learning (Gaussian processes), simulations, very fast equation solving (2× faster than LU).

---

# 12. Norms — Measuring Size of Vectors & Matrices

## What is a Norm?

A norm measures the **"size"** or **"length"** of a vector or matrix. Different norms measure size in different ways.

---

## Vector Norms

### L2 Norm (Euclidean Distance) — Most Common

$$\|v\|_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$

```python
v = np.array([3, 4])
print(np.linalg.norm(v))       # 5.0  (default is L2)
print(np.linalg.norm(v, 2))    # 5.0  (explicit L2)
```

### L1 Norm (Manhattan Distance)

$$\|v\|_1 = |v_1| + |v_2| + \cdots + |v_n|$$

```python
v = np.array([3, -4])
print(np.linalg.norm(v, 1))    # 7.0  (|3| + |-4|)
```

**Analogy:** L1 is like walking on a city grid (Manhattan) — you can only walk along streets, not diagonally.

### L∞ Norm (Max Norm)

$$\|v\|_\infty = \max(|v_1|, |v_2|, \ldots, |v_n|)$$

```python
v = np.array([3, -7, 2])
print(np.linalg.norm(v, np.inf))   # 7.0  (the largest absolute value)
```

---

## Matrix Norms

```python
A = np.array([[1, 2],
              [3, 4]])

# Frobenius norm (default for matrices) — like L2 for all elements
print(np.linalg.norm(A, 'fro'))   # √(1² + 2² + 3² + 4²) = √30

# Spectral norm (largest singular value)
print(np.linalg.norm(A, 2))
```

---

## When to Use Which Norm?

| Norm | Use Case |
|---|---|
| **L2** | Default distance, most ML algorithms |
| **L1** | Sparsity (Lasso regression), robust to outliers |
| **L∞** | When the worst-case component matters |
| **Frobenius** | Matrix "size", comparing matrices |

---

# 13. Solving Linear Systems of Equations

## The Problem

Given a system of equations:

$$2x + y = 5$$
$$x + 3y = 7$$

This can be written as $Ax = b$:

$$\begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 5 \\ 7 \end{pmatrix}$$

---

## Method 1: `np.linalg.solve()` (Preferred)

```python
import numpy as np

A = np.array([[2, 1],
              [1, 3]])
b = np.array([5, 7])

x = np.linalg.solve(A, b)
print(x)   # [1.6, 1.8]

# Verify: A @ x should equal b
print(A @ x)   # [5. 7.] ✓
```

### Why Not Use Inverse?

```python
# This works but is SLOWER and LESS NUMERICALLY STABLE:
x = np.linalg.inv(A) @ b

# Always prefer np.linalg.solve(A, b) instead!
```

| Method | Speed | Stability | Use? |
|---|---|---|---|
| `np.linalg.solve(A, b)` | Fast | Stable | ✅ Always prefer |
| `np.linalg.inv(A) @ b` | Slower | Less stable | ❌ Avoid |

---

## Method 2: Least Squares (When Exact Solution Doesn't Exist)

When you have **more equations than unknowns** (overdetermined system):

```python
# 3 equations, 2 unknowns — no exact solution
A = np.array([[1, 1],
              [2, 1],
              [1, 2]])
b = np.array([3, 4, 4])

# Least squares: find x that minimizes ||Ax - b||²
x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
print("Solution:", x)
print("Residuals:", residuals)   # How far off the solution is
```

**This is the foundation of linear regression!**

---

# 14. Special Matrices You Should Know

## Identity Matrix

The "do nothing" matrix — multiplying by it changes nothing:

$$A \cdot I = I \cdot A = A$$

```python
I = np.eye(3)
print(I)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(np.allclose(A @ I, A))   # True
```

**Analogy:** Identity matrix is like multiplying a number by 1 — nothing changes.

---

## Diagonal Matrix

Non-zero values only on the main diagonal:

```python
D = np.diag([2, 3, 4])
print(D)
# [[2 0 0]
#  [0 3 0]
#  [0 0 4]]

# Extracting the diagonal from a matrix:
A = np.array([[1, 2], [3, 4]])
print(np.diag(A))   # [1, 4]
```

**Properties:** Easy to invert, easy to compute determinant, eigenvalues are the diagonal entries.

---

## Symmetric Matrix

A matrix equals its transpose: $A = A^T$

```python
S = np.array([[1, 2, 3],
              [2, 5, 6],
              [3, 6, 9]])

print(np.allclose(S, S.T))   # True — it's symmetric!
```

**Important in ML:** Covariance matrices are always symmetric. Use `eigh()` for their eigendecomposition.

---

## Orthogonal Matrix

A matrix whose transpose equals its inverse: $A^T = A^{-1}$

```python
# Rotation matrix (30 degrees)
theta = np.radians(30)
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

# Verify: R^T @ R should equal Identity
print(np.round(R.T @ R))
# [[1. 0.]
#  [0. 1.]]   ← Identity! So R is orthogonal ✓
```

**Properties:**
- Determinant is always ±1
- Preserves lengths and angles (just rotates/reflects)
- Very numerically stable

---

## Sparse Matrix

A matrix where **most elements are zero**:

```python
from scipy.sparse import csr_matrix

# Dense (wasteful for mostly-zero data)
dense = np.array([[1, 0, 0, 0],
                  [0, 0, 2, 0],
                  [0, 0, 0, 0],
                  [3, 0, 0, 4]])

# Sparse (only stores non-zero values — much more efficient!)
sparse = csr_matrix(dense)
print(sparse)
```

**Used in:** NLP (document-term matrices), recommendation systems (user-item matrices), graph adjacency matrices.

---

# 15. How All of This Connects to AI/ML

## Linear Regression

$$\hat{y} = X \cdot w$$

- **Data** is a matrix $X$ (rows = samples, columns = features)
- **Weights** are a vector $w$ (what the model learns)
- **Prediction** is matrix-vector multiplication
- **Solution** uses least squares (or inverse): $w = (X^TX)^{-1}X^Ty$

---

## Principal Component Analysis (PCA)

1. Compute the **covariance matrix** of your data
2. Find its **eigenvalues and eigenvectors**
3. The eigenvectors with the **largest eigenvalues** are the principal components
4. Project data onto these directions → dimensionality reduction

```python
# PCA using eigendecomposition:
data = np.random.rand(100, 5)    # 100 samples, 5 features
data_centered = data - data.mean(axis=0)

cov_matrix = np.cov(data_centered.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort by eigenvalue (descending)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Project to 2D (keep top 2 components)
data_2d = data_centered @ eigenvectors[:, :2]
print(data_2d.shape)   # (100, 2) — reduced from 5D to 2D!
```

---

## Neural Networks

Every layer in a neural network is basically:

$$\text{output} = \text{activation}(W \cdot x + b)$$

- $W$ = weight **matrix** (learned during training)
- $x$ = input **vector**
- $b$ = bias **vector**
- Matrix multiplication is the **core computation** of every neural network

---

## Summary: What Each Concept Does in ML

| Math Concept | ML Application |
|---|---|
| **Vectors** | Data points, features, weights, embeddings |
| **Matrices** | Datasets, weight layers, transformations |
| **Dot Product** | Similarity, predictions, attention mechanism |
| **Determinant** | Check if solution exists, data rank |
| **Inverse** | Solving equations (linear regression) |
| **Eigenvalues/vectors** | PCA, understanding data directions |
| **SVD** | Compression, recommendation systems, PCA |
| **Norms** | Loss functions, regularization (L1/L2) |
| **Linear systems** | Regression, optimization |

---

# 16. Quick Reference — NumPy Cheat Sheet

## Creating

```python
import numpy as np

# Vectors
v = np.array([1, 2, 3])

# Matrices
A = np.array([[1, 2], [3, 4]])
np.zeros((3, 3))                  # All zeros
np.ones((2, 4))                   # All ones
np.eye(3)                         # Identity matrix
np.diag([1, 2, 3])               # Diagonal matrix
np.arange(1, 10).reshape(3, 3)   # Sequential, reshaped
np.random.rand(3, 3)             # Random [0, 1)
```

## Arithmetic

```python
A + B          # Addition
A - B          # Subtraction
2 * A          # Scalar multiplication
A * B          # Element-wise multiplication
A @ B          # Matrix multiplication
np.dot(A, B)   # Matrix multiplication (same as @)
A.T            # Transpose
```

## Linear Algebra

```python
np.linalg.det(A)          # Determinant
np.linalg.inv(A)          # Inverse
np.linalg.pinv(A)         # Pseudo-inverse
np.linalg.solve(A, b)     # Solve Ax = b
np.linalg.lstsq(A, b)     # Least squares solution
```

## Decompositions

```python
np.linalg.eig(A)          # Eigenvalues & eigenvectors
np.linalg.eigh(A)         # For symmetric matrices (faster, sorted)
np.linalg.svd(A)          # Singular Value Decomposition
np.linalg.qr(A)           # QR decomposition
np.linalg.cholesky(A)     # Cholesky decomposition
```

## Norms & Properties

```python
np.linalg.norm(v)         # L2 norm (default)
np.linalg.norm(v, 1)      # L1 norm
np.linalg.norm(v, np.inf) # L∞ norm
np.linalg.norm(A, 'fro')  # Frobenius norm (matrices)
np.linalg.matrix_rank(A)  # Rank of matrix
np.trace(A)                # Sum of diagonal elements
```

## Verification Tricks

```python
# Check if two arrays are almost equal (handles floating-point)
np.allclose(A, B)

# Check if matrix is symmetric
np.allclose(A, A.T)

# Check if orthogonal
np.allclose(A.T @ A, np.eye(len(A)))

# Round for display
np.round(A, 2)
```

---

> **Tip:** This guide covers the essential linear algebra needed for AI/ML. As you progress, topics like **tensor operations**, **gradient computation**, and **matrix calculus** will build directly on these foundations. Master these basics and everything else becomes much easier.
