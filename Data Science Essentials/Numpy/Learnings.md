# NumPy Array Indexing - Key Learnings

## 2D Array Indexing Basics

When working with 2D NumPy arrays, indexing follows the pattern: `array[row, column]`

### Example Array
```python
test_arr = np.array([[1,2,3],   # row 0
                     [4,5,6]])   # row 1
```

## Important Concepts

### 1. Single Parameter Indexing (First Dimension)
When you use only one index or slice, it operates on the **first dimension (rows)**:

```python
test_arr[1:4]  # Slices rows 1 to 4
# Returns: [[4,5,6]]
# Only returns row 1 because the array only has 2 rows (indices 0 and 1)
```

### 2. Out of Bounds Slicing Behavior
**Key Learning**: NumPy slicing does NOT raise an error when the end index is out of bounds!

```python
test_arr[1:4]   # No error, even though index 4 doesn't exist
test_arr[1:3]   # Returns the same result as above
test_arr[1:1000]  # Also valid! Returns from index 1 to the end
```

However, **direct indexing** (without slicing) WILL raise an error:
```python
test_arr[4]  # IndexError: index 4 is out of bounds
```

### 3. Two-Dimensional Indexing: `[row, column]`

#### Single Element Access
```python
test_arr[1, 2]  # Row 1, Column 2 → Returns: 6
# Row 1 is [4,5,6], and column index 2 gives us 6
```

#### Row with Column Slice
```python
test_arr[0, 1:3]  # Row 0, Columns 1 to 3 → Returns: [2,3]
# Row 0 is [1,2,3]
# Columns 1:3 means indices 1 and 2 → [2,3]
```

#### All Rows, Specific Column (Vertical Slice)
```python
test_arr[:, 1]  # All rows, Column 1 → Returns: [2,5]
# ':' means all rows
# Column 1 from row 0 → 2
# Column 1 from row 1 → 5
# Result: [2,5]
```

## Quick Reference

| Syntax | Meaning | Example Result |
|--------|---------|----------------|
| `arr[1]` | Get row 1 | `[4,5,6]` |
| `arr[1, 2]` | Row 1, Column 2 | `6` |
| `arr[0, 1:3]` | Row 0, Columns 1-2 | `[2,3]` |
| `arr[:, 1]` | All rows, Column 1 | `[2,5]` |
| `arr[1:4]` | Rows 1 to end (no error if 4 is out of bounds) | `[[4,5,6]]` |

## Key Takeaways

1. **First parameter = rows, Second parameter = columns**

2. **Slicing beyond array bounds doesn't error** - it just returns up to the last available element

3. **Use `:` to select all elements** in a dimension

4. **Direct indexing out of bounds DOES error**, but slicing doesn't

---

## Understanding the `axis` Parameter in Aggregate Functions

The `axis` parameter is used in functions like `np.sum()`, `np.mean()`, `np.max()`, etc., and can be confusing at first.

### Core Concept: **Axis specifies which dimension to collapse/eliminate**

Think of it this way:
- `axis=0` → Collapse **rows** → Operate **vertically** down columns

- `axis=1` → Collapse **columns** → Operate **horizontally** across rows

### Visual Example with 2x3 Array

```python
test_arr = np.array([[1, 2, 3],   # Row 0
                     [4, 5, 6]])   # Row 1
```

#### `axis=0` (Sum down each column)
```python
np.sum(test_arr, axis=0)  # [5, 7, 9]
```
```
[[1, 2, 3],
 [4, 5, 6]]
  ↓  ↓  ↓
 [5, 7, 9]

Column 0: 1+4 = 5
Column 1: 2+5 = 7
Column 2: 3+6 = 9
```
**Result:** One value per column → `[5, 7, 9]`

#### `axis=1` (Sum across each row)
```python
np.sum(test_arr, axis=1)  # [6, 15]
```
```
[[1, 2, 3] → 6,
 [4, 5, 6] → 15]

Row 0: 1+2+3 = 6
Row 1: 4+5+6 = 15
```
**Result:** One value per row → `[6, 15]`

#### No axis (Sum everything)
```python
np.sum(test_arr)  # 21
```
All elements: 1+2+3+4+5+6 = 21

### Example with 3x3 Matrix

```python
test_arr = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
```

#### `axis=0` (Vertical operation)
```python
np.sum(test_arr, axis=0)  # [12, 15, 18]
```
- Column 0: 1+4+7 = 12
- Column 1: 2+5+8 = 15
- Column 2: 3+6+9 = 18

#### `axis=1` (Horizontal operation)
```python
np.sum(test_arr, axis=1)  # [6, 15, 24]
```
- Row 0: 1+2+3 = 6
- Row 1: 4+5+6 = 15
- Row 2: 7+8+9 = 24

### Memory Trick

**"Axis N means: operate along that axis and make it disappear"**

- `axis=0` → The first dimension (rows) disappears → Left with column-wise results
- `axis=1` → The second dimension (columns) disappears → Left with row-wise results

### Works with All Aggregate Functions

This pattern applies to all aggregate functions:
```python
np.mean(arr, axis=0)    # Mean of each column
np.mean(arr, axis=1)    # Mean of each row
np.max(arr, axis=0)     # Max of each column
np.min(arr, axis=1)     # Min of each row
np.std(arr, axis=0)     # Std dev of each column
```

### Quick Reference Table

| Operation | What It Does | Result Shape |
|-----------|--------------|--------------|
| `np.sum(arr)` | Sum all elements | Single value |
| `np.sum(arr, axis=0)` | Sum down columns (↓) | Shape of columns |
| `np.sum(arr, axis=1)` | Sum across rows (→) | Shape of rows |

---

## Boolean Indexing and Fancy Indexing

### Boolean Indexing

Boolean indexing uses **True/False conditions** to filter and select elements from an array. You create a boolean mask (an array of True/False values) and use it to extract only the elements where the condition is True.

#### How It Works

```python
arr = np.array([1, 2, 3, 4, 5])

# Step 1: Create a boolean mask
mask = arr > 3
print(mask)  # [False, False, False, True, True]

# Step 2: Use the mask to filter
result = arr[mask]
print(result)  # [4, 5]

# Or do it in one line:
arr[arr > 3]  # [4, 5]
```

#### Common Use Cases

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Get even numbers
arr[arr % 2 == 0]  # [2, 4, 6, 8, 10]

# Get numbers between 3 and 7
arr[(arr >= 3) & (arr <= 7)]  # [3, 4, 5, 6, 7]
# Note: Use & for AND, | for OR, ~ for NOT, and always use parentheses!

# Get numbers less than 3 OR greater than 8
arr[(arr < 3) | (arr > 8)]  # [1, 2, 9, 10]
```

#### With 2D Arrays

```python
arr2d = np.array([[1, 2, 3], 
                  [4, 5, 6], 
                  [7, 8, 9]])

# Get all elements greater than 5
arr2d[arr2d > 5]  # [6, 7, 8, 9] (flattened result)

# Replace values conditionally
arr2d[arr2d > 5] = 0
# Result: [[1, 2, 3],
#          [4, 5, 0],
#          [0, 0, 0]]
```

### Fancy Indexing

Fancy indexing uses **arrays of indices** (or lists) to select multiple specific elements at once. Instead of using boolean conditions, you explicitly specify which positions you want.

#### How It Works

```python
arr = np.array([10, 20, 30, 40, 50])

# Select elements at positions 0, 2, and 4
indices = [0, 2, 4]
result = arr[indices]
print(result)  # [10, 30, 50]

# Or directly:
arr[[0, 2, 4]]  # [10, 30, 50]
```

#### 1D Array Examples

```python
arr = np.array([100, 200, 300, 400, 500, 600])

# Select specific positions
arr[[1, 3, 5]]  # [200, 400, 600]

# You can repeat indices
arr[[0, 0, 2, 2]]  # [100, 100, 300, 300]

# Order matters!
arr[[4, 2, 0]]  # [500, 300, 100] - reverse selection
```

#### 2D Array Examples

```python
arr2d = np.array([[1,  2,  3,  4],
                  [5,  6,  7,  8],
                  [9, 10, 11, 12]])

# Select specific rows
arr2d[[0, 2]]  # Rows 0 and 2
# Result: [[1, 2, 3, 4],
#          [9, 10, 11, 12]]

# Select specific elements using row and column indices
rows = [0, 1, 2]
cols = [1, 2, 3]
arr2d[rows, cols]  # [2, 7, 12] - gets (0,1), (1,2), (2,3)
```

### Key Differences

| Feature | Boolean Indexing | Fancy Indexing |
|---------|-----------------|----------------|
| **Uses** | Conditions (True/False) | Index positions (integers) |
| **Example** | `arr[arr > 5]` | `arr[[0, 2, 4]]` |
| **Selection** | Based on values | Based on positions |
| **Result** | Elements matching condition | Elements at specified indices |
| **Can repeat?** | No | Yes - can select same index multiple times |

---

## Understanding `argmin` and `argmax`

`argmin` and `argmax` return the **index (position)** of the minimum or maximum value, not the value itself.

### `argmin` - Index of Minimum Value

```python
arr = np.array([5, 2, 8, 1, 9])

np.min(arr)     # 1 (the minimum value)
np.argmin(arr)  # 3 (the index where 1 is located)
```

Position: `[5, 2, 8, 1, 9]`  
Index:     `[0, 1, 2, 3, 4]`  
→ The minimum value `1` is at index `3`

### `argmax` - Index of Maximum Value

```python
arr = np.array([5, 2, 8, 1, 9])

np.max(arr)     # 9 (the maximum value)
np.argmax(arr)  # 4 (the index where 9 is located)
```

### With 2D Arrays

```python
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# Without axis - flattened index
np.argmin(arr2d)  # 0 (first element in flattened array)
np.argmax(arr2d)  # 8 (last element in flattened array)

# With axis=0 - index within each column
np.argmin(arr2d, axis=0)  # [0, 0, 0] (min in each column is in row 0)
np.argmax(arr2d, axis=0)  # [2, 2, 2] (max in each column is in row 2)

# With axis=1 - index within each row
np.argmin(arr2d, axis=1)  # [0, 0, 0] (min in each row is in column 0)
np.argmax(arr2d, axis=1)  # [2, 2, 2] (max in each row is in column 2)
```

### Practical Use Case

```python
scores = np.array([78, 92, 85, 67, 95, 88])

best_index = np.argmax(scores)   # 4
worst_index = np.argmin(scores)  # 3

print(f"Best score: {scores[best_index]} at position {best_index}")   # 95 at position 4
print(f"Worst score: {scores[worst_index]} at position {worst_index}") # 67 at position 3
```

**Key Point:** Use `min/max` to get the **value**, use `argmin/argmax` to get the **position**.

---

## Array Normalization

Normalizing (or standardizing) an array means **transforming the data to a standard scale** so that all values are comparable and centered around a common reference point.

### Why Normalize?

- **Machine Learning**: Algorithms work better when features are on similar scales
- **Comparison**: Compare data from different units or ranges
- **Faster Convergence**: Neural networks train faster with normalized data

### Common Normalization Methods

#### 1. Z-Score Normalization (Standardization)
Transforms data to have **mean = 0** and **standard deviation = 1**

```python
arr = np.array([1, 2, 3, 4, 5])

normalized = (arr - np.mean(arr)) / np.std(arr)
print(normalized)  # [-1.41421356, -0.70710678,  0., 0.70710678,  1.41421356]

# Verify:
print(np.mean(normalized))  # ~0 (close to zero)
print(np.std(normalized))   # 1.0
```

**Formula:** `(x - mean) / std`

**Visual:**
```
Original:   [1,  2,  3,  4,  5]     (mean=3, std≈1.41)
Normalized: [-1.41, -0.71, 0, 0.71, 1.41]  (mean=0, std=1)
```

#### 2. Min-Max Normalization (Scaling to 0-1)
Scales data to a range between **0 and 1**

```python
arr = np.array([10, 20, 30, 40, 50])

normalized = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
print(normalized)  # [0.0, 0.25, 0.5, 0.75, 1.0]
```

**Formula:** `(x - min) / (max - min)`

**How It Works:**
```
Original array: [10, 20, 30, 40, 50]

Subtract minimum (10): [0, 10, 20, 30, 40]
Divide by range (50-10=40): [0/40, 10/40, 20/40, 30/40, 40/40]
Result: [0, 0.25, 0.5, 0.75, 1.0]
```

The smallest value becomes **0**, the largest becomes **1**, and everything else scales proportionally in between.

### Real-World Example

```python
# House prices (in thousands) and square footage
prices = np.array([200, 350, 280, 420, 310])    # Range: 200-420
sqft = np.array([1500, 2200, 1800, 2500, 2000]) # Range: 1500-2500

# Without normalization - prices dominate because of larger scale
print("Prices:", prices)
print("Sqft:", sqft)

# With Z-score normalization - both on same scale
prices_norm = (prices - np.mean(prices)) / np.std(prices)
sqft_norm = (sqft - np.mean(sqft)) / np.std(sqft)

print("\nNormalized prices:", prices_norm)
print("Normalized sqft:", sqft_norm)
# Now both have mean≈0 and std=1, making them comparable!
```

### Quick Comparison

| Method | Formula | Result Range | Use Case |
|--------|---------|--------------|----------|
| **Z-Score** | `(x - mean) / std` | Mean=0, Std=1 | Most ML algorithms |
| **Min-Max** | `(x - min) / (max - min)` | 0 to 1 | Neural networks, image data |

**Key Takeaway:** Normalization transforms data to a standard scale without changing the distribution's shape, making it easier to compare and process.

---

## Random Seeds in NumPy

**"Setting up seeds"** means setting a **random seed** - a starting point for the random number generator.

### The Core Problem and Solution

NumPy's random functions generate "pseudo-random" numbers. Setting a seed makes the randomness **reproducible** - you get the same "random" numbers every time you run the code.

### How It Works

```python
import numpy as np

# Without seed - different results each time
print(np.random.rand(3))  # [0.123, 0.456, 0.789]
print(np.random.rand(3))  # [0.234, 0.567, 0.891] - different!

# With seed - same results every time
np.random.seed(42)
print(np.random.rand(3))  # [0.374, 0.950, 0.731]

np.random.seed(42)  # Reset to same seed
print(np.random.rand(3))  # [0.374, 0.950, 0.731] - same as before!
```

### Understanding the Seed Value

The number you put inside `np.random.seed()` is just an **identifier** or **starting point** for the random number generator. It has **no inherent meaning** - it's arbitrary.

```python
np.random.seed(42)   # One sequence
np.random.seed(100)  # Different sequence
np.random.seed(999)  # Another different sequence
```

Each seed produces a **different but predictable sequence** of "random" numbers. The seed value itself doesn't matter - only consistency does. Same seed = Same sequence.

**Why is 42 popular?** Cultural reference from "The Hitchhiker's Guide to the Galaxy" - no mathematical significance!

### The Paradox: "Why use random if we're fixing it?"

This seems contradictory, but we need **two things at the same time**:
1. **Randomness** - Unpredictable, varied data
2. **Reproducibility** - Ability to recreate the EXACT same randomness when needed

**Think of it like a shuffled deck of cards:**

#### Without Seed (Real Casino)
```python
np.random.shuffle(deck)  # Shuffle 1: King, 7, Ace, 3...
np.random.shuffle(deck)  # Shuffle 2: 5, Queen, 2, Jack... (completely different)
```
You **can't replay** the same game.

#### With Seed (Recorded Game)
```python
np.random.seed(42)
np.random.shuffle(deck)  # King, 7, Ace, 3... (random-looking)

np.random.seed(42)
np.random.shuffle(deck)  # King, 7, Ace, 3... (same order!)
```
The deck is **still randomly shuffled**, but you **can replay** the exact same shuffle.

### Real-World Use Cases

#### 1. Debugging
```python
# With seed - bug appears every time, easy to fix
np.random.seed(42)
def buggy_function():
    arr = np.random.randint(0, 100, 10)
    return arr[arr > 95]  # Now crashes consistently, you can debug it
```

#### 2. Team Collaboration
```python
# You send code to your colleague
np.random.seed(123)
experiment_data = np.random.normal(0, 1, size=1000)
result = analyze(experiment_data)
print(result)  # 0.567

# Your colleague runs THE EXACT SAME CODE with same seed
np.random.seed(123)
experiment_data = np.random.normal(0, 1, size=1000)
result = analyze(experiment_data)
print(result)  # 0.567 - Same result! Can verify your findings
```

#### 3. Machine Learning
```python
# Training Phase
np.random.seed(42)
training_data = np.random.rand(1000, 20)
# Train model - Accuracy: 85%

# 6 months later: Retrain with EXACT same data
np.random.seed(42)
training_data = np.random.rand(1000, 20)
# EXACT same data - can reproduce the 85% accuracy!
```

### Why Seeds Get "Consumed"

The seed creates a **sequence** (like a playlist):

```python
np.random.seed(42)
print(np.random.rand(3))  # [0.374, 0.950, 0.731] - Takes first 3
print(np.random.rand(3))  # [0.598, 0.156, 0.155] - Takes next 3
print(np.random.rand(3))  # [0.058, 0.866, 0.601] - Continues further

# Reset to beginning
np.random.seed(42)
print(np.random.rand(3))  # [0.374, 0.950, 0.731] - Back to start!
```

**Think of it like a tape player:**
- `np.random.seed(42)` = Rewind tape to beginning
- Each `np.random.rand()` = Play next section
- Reset seed = Rewind back to start

### When to Use Seeds

| Situation | Use Seed? | Why? |
|-----------|-----------|------|
| **Development/Testing** | ✅ Yes | Need to reproduce bugs |
| **Machine Learning Training** | ✅ Yes | Compare model versions fairly |
| **Research Papers** | ✅ Yes | Others can verify your work |
| **Unit Tests** | ✅ Yes | Tests should be deterministic |
| **Production/Real Users** | ❌ No | Want true unpredictability |
| **Games (for real players)** | ❌ No | Should be fair and unpredictable |
| **Security/Cryptography** | ❌ No | Must be truly random |

### Key Insights

**"Random" has two meanings:**
1. **Random Distribution** - Numbers are scattered/varied (not 1,2,3,4,5...)
2. **Random Unpredictability** - Can't predict what comes next

**Seeds preserve #1 (distribution) but remove #2 (unpredictability)**

### The Special Functionality

**Seeds give you CONTROL over randomness:**
- ✅ **Reproducibility**: Same results every time
- ✅ **Debugging**: Consistent errors you can fix
- ✅ **Verification**: Others can confirm your work
- ✅ **Fairness**: Same test conditions for comparing models
- ✅ **Reusability**: Can reuse the EXACT same random set in future

**Analogy**: Seeds are like a "bookmark" for a specific random sequence. Same seed = Same random data, every single time. You can say: *"Give me that random set I used before"* - even if it was months ago!

**Bottom Line**: Seeds don't make numbers "not random" - they make randomness **controllable and reproducible**. It's like having a remote control for randomness with Play and Rewind buttons.
