# AI-Dev — AI Development Learning Repository

A structured, hands-on learning repository for building strong foundations in **Python**, **Data Science (NumPy/Pandas/Matplotlib)**, **Exploratory Data Analysis (EDA)**, and **Mathematics for AI/ML**.

> This repo is organized as a learning path: read the notes, run the scripts, and practice on included datasets.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [How to Run](#how-to-run)
- [Learning Path](#learning-path)
- [Datasets](#datasets)
- [What You’ll Learn](#what-youll-learn)
- [Notes](#notes)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This repository contains:

- **Topic-wise Python scripts** (Part1/Part2/… style progression)
- **Practice exercises** and mini EDA projects
- **Markdown guides** consolidating key concepts
- **CSV datasets** for reproducible analysis

---

## Repository Structure

> Names below match the current folder layout.

- **Data Science Essentials/**
  - **Numpy/** — arrays, vectorization, broadcasting, linear algebra
  - **Pandas/** — data wrangling, cleaning, aggregation, joins, statistics
  - **Matplotlib/** — basic plotting
  - **Practice/** — EDA workflows and practice problems
  - **EDA_Data_Cleaning_7-Step_Playbook.md** — an EDA + cleaning framework

- **Maths/**
  - **Linear Algebra/** — vectors, matrices, transformations
  - **Calculus/** — differentiation, optimization basics
  - **Mathematics_Complete_Master_Guide_for_AI.md** — consolidated math notes

- **Pythonic Code/**
  - **ListComprehension/** — pythonic patterns and best practices

- **Projects/**
  - **tasks.txt** — task list/objectives
  - **CMD/** — command-line/system automation utilities

---

## Getting Started

### Prerequisites

- Python **3.10+** (3.8+ should work for most scripts)
- (Optional) Jupyter Notebook / JupyterLab

### Recommended setup

```bash
# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install common data-science libraries
pip install -U pip
pip install numpy pandas matplotlib jupyter
```

---

## How to Run

### Run a Python script

```bash
python "Data Science Essentials/Pandas/Part1.py"
python "Data Science Essentials/Practice/Combined.py"
```

### Run notebooks (if present)

```bash
jupyter notebook
```

Then open:

- `Data Science Essentials/Numpy/OneShot/part_01.ipynb`
- `Data Science Essentials/Numpy/OneShot/part_02.ipynb`

---

## Learning Path

### Beginner

1. `Data Science Essentials/Numpy/Part1.py`
2. `Data Science Essentials/Pandas/Part1.py`
3. `Data Science Essentials/EDA_Data_Cleaning_7-Step_Playbook.md`
4. `Data Science Essentials/Practice/Pandas/Ex1.py` → `Ex4.py`

### Intermediate

1. `Data Science Essentials/Pandas/Statistical_Analysis_Guide.py`
2. `Data Science Essentials/Practice/Combined.py` and `Combined_Part2.py`
3. `Maths/Linear Algebra/Day_01.py`
4. `Pythonic Code/ListComprehension/Part1.py`

### Advanced

1. `Data Science Essentials/Practice/Numpy/AdvanceRevision.py`
2. `Data Science Essentials/Pandas/Statistical_Analysis_Solutions.py`
3. Extend the EDA playbook to new datasets / build your own templates
4. Add mini-projects under `Projects/`

---

## Datasets

Common datasets included in this repo:

- `student_performance.csv` — student scores (gender + 3 subjects)
- `Coffee_Stats.csv` — coffee brand sales across months
- `iris.csv` — classic ML dataset (150 rows, 3 classes)

---

## What You’ll Learn

- **NumPy**: arrays, indexing, broadcasting, vectorization, linear algebra basics
- **Pandas**: loading data, cleaning, filtering, groupby/aggregation, merge/join, reshaping
- **EDA**: univariate/bivariate analysis, correlations, outliers (IQR), summary writing
- **Visualization**: essential plots with Matplotlib
- **Maths for AI**: linear algebra + calculus fundamentals
- **Pythonic coding**: readable patterns and best practices

---

## Notes

- Scripts are meant to be **run independently**.
- CSV files are included for reproducibility.
- Markdown guides contain consolidated takeaways and references.

---

## Contributing

Contributions are welcome.

- Open an issue to propose improvements (typos, structure, more exercises)
- Prefer small, focused PRs
- Keep file/folder naming consistent with the existing structure

---

## License

No license file is currently included. If you want this repository to be reusable by others, add a `LICENSE` (common choices: MIT, Apache-2.0, GPL-3.0).
