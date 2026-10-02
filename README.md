# 🏗️ System of Linear Equations Solver
### *Numerical Linear Algebra via Custom Gauss-Jordan Elimination with Scale-Aware Pivoting*

**Davao Oriental State University**  
Faculty of Computing, Engineering and Technology — Department of Civil Engineering  

[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://mlgaussjordan.streamlit.app)
[![Python](https://img.shields.io/badge/Language-Python%203.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic%20Use-blue?style=for-the-badge)](#)

---

## 🌐 Live Web Application
You can try the platform directly in your browser without local installation:  
👉 **[mlgaussjordan.streamlit.app](https://mlgaussjordan.streamlit.app)**

---

## 👨‍🏫 Course Instructor
* **Engr. Sammuel Erespe**

## 👥 Proponents & Contributors
* **John Joshua D. Ilisan**
* **Darius Lape**
* **Briel Jan M. Lacia**

---

## 📌 Executive Summary
In engineering analysis and numerical computation, solving systems of linear equations ($Ax = b$) is a foundational requirement for structural analysis, network flow calculations, and frame modeling. This project provides a robust, transparent, first-principles implementation of **Gauss-Jordan Elimination** designed to solve general $n \times n$ systems of linear equations without relying on external linear algebra solvers or commercial libraries.

### Core Architecture & Components
1. **Custom Gauss-Jordan Elimination Engine (`gauss_jordan.py`):** Solves general square systems using scale-aware relative tolerance singularity detection, partial pivoting for numerical stability, and full audit logging of row transformations.
2. **Interactive Streamlit Web Dashboard (`app.py`):** Provides a visual, interactive interface allowing users to dynamically adjust matrix dimensions ($2 \times 2$ up to $8 \times 8$), manually input or modify matrix coefficients, compute exact solutions, and trace every intermediate row operation step-by-step.

---

## 📐 Mathematical Formulation

### 1. General System of Linear Equations
A system of $n$ linear equations with $n$ unknown variables is expressed in matrix notation as:

$$A x = b$$

$$\begin{bmatrix} 
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \dots & \vdots \\
a_{n1} & a_{n2} & \cdots & a_{nn}
\end{bmatrix}
\begin{bmatrix} 
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}
=
\begin{bmatrix} 
b_1 \\
b_2 \\
\vdots \\
b_n
\end{bmatrix}$$

The goal is to determine the unknown vector $x = [x_1, x_2, \dots, x_n]^T$ by reducing the augmented matrix $[A \mid b]$ into reduced row echelon form (RREF) $[I \mid x]$, where $I$ is the $n \times n$ identity matrix.

---

### 2. Scale-Aware Gauss-Jordan Elimination with Partial Pivoting
The solver in `gauss_jordan.py` implements the following systematic numerical steps:

#### Step 1: Augmented Matrix Setup
The coefficient matrix $A$ and constant vector $b$ are combined into an augmented matrix $M$ of dimension $n \times (n+1)$:

$$M = [A \mid b]$$

#### Step 2: Partial Pivoting
To maintain numerical stability and avoid division by zero (or small numbers that introduce roundoff errors), row swaps are performed at each column $k$:

$$\text{Find } r \text{ such that } \lvert M_{r,k} \rvert = \max_{i \ge k} \lvert M_{i,k} \rvert$$

If $r \neq k$, Row $k$ and Row $r$ are swapped.

#### Step 3: Scale-Aware Singularity Detection
Rather than relying on a rigid fixed tolerance, the solver establishes a relative singularity threshold based on the overall magnitude of the working matrix:

$$\text{threshold} = \max(\text{tol} \times \max(\lvert M \rvert), 10^{-14}) \quad \text{where } \text{tol} = 10^{-10}$$

If the pivot value $\lvert M_{k,k} \rvert < \text{threshold}$, the system is flagged as singular or ill-conditioned, and execution halts cleanly with an informative explanation.

#### Step 4: Row Normalization & Elementary Transformations
1. **Normalize Pivot Row:** Divide the entire pivot row by the diagonal entry $M_{k,k}$ so that $M_{k,k} = 1$:
   $$M_{k, j} \leftarrow \frac{M_{k, j}}{M_{k, k}} \quad \text{for } j = k, \dots, n$$

2. **Eliminate Non-Zero Column Entries:** Zero out all off-diagonal entries in column $k$ across all other rows $i \neq k$:
   $$M_{i, j} \leftarrow M_{i, j} - (M_{i, k} \times M_{k, j}) \quad \text{for all } i \neq k$$

Upon completing elimination across all $n$ columns, the solution vector is directly read from the final column:

$$x_i = M_{i, n+1} \quad \text{for } i = 1, 2, \dots, n$$

---

## 🔬 Key Technical Features

| Feature | Description | Engineering Benefit |
| :--- | :--- | :--- |
| **Partial Pivoting** | Interchanges rows to place the absolute largest available pivot on the diagonal. | Prevents division by zero and minimizes floating-point precision degradation. |
| **Scale-Aware Singularity Checking** | Evaluates pivots against a dynamic threshold based on matrix magnitude ($\text{tol} \times \max \lvert M \rvert$). | Prevents false singular rejections on high-magnitude matrices while catching ill-conditioned systems. |
| **Auditable Transformation Logging** | Records every intermediate matrix state and human-readable row operation formula. | Serves as an educational step-by-step calculation trace for verification. |
| **Flexible System Sizing** | Dynamically scales from $2 \times 2$ up to $8 \times 8$ linear systems via interactive UI controls. | Accommodates a wide variety of structural and linear algebra problems. |

---

## 🚀 Installation & How to Run

Copy and run the code block below in your terminal to set up the repository, virtual environment, dependencies, and launch the application:

```bash
# Clone the repository and enter directory
git clone <your-repository-url>
cd <repository-folder-name>

# Create virtual environment
python -m venv venv

# Activate virtual environment (Linux/macOS)
source venv/bin/activate
# Note for Windows users: run "venv\Scripts\activate" instead

# Install required dependencies
pip install -r requirements.txt

# Launch the Streamlit application
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

### 🧪 Standalone CLI Testing
To test the numerical solver directly in the command line without launching the UI:

```bash
python gauss_jordan.py
```

---

## 📂 Project Structure

```text
├── app.py           # Interactive Streamlit dashboard for matrix editing & visual solver logs
├── gauss_jordan.py  # Standalone Gauss-Jordan elimination algorithm with scale-aware pivoting
├── requirements.txt # Project dependencies (streamlit, numpy, pandas)
└── README.md        # Technical project documentation
```
