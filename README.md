# Concrete Compressive Strength Predictor & Linear Systems Solver
### *Supervised Machine Learning via Custom Gauss-Jordan Elimination with Scale-Aware Pivoting*

**Davao Oriental State University**  
Faculty of Computing Engineering and Technology  — Department of Civil Engineering  
*Project Defense Documentation & Technical Implementation (September 2026)*

[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Language-Python%203.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic%20Use-blue?style=for-the-badge)](#)

---
## 👥 Instructor
* **Engr. Samuel Erespe**
  
## 👥 Proponents & Contributors
* **John Joshua D. Ilisan**
* **Darius Lape**
* **Briel Jan M. Lacia**

---

## 📌 Executive Summary
In civil engineering practice, evaluating the 28-day compressive strength of concrete conventionally requires casting standard cylinders and curing them in water tanks for nearly a month before conducting destructive testing on a Universal Testing Machine (UTM). This project demonstrates how **Machine Learning** can estimate concrete compressive strength directly from batch proportions from first principles. Rather than relying on commercial, closed-source "black box" libraries such as Scikit-Learn, all numerical and machine learning foundations are implemented from scratch:

1. **Custom Gauss-Jordan Elimination Solver (`gauss_jordan.py`):** Solves general $n \times n$ systems using partial pivoting, scale-aware relative tolerance singularity detection, and an auditable step-by-step transformation logging pipeline.
2. **Multiple Linear Regression Engine (`linear_regression.py`):** Solves Ordinary Least Squares (OLS) via the Normal Equations. Features custom matrix inversion of $X^T X$ (solving unit column vectors $e_i$ via Gauss-Jordan), coefficient hypothesis testing ($t$-statistics and $p$-values), and Leave-One-Out Cross-Validation (LOOCV).
3. **Interactive Streamlit Web Dashboard (`app.py`):** Includes a general matrix solver (Tab 1) and an applied civil engineering predictor (Tab 2) equipped with multicollinearity screening (VIF), interactive data editing, in-sample vs. held-out validation metrics, and mix design simulation sliders.

---

## 📐 Mathematical Formulation

### 1. The Concrete Compressive Strength Model
Concrete compressive strength is modeled as a multi-variable linear function of three primary mix parameters:

$$\text{Strength} = \beta_0 + \beta_1(\text{Cement}) + \beta_2(w/c) + \beta_3(\text{Age})$$

| Parameter | Type | Unit | Engineering Role & Behavior |
| :--- | :--- | :--- | :--- |
| **$\beta_0$** | Unknown | $\text{MPa}$ | **Baseline Intercept**: Mathematical datum of the regression hyperplane. |
| **$x_1$ / $\beta_1$** | Feature | $\text{kg/m}^3$ | **Cement Content**: Primary binder. Positive effect ($\beta_1 > 0$) as binder paste densifies. |
| **$x_2$ / $\beta_2$** | Feature | Decimal | **Water-Cement Ratio ($w/c$)**: Governed by Abrams' Law. Negative effect ($\beta_2 < 0$) due to capillary voids left by unreacted water. |
| **$x_3$ / $\beta_3$** | Feature | Days | **Curing Age**: Hydration progression over time. Positive effect ($\beta_3 > 0$) as Calcium Silicate Hydrates (C-S-H) form. |
| **$y$** | Target | $\text{MPa}$ | **Compressive Strength**: Failure stress under UTM compression. |

---

### 2. Ordinary Least Squares via the Normal Equations
Because the number of cylinder test batches ($N$) exceeds the number of regression parameters ($k = 4$), the experimental system is rectangular and overdetermined ($X\beta \approx y$). To minimize the sum of squared prediction errors $\sum (y - \hat{y})^2$, calculus yields the **Normal Equations**:

$$(X^T X)\beta = X^T y \quad \iff \quad A\beta = b$$

* $X$ is the $(N \times 4)$ design matrix with a leading bias column of $1$s.
* $X^T X$ compresses the $N$ observations into a symmetric, square $(4 \times 4)$ coefficient matrix ($A$).
* $X^T y$ compresses target values into a $(4 \times 1)$ column vector ($b$).
* The unknown weight vector $\beta = [\beta_0, \beta_1, \beta_2, \beta_3]^T$ is solved directly by passing $[(X^T X) \mid (X^T y)]$ into our custom Gauss-Jordan solver.

---

### 3. Scale-Aware Gauss-Jordan Elimination
The custom solver in `gauss_jordan.py` applies **Partial Pivoting** to maintain numerical stability by swapping rows to position the maximum absolute pivot value on the diagonal before row normalization.

* **Scale-Aware Singularity Check:** Because civil engineering data combines variables of drastically different scales (cement content $\sim 300\text{ kg/m}^3$ vs. $w/c$ ratio $\sim 0.4$), fixed absolute tolerances fail. The solver dynamically establishes a relative singularity threshold:

$$\text{threshold} = \max(\text{tol} \times \max(\vert M \vert), 10^{-14}) \quad \text{where } \text{tol} = 10^{-10}$$

If a pivot falls below this relative threshold, the system is rejected as singular or ill-conditioned.

---

### 4. Statistical Inference & Custom Matrix Inversion
To evaluate whether regression coefficients are statistically meaningful or merely artifacts of small-sample noise, the standard errors of $\beta$ are computed:

$$\text{SE}(\beta) = \sqrt{\sigma^2 \cdot \text{diag}((X^T X)^{-1})}, \quad \text{where } \sigma^2 = \frac{\sum (y - \hat{y})^2}{N - p}$$

* **No Library Inversion:** Rather than calling `np.linalg.inv`, the system computes $(X^T X)^{-1}$ using the custom Gauss-Jordan engine by solving $(X^T X)z_i = e_i$ for each standard unit basis vector $e_i$.
* **Hypothesis Testing:** Two-tailed $t$-statistics ($t = \beta / \text{SE}$) and $p$-values are evaluated across $N - p$ degrees of freedom.

---

### 5. Multicollinearity & Honest Cross-Validation
* **Variance Inflation Factor (VIF):** Computed prior to training to ensure that physical interdependencies between cement content and water ratio do not cause collinearity breakdown. Each predictor is regressed against all other predictors using our custom regression class.
* **Leave-One-Out Cross-Validation (LOOCV):** With small batch samples, in-sample $R^2$ is optimistically biased. The system iteratively withholds each observation, retrains the model on the remaining $N-1$ specimens, and predicts the held-out point. Unstable or ill-conditioned folds are flagged and reported.

---

## 🔬 Methodological Improvements

| Engineering Aspect | Initial Implementation | Revised & Verified Implementation |
| :--- | :--- | :--- |
| **Singularity Threshold** | Fixed absolute tolerance ($10^{-12}$). | Scale-aware relative tolerance ($\text{tol} \times \max(\vert M \vert)$) preventing false passes on ill-conditioned systems. |
| **Validation Rigor** | In-sample training fit labeled as "validation". | Added full **Leave-One-Out Cross-Validation (LOOCV)** as an independent accuracy check. |
| **Statistical Inference** | Point estimates reported without uncertainty bounds. | Standard errors, $t$-statistics, degrees of freedom, and $p$-values derived via custom $(X^T X)^{-1}$. |
| **Multicollinearity Checks** | Not assessed. | Automated Variance Inflation Factor (VIF) and pairwise correlation matrix computed pre-training. |
| **Accuracy Metric** | Ad-hoc per-batch percentage. | Standardized $R^2$, $\text{MAE}$, $\text{RMSE}$, and $100\% - \text{MAPE}$. |
| **Solver Correctness** | Asserted without external cross-check. | Independently verified against `numpy.linalg.solve` across multiple test matrices to 6 decimal places. |

---

## 📊 Empirical Results & Performance Benchmarks

### 1. In-Sample Fit vs. Cross-Validated Generalization
Benchmarked using standard laboratory mix test observations ($N = 8$):

| Performance Metric | In-Sample Training Fit | Leave-One-Out Cross-Validation (LOOCV) |
| :--- | :--- | :--- |
| **$R^2$ Score** | **$0.977$** | **$0.942$** (across 7 valid folds) |
| **Mean Absolute Error (MAE)** | **$0.78\text{ MPa}$** | **$1.18\text{ MPa}$** |
| **Root Mean Squared Error (RMSE)** | **$0.93\text{ MPa}$** | **$1.41\text{ MPa}$** |
| **Average Accuracy ($100\% - \text{MAPE}$)** | **$97.63\%$** | **$96.34\%$** |
| **Interpretation** | Optimistic upper-bound fit. | Unbiased proxy for real-world field predictions. |

> *Note on Numerical Conditioning:* Removing one batch during LOOCV revealed that 1 fold hit an ill-conditioned system ($X^T X$ pivot dropped to $\approx 7 \times 10^{-5}$). The solver identified and skipped this unstable fold, highlighting the sample-size limitation.

### 2. Learned Weights & Coefficient Significance

$$\text{Strength} = -5.99 + 0.1176(\text{Cement}) - 19.8648(w/c) + 0.5463(\text{Age})$$

| Coefficient | Parameter | Estimate | Std. Error | $t$-statistic | $p$-value | Significant ($p < 0.05$)? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$\beta_0$** | Intercept | $-5.9894$ | $115.59$ | $-0.05$ | $0.961$ | No |
| **$\beta_1$** | Cement ($\text{kg/m}^3$) | $+0.1176$ | $0.16$ | $+0.75$ | $0.495$ | No |
| **$\beta_2$** | $w/c$ Ratio | $-19.8648$ | $142.65$ | $-0.14$ | $0.896$ | No |
| **$\beta_3$** | Curing Age (Days) | $+0.5463$ | $0.075$ | $+7.30$ | **$0.002$** | **Yes (Statistically Meaningful)** |

* **Engineering Takeaway:** Curing age is statistically distinguishable from zero ($p = 0.002$). While cement content and water-cement ratio conform to Abrams' Law directionally, a larger batch sample size is required to narrow their confidence intervals.

---

## 📂 Project Structure

```text
├── app.py               # Streamlit web application with VIF, audit tables, and LOOCV UI
├── gauss_jordan.py      # Custom solver with partial pivoting & scale-aware singularity checks
├── linear_regression.py # OLS Normal Equations engine, custom (XᵀX)⁻¹ inversion, and LOOCV
├── requirements.txt     # Dependencies (streamlit, numpy, pandas, scipy)
└── README.md            # Project documentation and engineering defense report

```

---

## 🚀 Installation & Local Deployment

### 1. Prerequisites

Ensure Python 3.9+ is installed on your machine.

### 2. Clone the Repository

```bash
git clone [https://github.com/your-username/ML-Using-GaussJordan-Method.git](https://github.com/your-username/ML-Using-GaussJordan-Method.git)
cd ML-Using-GaussJordan-Method

```

### 3. Install Dependencies

```bash
pip install streamlit numpy pandas scipy

```

### 4. Run the Streamlit Application

```bash
streamlit run app.py

```

The interface will automatically launch at `http://localhost:8501`. The app runs fully offline without requiring an active internet connection.

---

## 🖥️ How to Use the Application

### Tab 1: System of Linear Equations ($Ax = b$)

* **Dimension Selection:** Select matrix dimensions $n \times n$ (from $2 \times 2$ to $8 \times 8$).
* **Interactive Matrix Entry:** Edit coefficient values and constants vector $b$ directly inside the interactive table.
* **Detailed Step Log:** Click **"Solve with Gauss-Jordan"** to view solution variables with clean unicode subscripts ($x_1, x_2, \dots$) and inspect every row swap, normalization, and elimination step.

### Tab 2: Concrete Strength Predictor

* **Multicollinearity Pre-Screening:** Expand the pre-training panel to inspect Variance Inflation Factors (VIF) and correlation coefficients.
* **Train Model:** Click **"🧠 Train Regression Model"** to compute $(X^T X)\beta = X^T y$.
* **Inspect Step-by-Step Derivation:** Review the symbolic formula, the explicit data summation table, the assembled augmented matrix, and the step-by-step reduction log.
* **Examine Validation:** Inspect in-sample metrics ($R^2$, MAE, RMSE) and compare them with the Leave-One-Out Cross-Validation (LOOCV) results.
* **Interactive Mix Simulator:** Adjust the Cement, $w/c$ ratio, and Curing Age sliders to predict compressive strength in real time.

---

## ⚠️ Engineering Limitations & Future Work

* **Sample Size:** With $N = 8$ batches for 4 unknowns, the degrees of freedom ($N - p = 4$) remain constrained. Expanding the dataset to $N \ge 40$ will narrow parameter standard errors.
* **Condition Squaring:** Forming $X^T X$ squares the matrix condition number, which contributed to one ill-conditioned cross-validation fold. Future iterations could implement QR decomposition or Singular Value Decomposition (SVD).
* **Linear Extrapolation:** Being an unconstrained linear model, physical bounds are only guaranteed within the domain of the training observations.

```

```
