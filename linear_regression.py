import numpy as np
from gauss_jordan import solve_gauss_jordan


class MultipleLinearRegression:

  def __init__(self):
    self.coefficients = None
    self.XT_X = None
    self.XT_y = None
    self.steps = None
    self.n_features = None

  def fit(self, X, y, return_steps=False):
    """Solves (X^T * X) * beta = (X^T * y) using the custom Gauss-Jordan solver."""
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    # 1. Add bias column (vector of 1s for the intercept beta_0)
    ones = np.ones((X.shape[0], 1))
    X_design = np.hstack([ones, X])

    # 2. Compute Normal Equations matrices
    XT_X = np.dot(X_design.T, X_design)
    XT_y = np.dot(X_design.T, y)

    self.XT_X = XT_X
    self.XT_y = XT_y
    self.n_features = X.shape[1]

    # 3. Solve system using our custom Gauss-Jordan engine
    if return_steps:
      self.coefficients, self.steps = solve_gauss_jordan(
          XT_X.tolist(), XT_y.tolist(), return_steps=True
      )
      self.coefficients = np.array(self.coefficients)
      return self.coefficients, self.steps
    else:
      self.coefficients = np.array(
          solve_gauss_jordan(XT_X.tolist(), XT_y.tolist(), return_steps=False)
      )
      return self.coefficients

  def predict(self, X):
    """Computes y_pred = X_design * beta."""
    if self.coefficients is None:
      raise ValueError("Model has not been trained yet. Call fit() first.")

    X = np.array(X, dtype=float)
    if X.ndim == 1:
      X = X.reshape(1, -1)

    if X.shape[1] != self.n_features:
      raise ValueError(
          f"Model was trained on {self.n_features} feature(s) but received "
          f"{X.shape[1]}. Check that the columns passed to predict() match "
          f"the columns used in fit()."
      )

    ones = np.ones((X.shape[0], 1))
    X_design = np.hstack([ones, X])

    return np.dot(X_design, self.coefficients)

  def inverse_XtX(self):
    """Computes (X^T X)^-1 by solving (X^T X) * z_i = e_i for each unit vector e_i."""
    if self.XT_X is None:
      raise ValueError("Model has not been trained yet. Call fit() first.")

    n = self.XT_X.shape[0]
    XT_X_list = self.XT_X.tolist()
    inv_columns = []
    for i in range(n):
      e_i = [1.0 if j == i else 0.0 for j in range(n)]
      col = solve_gauss_jordan(XT_X_list, e_i, return_steps=False)
      inv_columns.append(col)
    return np.array(inv_columns).T

  def coefficient_stats(self, X, y):
    """Standard errors and t-statistics for each coefficient."""
    if self.coefficients is None:
      raise ValueError("Model has not been trained yet. Call fit() first.")

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n_samples = X.shape[0]
    n_params = len(self.coefficients)
    dof = n_samples - n_params

    if dof <= 0:
      raise ValueError(
          "Not enough observations to estimate coefficient uncertainty: "
          f"{n_samples} rows but {n_params} parameters (need more rows than "
          "parameters)."
      )

    residuals = y - self.predict(X)
    sigma2 = np.sum(residuals**2) / dof
    inv_XtX = self.inverse_XtX()
    se = np.sqrt(sigma2 * np.diag(inv_XtX))
    t_stats = self.coefficients / se

    p_values = None
    try:
      from scipy import stats

      p_values = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), dof))
    except ImportError:
      pass

    return {"se": se, "t_stats": t_stats, "p_values": p_values, "dof": dof}

  def loocv(self, X, y):
    """Leave-One-Out Cross-Validation."""
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n = X.shape[0]

    preds = np.full(n, np.nan)
    failed_indices = []
    for i in range(n):
      mask = np.ones(n, dtype=bool)
      mask[i] = False
      fold_model = MultipleLinearRegression()
      try:
        fold_model.fit(X[mask], y[mask])
        preds[i] = fold_model.predict(X[i : i + 1])[0]
      except ValueError:
        failed_indices.append(i)

    valid = ~np.isnan(preds)
    if not np.any(valid):
      raise ValueError(
          "Every leave-one-out fold hit a singular system -- this dataset is"
          " too small and/or too collinear for leave-one-out cross-validation."
      )

    residuals = np.where(valid, y - preds, np.nan)
    valid_residuals = residuals[valid]
    ss_tot = np.sum((y[valid] - np.mean(y[valid])) ** 2)
    ss_res = np.sum(valid_residuals**2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    mae = np.mean(np.abs(valid_residuals))
    rmse = np.sqrt(np.mean(valid_residuals**2))

    return {
        "predictions": preds,
        "residuals": residuals,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "n_valid": int(valid.sum()),
        "n_failed": len(failed_indices),
        "failed_indices": failed_indices,
    }
