import copy


def solve_gauss_jordan(A, b, return_steps=False, tol=1e-10):
    """Solves Ax = b using Gauss-Jordan elimination with partial pivoting.

    Optionally returns every intermediate matrix and row operation, for
    step-by-step auditing in the UI.

    `tol` is a *relative* tolerance: a pivot is treated as singular if it is
    smaller than `tol * (largest absolute value in the working matrix)`.
    """
    n = len(A)

    # 1. Build initial augmented matrix [A | b]
    M = []
    for i in range(n):
        M.append([float(x) for x in A[i]] + [float(b[i])])

    # Scale-aware singularity threshold
    matrix_scale = max((abs(v) for row in M for v in row), default=1.0)
    abs_tol = max(tol * matrix_scale, 1e-14)

    steps = []
    if return_steps:
        steps.append(
            {
                "title": "Initial Augmented Matrix [A | b]",
                "operation": (
                    "Formed by placing coefficients A on the left and"
                    " constants b on the right."
                ),
                "matrix": copy.deepcopy(M),
            }
        )

    # 2. Iterate through each pivot column
    for col in range(n):
        # --- A. Partial Pivoting ---
        max_row = col
        for row in range(col + 1, n):
            if abs(M[row][col]) > abs(M[max_row][col]):
                max_row = row

        if max_row != col:
            M[col], M[max_row] = M[max_row], M[col]
            if return_steps:
                steps.append(
                    {
                        "title": f"Column {col+1}: Partial Pivoting (Row Swap)",
                        "operation": (
                            f"Swapped Row {col+1} with Row {max_row+1} to place"
                            " the largest available value"
                            f" ({M[col][col]:.4f}) on the diagonal."
                        ),
                        "matrix": copy.deepcopy(M),
                    }
                )

        pivot = M[col][col]
        if abs(pivot) < abs_tol:
            raise ValueError(
                f"Matrix is singular or too ill-conditioned at column {col+1} "
                f"(pivot={pivot:.2e}, threshold={abs_tol:.2e}). This usually"
                " means two rows/predictors are duplicated or near-perfectly"
                " correlated -- check your input data for collinear columns."
            )

        # --- B. Normalize Pivot Row ---
        for j in range(col, n + 1):
            M[col][j] /= pivot

        if return_steps:
            steps.append(
                {
                    "title": f"Column {col+1}: Normalize Pivot Row",
                    "operation": (
                        f"Divided entire Row {col+1} by pivot value"
                        f" {pivot:.4f} so the diagonal entry becomes 1.0."
                    ),
                    "matrix": copy.deepcopy(M),
                }
            )

        # --- C. Row Elimination (Zero out all other entries in this column) ---
        elimination_notes = []
        for row in range(n):
            if row != col:
                factor = M[row][col]
                if abs(factor) > 1e-12:
                    for j in range(col, n + 1):
                        M[row][j] -= factor * M[col][j]
                    elimination_notes.append(
                        f"R{row+1} = R{row+1} - ({factor:.4f} x R{col+1})"
                    )

        if return_steps:
            notes_text = (
                "; ".join(elimination_notes)
                if elimination_notes
                else "Already 0.0"
            )
            steps.append(
                {
                    "title": f"Column {col+1}: Eliminate Other Column Entries",
                    "operation": f"Targeted zeroing operations: {notes_text}.",
                    "matrix": copy.deepcopy(M),
                }
            )

    solution = [M[i][n] for i in range(n)]

    if return_steps:
        return solution, steps
    return solution
