import numpy as np
import pandas as pd
import streamlit as st
from gauss_jordan import solve_gauss_jordan
from linear_regression import MultipleLinearRegression

# -----------------------------------------------------------------------------
# HELPER: UNICODE SUBSCRIPT GENERATOR (x₁ instead of x_1)
# -----------------------------------------------------------------------------
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


# -----------------------------------------------------------------------------
# HELPER: VARIANCE INFLATION FACTOR (multicollinearity check)
# -----------------------------------------------------------------------------
def compute_vif(X, feature_names):
    """VIF_i measures how well predictor i is explained by the other predictors."""
    X = np.array(X, dtype=float)
    vifs = []
    for i in range(X.shape[1]):
        y_i = X[:, i]
        X_other = np.delete(X, i, axis=1)
        helper_model = MultipleLinearRegression()
        helper_model.fit(X_other, y_i)
        pred = helper_model.predict(X_other)
        ss_res = np.sum((y_i - pred) ** 2)
        ss_tot = np.sum((y_i - np.mean(y_i)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        vif = float("inf") if r2 > 0.9999 else 1.0 / (1.0 - r2)
        vifs.append(vif)
    return pd.DataFrame({"Predictor": feature_names, "VIF": vifs})


# -----------------------------------------------------------------------------
# 1. PAGE SETUP & METADATA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Civil Engineering Matrix & ML Solver",
    page_icon="🏗️",
    layout="wide",
)

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS STYLING (FULL PALETTE INTEGRATION)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* 1. Full Page & Canvas Backgrounds */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #17112E !important;
        color: #F8FAFC !important;
    }

    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* 2. Sidebar Theming */
    [data-testid="stSidebar"] {
        background-color: #1E153B !important;
        border-right: 1px solid rgba(242, 170, 82, 0.15) !important;
    }
    [data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }

    /* 3. Containers, Cards & Expanders */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #221844 !important;
        border: 1px solid rgba(242, 170, 82, 0.2) !important;
        border-radius: 10px !important;
    }
    [data-testid="stExpander"] {
        background-color: #221844 !important;
        border: 1px solid rgba(242, 170, 82, 0.2) !important;
        border-radius: 8px !important;
    }
    details summary {
        color: #F2AA52 !important;
    }

    /* 4. Metric Cards (#2D2059) */
    [data-testid="stMetric"] {
        background-color: #2D2059 !important;
        border: 1px solid rgba(242, 170, 82, 0.3) !important;
        padding: 16px 20px !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3) !important;
    }
    [data-testid="stMetricLabel"] {
        color: #F2AA52 !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* 5. Primary Action Buttons (#D95F18 resting, #F28B30 hover) */
    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="baseButton-primary"] {
        background-color: #D95F18 !important;
        border: 1px solid #D95F18 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.25s ease !important;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="baseButton-primary"]:hover {
        background-color: #F28B30 !important;
        border-color: #F28B30 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 14px rgba(242, 139, 48, 0.45) !important;
    }

    /* 6. Hero Top Banner */
    .hero-banner {
        background: linear-gradient(135deg, #2D2059 0%, #1A1238 100%) !important;
        border-left: 6px solid #F28B30 !important;
        padding: 22px 26px !important;
        border-radius: 10px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
    }

    /* 7. Tab Highlights */
    button[data-baseweb="tab"] {
        color: #C3BED7 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom-color: #F28B30 !important;
        color: #F28B30 !important;
    }

    /* 8. Sliders and Inputs */
    div[data-baseweb="slider"] div {
        color: #F28B30 !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & GROUP MEMBERS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("Project Info")
    st.markdown("**Institution:** Faculty of Computing, Engineering, and Technology")
    st.markdown("**Department:** Civil Engineering")
    st.markdown("**Core Method:** Gauss-Jordan Elimination")
    st.markdown("**ML Framework:** Multiple Linear Regression (OLS)")

    st.divider()

    st.subheader("👥 Proponents")
    st.markdown("• **John Joshua D. Ilisan**")
    st.markdown("• **Darius Lape**")
    st.markdown("• **Briel Jan M. Lacia**")

    st.divider()
    st.caption("Engineered for offline and online structural analysis.")

# -----------------------------------------------------------------------------
# 4. TOP HERO BANNER & APPLICATION OVERVIEW
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h2 style="margin: 0; color: #FFFFFF;">Concrete Compressive Strength Predictor & Linear Systems Solver</h2>
        <p style="margin: 6px 0 10px 0; color: #DDD8F0;">
            Supervised Machine Learning via Custom Gauss-Jordan Elimination with Scale-Aware Partial Pivoting.
        </p>
        <p style="margin: 0; font-size: 0.9rem; color: #F2AA52;">
            <strong>Proponents:</strong> John Joshua D. Ilisan | Darius Lape | Briel Jan M. Lacia &nbsp;|&nbsp; <em>September 2026</em>
        </p>
    </div>
""",
    unsafe_allow_html=True,
)

with st.expander("📖 About This Application & Engineering Methodology", expanded=False):
    col_desc1, col_desc2 = st.columns(2)
    with col_desc1:
        st.markdown("### Civil Engineering Problem")
        st.markdown(
            "In structural design and construction quality assurance, assessing the **28-day compressive strength** "
            "of concrete is the standard measure of mix adequacy. Conventionally, this requires preparing cylinder "
            "specimens and curing them in water tanks for nearly a month before conducting destructive testing on a "
            "**Universal Testing Machine (UTM)**."
        )
        st.markdown(
            "This application provides an early screening model to predict compressive strength directly from batch mix "
            "proportions (**Cement Content**, **Water-Cement Ratio**, and **Curing Age**) before casting."
        )

    with col_desc2:
        st.markdown("### First-Principles Numerical Engine")
        st.markdown(
            "Rather than treating Machine Learning as a 'black box' via libraries like Scikit-Learn, this platform is "
            "engineered completely from scratch:"
        )
        st.markdown(
            "• **Custom Gauss-Jordan Solver:** Implements scale-aware partial pivoting to avoid division by zero and "
            "suppress floating-point roundoff drift.\n"
            "• **Normal Equations Architecture:** Minimizes squared error by solving $(X^T X)\\beta = X^T y$ directly.\n"
            "• **Rigorous Validation:** Includes standard error hypothesis tests ($t$-statistics, $p$-values) and "
            "**Leave-One-Out Cross-Validation (LOOCV)** for honest generalizability assessment."
        )

    st.divider()
    st.markdown("### How to Use This Platform")
    c_tab1, c_tab2 = st.columns(2)
    with c_tab1:
        st.info(
            "**🔢 Tab 1: Linear System Solver**\n\n"
            "General-purpose $n \\times n$ matrix solver with editable inputs and an auditable step-by-step reduction log."
        )
    with c_tab2:
        st.info(
            "**📈 Tab 2: Concrete Strength Predictor**\n\n"
            "Inspect multicollinearity (VIF), trace the $(X^T X)\\beta = X^T y$ derivation, and simulate mix designs in real time."
        )

tab1, tab2 = st.tabs(
    ["🔢 System of Linear Equations", "📈 Multiple Linear Regression"]
)

# -----------------------------------------------------------------------------
# TAB 1: GAUSS-JORDAN MATRIX SOLVER
# -----------------------------------------------------------------------------
with tab1:
    st.header("Solve a System of Linear Equations ($Ax = b$)")
    st.caption("General-purpose numerical solver powered by our custom Gauss-Jordan elimination engine.")

    with st.container(border=True):
        col_ctrl1, col_ctrl2 = st.columns([1, 3])
        with col_ctrl1:
            n = st.number_input(
                "Matrix Dimension (n x n):",
                min_value=2,
                max_value=8,
                value=4,
                step=1,
            )
        with col_ctrl2:
            st.info(
                "Click any cell in the table below to edit coefficients. Always press **Enter** after typing."
            )

        default_data = [
            [4.0, -1.0, 0.0, 2.0, 15.0],
            [-1.0, 5.0, -2.0, 0.0, 10.0],
            [0.0, -2.0, 6.0, -1.0, 8.0],
            [2.0, 0.0, -1.0, 4.0, 20.0],
        ]

        col_headers = [f"x{str(i+1).translate(sub)}" for i in range(n)] + [
            "Constants (b)"
        ]

        if n == 4:
            df_init = pd.DataFrame(default_data, columns=col_headers)
        else:
            df_init = pd.DataFrame(0.0, index=range(n), columns=col_headers)

        edited_df = st.data_editor(
            df_init,
            use_container_width=True,
            num_rows="fixed",
            key=f"matrix_editor_{n}",
        )

        solve_clicked = st.button(
            "Solve with Gauss-Jordan", type="primary", use_container_width=True
        )

    if solve_clicked:
        try:
            matrix_vals = edited_df.values
            A = matrix_vals[:, :n].tolist()
            b = matrix_vals[:, n].tolist()

            solution, steps = solve_gauss_jordan(A, b, return_steps=True)

            st.success("✅ System successfully reduced to identity matrix!")

            cols = st.columns(n)
            for i in range(n):
                with cols[i]:
                    var_name = f"x{str(i+1).translate(sub)}"
                    st.metric(label=var_name, value=f"{solution[i]:.4f}")

            st.divider()
            st.subheader("📝 Step-by-Step Row Operations")
            st.caption(f"Reduced in {len(steps)} systematic transformations.")

            for idx, step_info in enumerate(steps):
                with st.expander(
                    f"Step {idx+1}: {step_info['title']}", expanded=(idx == 0)
                ):
                    st.markdown(f"**Action:** `{step_info['operation']}`")
                    step_df = pd.DataFrame(
                        step_info["matrix"], columns=col_headers
                    )
                    st.dataframe(
                        step_df.style.format("{:.4f}"), use_container_width=True
                    )

        except Exception as e:
            st.error(f"Solver Error: {e}")

# -----------------------------------------------------------------------------
# TAB 2: MULTIPLE LINEAR REGRESSION (ML LAYER)
# -----------------------------------------------------------------------------
with tab2:
    st.header("Concrete Strength Predictor")
    st.caption(
        "Fits Ordinary Least Squares via the Normal Equations using our custom"
        " Gauss-Jordan solver."
    )

    with st.container(border=True):
        st.subheader("1. Training Dataset (Mix Batch Observations)")
        default_ml_data = {
            "Cement_kg_m3": [
                280.0,
                280.0,
                350.0,
                350.0,
                420.0,
                420.0,
                310.0,
                390.0,
            ],
            "Water_Cement_Ratio": [
                0.50,
                0.50,
                0.42,
                0.42,
                0.35,
                0.35,
                0.48,
                0.38,
            ],
            "Curing_Age_Days": [7.0, 28.0, 7.0, 28.0, 7.0, 28.0, 14.0, 14.0],
            "Strength_MPa": [21.5, 32.0, 29.8, 41.2, 38.5, 52.0, 28.4, 43.1],
        }
        if "ml_training_data" not in st.session_state:
            st.session_state.ml_training_data = pd.DataFrame(default_ml_data)
        if "ml_editor_version" not in st.session_state:
            st.session_state.ml_editor_version = 0

        edited_data = st.data_editor(
            st.session_state.ml_training_data,
            use_container_width=True,
            num_rows="dynamic",
            key=f"ml_data_editor_{st.session_state.ml_editor_version}",
        )
        st.session_state.ml_training_data = edited_data

        remove_col, button_col = st.columns([3, 1])
        with remove_col:
            if len(edited_data) > 1:
                batch_to_remove = st.selectbox(
                    "Select a batch to remove",
                    options=list(edited_data.index),
                    format_func=lambda i: (
                        f"Batch {i + 1}: Cement={edited_data.loc[i, 'Cement_kg_m3']:.0f} kg/m³, "
                        f"w/c={edited_data.loc[i, 'Water_Cement_Ratio']:.2f}, "
                        f"Age={edited_data.loc[i, 'Curing_Age_Days']:.0f}d, "
                        f"Strength={edited_data.loc[i, 'Strength_MPa']:.1f} MPa"
                    ),
                    label_visibility="collapsed",
                )
            else:
                batch_to_remove = None
                st.caption("At least one batch must remain — add more before removing this one.")
        with button_col:
            if st.button(
                "🗑️ Remove Batch",
                use_container_width=True,
                disabled=(batch_to_remove is None),
            ):
                st.session_state.ml_training_data = edited_data.drop(
                    index=batch_to_remove
                ).reset_index(drop=True)
                st.session_state.ml_editor_version += 1
                st.rerun()

        feature_cols_preview = [
            "Cement_kg_m3",
            "Water_Cement_Ratio",
            "Curing_Age_Days",
        ]
        with st.expander(
            "🔎 Check Predictor Multicollinearity (VIF) Before Training",
            expanded=False,
        ):
            st.caption(
                "Cement content and water-cement ratio are physically linked in mix design. "
                "Rule of thumb: VIF > 5–10 signals problematic collinearity."
            )
            try:
                vif_df = compute_vif(
                    edited_data[feature_cols_preview].values,
                    feature_cols_preview,
                )
                st.dataframe(
                    vif_df.style.format({"VIF": "{:.2f}"}),
                    use_container_width=True,
                    hide_index=True,
                )
                corr_df = edited_data[feature_cols_preview].corr()
                st.caption("Pairwise correlation matrix:")
                st.dataframe(
                    corr_df.style.format("{:.2f}"),
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"Could not compute VIF yet: {e}")

        train_clicked = st.button(
            "🧠 Train Regression Model",
            type="primary",
            use_container_width=True,
        )

    if train_clicked:
        try:
            feature_cols = [
                "Cement_kg_m3",
                "Water_Cement_Ratio",
                "Curing_Age_Days",
            ]
            X = edited_data[feature_cols].values
            y = edited_data["Strength_MPa"].values

            model = MultipleLinearRegression()
            coeffs, ml_steps = model.fit(X, y, return_steps=True)

            st.session_state["trained_model"] = model
            st.session_state["coeffs"] = coeffs
            st.session_state["ml_steps"] = ml_steps
            st.session_state["XT_X"] = model.XT_X
            st.session_state["XT_y"] = model.XT_y
            st.session_state["X_data"] = X
            st.session_state["y_data"] = y

            st.success("✅ Model weights optimized successfully!")
        except Exception as e:
            st.error(f"Training failed: {e}")

    if "trained_model" in st.session_state:
        coeffs = st.session_state["coeffs"]
        ml_steps = st.session_state["ml_steps"]
        XT_X = st.session_state["XT_X"]
        XT_y = st.session_state["XT_y"]
        X_data = st.session_state["X_data"]
        y_data = st.session_state["y_data"]

        with st.container(border=True):
            st.subheader("2. Learned Model Weights")
            c0, c1, c2, c3 = st.columns(4)
            c0.metric("Intercept (β₀)", f"{coeffs[0]:.4f}")
            c1.metric("Cement (β₁)", f"{coeffs[1]:.4f}")
            c2.metric("w/c Ratio (β₂)", f"{coeffs[2]:.4f}")
            c3.metric("Age Days (β₃)", f"{coeffs[3]:.4f}")

            st.latex(
                rf"\text{{Strength}} = {coeffs[0]:.2f} + ({coeffs[1]:.4f} \times \text{{Cement}}) + ({coeffs[2]:.2f} \times \text{{w/c}}) + ({coeffs[3]:.4f} \times \text{{Age}})"
            )

            st.markdown("##### Coefficient Significance")
            st.caption(
                "Standard error and t-stat are computed from (XᵀX)⁻¹ using the same custom solver."
            )
            try:
                stats_result = st.session_state[
                    "trained_model"
                ].coefficient_stats(X_data, y_data)
                se = stats_result["se"]
                t_stats = stats_result["t_stats"]
                p_values = stats_result["p_values"]
                dof = stats_result["dof"]

                sig_table = {
                    "Coefficient": ["β₀ (Intercept)", "β₁ (Cement)", "β₂ (w/c)", "β₃ (Age)"],
                    "Estimate": [f"{c:.4f}" for c in coeffs],
                    "Std. Error": [f"{s:.4f}" for s in se],
                    "t-stat": [f"{t:.2f}" for t in t_stats],
                }
                if p_values is not None:
                    sig_table["p-value"] = [f"{p:.4f}" for p in p_values]
                    sig_table["Significant (p<0.05)"] = [
                        "✅ Yes" if p < 0.05 else "— No" for p in p_values
                    ]
                st.dataframe(
                    pd.DataFrame(sig_table), use_container_width=True, hide_index=True
                )
                if p_values is None:
                    st.caption(
                        f"(degrees of freedom = {dof}; install scipy to view exact p-values)"
                    )
                else:
                    st.caption(f"degrees of freedom = {dof}")
            except ValueError as e:
                st.info(f"Coefficient significance unavailable: {e}")

        # ---------------------------------------------------------------------
        # 3. MATHEMATICAL DERIVATION: (XᵀX)β = Xᵀy
        # ---------------------------------------------------------------------
        with st.container(border=True):
            st.subheader(
                "3. 📝 Step-by-Step Derivation of $(X^T X)\\beta = X^T y$"
            )
            st.markdown(
                "Minimizing squared prediction errors yields the **Normal Equations**:"
            )
            st.latex(r"(X^T X)\beta = X^T y \quad \iff \quad A\beta = b")

            x1 = X_data[:, 0]
            x2 = X_data[:, 1]
            x3 = X_data[:, 2]
            y_vec = y_data
            N = len(y_vec)

            with st.expander(
                "📌 Part A: Matrix Equations (Symbolic Formula vs. Numerical Values)",
                expanded=True,
            ):
                st.markdown("**1. General Symbolic Matrix Structure:**")
                st.latex(r"""
                \begin{bmatrix}
                N & \sum x_1 & \sum x_2 & \sum x_3 \\
                \sum x_1 & \sum x_1^2 & \sum x_1 x_2 & \sum x_1 x_3 \\
                \sum x_2 & \sum x_2 x_1 & \sum x_2^2 & \sum x_2 x_3 \\
                \sum x_3 & \sum x_3 x_1 & \sum x_3 x_2 & \sum x_3^2
                \end{bmatrix}
                \begin{bmatrix} \beta_0 \\ \beta_1 \\ \beta_2 \\ \beta_3 \end{bmatrix}
                =
                \begin{bmatrix}
                \sum y \\
                \sum x_1 y \\
                \sum x_2 y \\
                \sum x_3 y
                \end{bmatrix}
                """)

                st.markdown(
                    "**2. Substituted with Computed Numerical Values:**"
                )
                st.latex(rf"""
                \begin{{bmatrix}}
                {XT_X[0,0]:.1f} & {XT_X[0,1]:.1f} & {XT_X[0,2]:.4f} & {XT_X[0,3]:.1f} \\
                {XT_X[1,0]:.1f} & {XT_X[1,1]:,.1f} & {XT_X[1,2]:,.1f} & {XT_X[1,3]:,.1f} \\
                {XT_X[2,0]:.4f} & {XT_X[2,1]:,.1f} & {XT_X[2,2]:.4f} & {XT_X[2,3]:.4f} \\
                {XT_X[3,0]:.1f} & {XT_X[3,1]:,.1f} & {XT_X[3,2]:.4f} & {XT_X[3,3]:,.1f}
                \end{{bmatrix}}
                \begin{{bmatrix}} \beta_0 \\ \beta_1 \\ \beta_2 \\ \beta_3 \end{{bmatrix}}
                =
                \begin{{bmatrix}}
                {XT_y[0]:.1f} \\
                {XT_y[1]:,.1f} \\
                {XT_y[2]:.3f} \\
                {XT_y[3]:,.1f}
                \end{{bmatrix}}
                """)

            with st.expander(
                "🔢 Part B: Calculations Breakdown (Data Values & Arithmetic)",
                expanded=True,
            ):
                st.markdown(
                    "Each entry in the matrix is formed by summing across all batch samples:"
                )

                def format_sum(arr):
                    if len(arr) <= 8:
                        return " + ".join([f"{v:g}" for v in arr])
                    return (
                        " + ".join([f"{v:g}" for v in arr[:4]])
                        + " + ... + "
                        + " + ".join([f"{v:g}" for v in arr[-2:]])
                    )

                def format_prod_sum(arr1, arr2):
                    if len(arr1) <= 8:
                        return " + ".join(
                            [f"({a:g}×{b:g})" for a, b in zip(arr1, arr2)]
                        )
                    first = " + ".join(
                        [f"({a:g}×{b:g})" for a, b in zip(arr1[:3], arr2[:3])]
                    )
                    return f"{first} + ... + ({arr1[-1]:g}×{arr2[-1]:g})"

                sum_table = {
                    "Matrix Term": [
                        "N (Sample Count)",
                        "Σ Cement (x₁)",
                        "Σ w/c Ratio (x₂)",
                        "Σ Age (x₃)",
                        "Σ Strength (y)",
                        "Σ Cement² (x₁²)",
                        "Σ (Cement × w/c)",
                        "Σ (Cement × Age)",
                        "Σ (Cement × Strength)",
                        "Σ (w/c)² (x₂²)",
                        "Σ (w/c × Age)",
                        "Σ (w/c × Strength)",
                        "Σ Age² (x₃²)",
                        "Σ (Age × Strength)",
                    ],
                    "Actual Numbers Being Added": [
                        " + ".join(["1"] * N),
                        format_sum(x1),
                        format_sum(x2),
                        format_sum(x3),
                        format_sum(y_vec),
                        format_sum(np.round(x1**2, 1)),
                        format_prod_sum(x1, x2),
                        format_prod_sum(x1, x3),
                        format_prod_sum(x1, y_vec),
                        format_sum(np.round(x2**2, 4)),
                        format_prod_sum(x2, x3),
                        format_prod_sum(x2, y_vec),
                        format_sum(np.round(x3**2, 1)),
                        format_prod_sum(x3, y_vec),
                    ],
                    "Computed Total": [
                        f"{N}",
                        f"{np.sum(x1):,.2f}",
                        f"{np.sum(x2):,.4f}",
                        f"{np.sum(x3):,.1f}",
                        f"{np.sum(y_vec):,.2f}",
                        f"{np.sum(x1**2):,.2f}",
                        f"{np.sum(x1*x2):,.2f}",
                        f"{np.sum(x1*x3):,.2f}",
                        f"{np.sum(x1*y_vec):,.2f}",
                        f"{np.sum(x2**2):,.4f}",
                        f"{np.sum(x2*x3):,.2f}",
                        f"{np.sum(x2*y_vec):,.2f}",
                        f"{np.sum(x3**2):,.2f}",
                        f"{np.sum(x3*y_vec):,.2f}",
                    ],
                }
                st.dataframe(
                    pd.DataFrame(sum_table),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown(
                "##### **Part C: Assembled Augmented Matrix $[(X^T X) \\mid (X^T y)]$**"
            )
            norm_col_headers = [
                "β₀ (Intercept)",
                "β₁ (Cement)",
                "β₂ (w/c)",
                "β₃ (Age)",
                "Constants (Xᵀy)",
            ]
            initial_augmented = np.hstack([XT_X, XT_y.reshape(-1, 1)])
            df_norm = pd.DataFrame(
                initial_augmented,
                columns=norm_col_headers,
                index=[
                    "Row 1 (β₀)",
                    "Row 2 (β₁)",
                    "Row 3 (β₂)",
                    "Row 4 (β₃)",
                ],
            )
            st.dataframe(
                df_norm.style.format("{:,.4f}"), use_container_width=True
            )

            st.divider()
            st.markdown(
                "##### **Part D: Step-by-Step Gauss-Jordan Reduction to Identity $[I \\mid \\beta]$**"
            )
            st.caption(
                f"Our custom solver reduces this 4×4 system in {len(ml_steps)} steps to solve for the final coefficients:"
            )

            for idx, step_info in enumerate(ml_steps):
                with st.expander(
                    f"Step {idx+1}: {step_info['title']}", expanded=(idx == 0)
                ):
                    st.markdown(f"**Action:** `{step_info['operation']}`")
                    step_df = pd.DataFrame(
                        step_info["matrix"],
                        columns=norm_col_headers,
                        index=["Row 1", "Row 2", "Row 3", "Row 4"],
                    )
                    st.dataframe(
                        step_df.style.format("{:,.4f}"),
                        use_container_width=True,
                    )

        # ---------------------------------------------------------------------
        # 4. ENHANCED MODEL VALIDATION (DETAILED ACTUAL VS. PREDICTED)
        # ---------------------------------------------------------------------
        with st.container(border=True):
            st.subheader("4. 📊 Training Fit: Actual vs. Predicted (In-Sample)")
            st.markdown(
                "This section measures fit on the **same batches used for training**."
            )

            y_pred = st.session_state["trained_model"].predict(X_data)
            residuals = y_data - y_pred
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            ss_res = np.sum(residuals**2)
            r2 = 1.0 - (ss_res / ss_tot)
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(np.mean(residuals**2))
            mape = np.mean(np.abs(residuals / y_data)) * 100.0
            accuracy_pct = 100.0 - mape

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("R² Score (Fit Quality)", f"{r2:.4f}")
            m2.metric("Mean Absolute Error (MAE)", f"{mae:.2f} MPa")
            m3.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f} MPa")
            m4.metric("Max Error", f"{np.max(np.abs(residuals)):.2f} MPa")
            m5.metric("Accuracy (100% − MAPE)", f"{accuracy_pct:.2f}%")

            st.info("""
            **Graph Legend & Axes:**
            * 📌 **Horizontal Axis (X-Axis):** **Batch Cylinder Specimen** (`Batch 1` to `Batch 8`).
            * 📌 **Vertical Axis (Y-Axis):** **Compressive Strength in Megapascals ($\text{MPa}$)**.
            * 🟡 **Amber Line (`#F2AA52`):** Actual Lab Measured Strength (from UTM testing).
            * 🟠 **Terracotta Line (`#D95F18`):** Model Predicted Strength (from Gauss-Jordan regression).
            """)

            batch_labels = [f"Batch {i+1}" for i in range(len(y_data))]
            chart_df = pd.DataFrame(
                {
                    "Actual Lab Strength (MPa)": y_data,
                    "Model Prediction (MPa)": y_pred,
                },
                index=batch_labels,
            )

            st.line_chart(
                chart_df,
                color=["#F2AA52", "#D95F18"],
                use_container_width=True,
            )

            with st.expander(
                "📋 View Detailed Batch-by-Batch Residual Error Table",
                expanded=True,
            ):
                audit_df = pd.DataFrame(
                    {
                        "Batch Specimen": batch_labels,
                        "Cement (kg/m³)": X_data[:, 0],
                        "w/c Ratio": X_data[:, 1],
                        "Curing Age (Days)": X_data[:, 2],
                        "Actual Strength (MPa)": y_data,
                        "Predicted Strength (MPa)": y_pred,
                        "Residual Error (MPa)": residuals,
                        "Accuracy (%)": 100.0 - (np.abs(residuals) / y_data * 100.0),
                    }
                )

                st.dataframe(
                    audit_df.style.format({
                        "Cement (kg/m³)": "{:.1f}",
                        "w/c Ratio": "{:.2f}",
                        "Curing Age (Days)": "{:.0f}",
                        "Actual Strength (MPa)": "{:.2f}",
                        "Predicted Strength (MPa)": "{:.2f}",
                        "Residual Error (MPa)": "{:+.2f}",
                        "Accuracy (%)": "{:.1f}%",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

        # ---------------------------------------------------------------------
        # 5. LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV)
        # ---------------------------------------------------------------------
        with st.container(border=True):
            st.subheader("5. 🎯 Cross-Validated Performance (Leave-One-Out)")
            st.markdown(
                "Each batch is held out in turn while the model is refitted from scratch on the remaining batches. "
                "This provides an honest estimate of real-world predictive generalization."
            )

            with st.spinner("Refitting model for each held-out batch..."):
                try:
                    loocv_result = st.session_state["trained_model"].loocv(
                        X_data, y_data
                    )
                except ValueError as e:
                    loocv_result = None
                    st.error(f"Cross-validation unavailable: {e}")

            if loocv_result is not None:
                if loocv_result["n_failed"] > 0:
                    st.warning(
                        f"⚠️ {loocv_result['n_failed']} of {len(y_data)} folds hit an ill-conditioned system "
                        f"when that batch was removed and were skipped. Metrics use the remaining {loocv_result['n_valid']} folds."
                    )

                lc1, lc2, lc3, lc4 = st.columns(4)
                lc1.metric("LOOCV R²", f"{loocv_result['r2']:.4f}")
                lc2.metric("LOOCV MAE", f"{loocv_result['mae']:.2f} MPa")
                lc3.metric("LOOCV RMSE", f"{loocv_result['rmse']:.2f} MPa")

                valid_mask = ~np.isnan(loocv_result["residuals"])
                loocv_mape = np.mean(
                    np.abs(
                        loocv_result["residuals"][valid_mask]
                        / y_data[valid_mask]
                    )
                ) * 100.0
                lc4.metric("LOOCV Accuracy (100% − MAPE)", f"{100.0 - loocv_mape:.2f}%")

                if loocv_result["r2"] < r2 - 0.15:
                    st.warning(
                        "⚠️ LOOCV performance is lower than in-sample fit, indicating potential sensitivity to small sample size."
                    )

                loocv_df = pd.DataFrame(
                    {
                        "Batch Specimen": batch_labels,
                        "Actual Strength (MPa)": y_data,
                        "Held-Out Prediction (MPa)": loocv_result["predictions"],
                        "Residual (MPa)": loocv_result["residuals"],
                    }
                )
                st.dataframe(
                    loocv_df.style.format(
                        {
                            "Actual Strength (MPa)": "{:.2f}",
                            "Held-Out Prediction (MPa)": "{:.2f}",
                            "Residual (MPa)": "{:+.2f}",
                        },
                        na_rep="— (skipped)",
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        # ---------------------------------------------------------------------
        # 6. INTERACTIVE MIX DESIGN PREDICTOR
        # ---------------------------------------------------------------------
        with st.container(border=True):
            st.subheader("6. 🎛️ Interactive Strength Predictor")
            st.write(
                "Use the sliders below to test arbitrary mix designs in real time:"
            )

            p1, p2, p3 = st.columns(3)
            with p1:
                in_cement = st.slider(
                    "Cement Content (kg/m³)", 150.0, 550.0, 350.0, 10.0
                )
            with p2:
                in_wc = st.slider(
                    "Water-Cement Ratio (w/c)", 0.25, 0.65, 0.42, 0.01
                )
            with p3:
                in_age = st.slider("Curing Age (Days)", 1.0, 90.0, 28.0, 1.0)

            sample = [[in_cement, in_wc, in_age]]
            pred_strength = st.session_state["trained_model"].predict(sample)[0]

            st.markdown(
                f"""
                <div style="background-color: #2D2059; padding: 22px; border-radius: 10px; border: 2px solid #F28B30; text-align: center; margin-top: 10px; box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);">
                    <span style="font-size: 1.1rem; color: #F2AA52; font-weight: 500;">Estimated Compressive Strength:</span>
                    <h2 style="margin: 6px 0 0 0; color: #FFFFFF; font-weight: 800;">{pred_strength:.2f} MPa</h2>
                </div>
            """,
                unsafe_allow_html=True,
            )
