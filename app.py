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
    page_title="System of Linear Equation Solver",
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
    [data-testid="stMetric"] {
        height: 100% !important;
        overflow: visible !important;
    }
    [data-testid="stMetricLabel"] {
        color: #F2AA52 !important;
        font-weight: 600 !important;
        min-height: 2.8em !important;
        line-height: 1.35 !important;
    }
    /* Streamlit truncates metric text with "..." by default; let it wrap instead */
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        overflow-wrap: anywhere !important;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: clamp(1.25rem, 1.9vw, 1.9rem) !important;
        line-height: 1.2 !important;
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
        <h2 style="margin: 0; color: #FFFFFF;">System of Linear Equation Solver</h2>
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

