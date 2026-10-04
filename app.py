import pandas as pd
import streamlit as st
from gauss_jordan import solve_gauss_jordan

# -----------------------------------------------------------------------------
# HELPER: UNICODE SUBSCRIPT GENERATOR (x₁ instead of x_1)
# -----------------------------------------------------------------------------
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

LIVE_APP_URL = "https://mlgaussjordan.streamlit.app"

# -----------------------------------------------------------------------------
# 1. PAGE SETUP & METADATA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="System of Linear Equations Solver",
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

    /* 9. "Coming Soon" roadmap badge */
    .coming-soon-badge {
        display: inline-block;
        background: rgba(242, 139, 48, 0.15);
        border: 1px solid #F28B30;
        color: #F28B30;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.6px;
        margin-bottom: 10px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION, INSTRUCTOR & GROUP MEMBERS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("Project Info")
    st.markdown("**Institution:** Davao Oriental State University")
    st.markdown("**Faculty:** Computing, Engineering and Technology")
    st.markdown("**Department:** Civil Engineering")
    st.markdown("**Core Method:** Gauss-Jordan Elimination")
    st.markdown("**Stability:** Partial Pivoting + Scale-Aware Tolerance")

    st.divider()

    st.subheader("👨‍🏫 Course Instructor")
    st.markdown("• **Engr. Samuel Erespe**")

    st.subheader("👥 Proponents & Contributors")
    st.markdown("• **John Joshua D. Ilisan**")
    st.markdown("• **Darius Lape**")
    st.markdown("• **Briel Jan M. Lacia**")

    st.divider()
    st.subheader("Roadmap")
    st.markdown(
        "**Next update:** Machine Learning module that predicts "
        "**concrete compressive strength**."
    )

    st.divider()
    st.caption("Engineered for offline and online structural analysis.")

# -----------------------------------------------------------------------------
# 4. TOP HERO BANNER & APPLICATION OVERVIEW
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h2 style="margin: 0; color: #FFFFFF;">🏗️ System of Linear Equations Solver</h2>
        <p style="margin: 6px 0 10px 0; color: #DDD8F0;">
            Numerical Linear Algebra via Custom Gauss-Jordan Elimination with Scale-Aware Pivoting.
        </p>
        <p style="margin: 0 0 4px 0; font-size: 0.9rem; color: #F2AA52;">
            <strong>Instructor:</strong> Engr. Samuel Erespe
        </p>
        <p style="margin: 0; font-size: 0.9rem; color: #F2AA52;">
            <strong>Proponents:</strong> John Joshua D. Ilisan | Darius Lape | Briel Jan M. Lacia &nbsp;|&nbsp; <em>September 2026</em>
        </p>
    </div>
""",
    unsafe_allow_html=True,
)

st.info(
    "🚀 **Coming in the next update:** a Machine Learning module that will "
    "**predict the compressive strength of concrete** from its mix "
    "proportions. Check the **Roadmap** tab for details."
)

with st.expander("📖 About This Application & Engineering Methodology", expanded=False):
    st.markdown("### 📌 Executive Summary")
    st.markdown(
        "In engineering analysis and numerical computation, solving systems of linear "
        "equations ($Ax = b$) is a foundational requirement for structural analysis, "
        "network flow calculations, and frame modeling. This project provides a robust, "
        "transparent, first-principles implementation of **Gauss-Jordan Elimination** "
        "designed to solve general $n \\times n$ systems of linear equations without "
        "relying on external linear algebra solvers or commercial libraries."
    )

    col_desc1, col_desc2 = st.columns(2)
    with col_desc1:
        st.markdown("### Custom Gauss-Jordan Elimination Engine")
        st.markdown(
            "`gauss_jordan.py` solves general square systems using:\n\n"
            "• Scale-aware relative tolerance singularity detection\n\n"
            "• Partial pivoting for numerical stability\n\n"
            "• Full audit logging of row transformations"
        )
    with col_desc2:
        st.markdown("### Interactive Streamlit Dashboard")
        st.markdown(
            "`app.py` lets you:\n\n"
            "• Adjust matrix dimensions from $2 \\times 2$ up to $8 \\times 8$\n\n"
            "• Manually input or modify matrix coefficients\n\n"
            "• Compute exact solutions and trace every intermediate row operation step-by-step"
        )

    st.divider()
    st.markdown("### How to Use This Platform")
    c_tab1, c_tab2, c_tab3 = st.columns(3)
    with c_tab1:
        st.info(
            "**🔢 Tab 1: Linear System Solver**\n\n"
            "General-purpose $n \\times n$ matrix solver with editable inputs and an "
            "auditable step-by-step reduction log."
        )
    with c_tab2:
        st.info(
            "**📐 Tab 2: Mathematical Formulation**\n\n"
            "The derivation behind the solver, plus its key technical features."
        )
    with c_tab3:
        st.info(
            "**🚀 Tab 3: Roadmap**\n\n"
            "What's coming next: ML-based prediction of concrete strength."
        )

tab1, tab2, tab3 = st.tabs(
    [
        "🔢 System of Linear Equations",
        "📐 Mathematical Formulation",
        "🚀 Roadmap: Concrete Strength ML",
    ]
)

# -----------------------------------------------------------------------------
# TAB 1: GAUSS-JORDAN MATRIX SOLVER
# -----------------------------------------------------------------------------
with tab1:
    st.header("Solve a System of Linear Equations ($Ax = b$)")
    st.caption(
        "General-purpose numerical solver powered by our custom Gauss-Jordan "
        "elimination engine."
    )

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
                "Click any cell in the table below to edit coefficients. "
                "Always press **Enter** after typing."
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
# TAB 2: MATHEMATICAL FORMULATION (from README)
# -----------------------------------------------------------------------------
with tab2:
    st.header("📐 Mathematical Formulation")

    st.subheader("1. General System of Linear Equations")
    st.markdown(
        "A system of $n$ linear equations with $n$ unknown variables is expressed "
        "in matrix notation as:"
    )
    st.latex(r"Ax = b")
    st.latex(
        r"""
        \begin{bmatrix}
        a_{11} & a_{12} & \cdots & a_{1n} \\
        a_{21} & a_{22} & \cdots & a_{2n} \\
        \vdots & \vdots & \ddots & \vdots \\
        a_{n1} & a_{n2} & \cdots & a_{nn}
        \end{bmatrix}
        \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}
        =
        \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_n \end{bmatrix}
        """
    )
    st.markdown(
        "The goal is to determine the unknown vector "
        "$x = [x_1, x_2, \\dots, x_n]^T$ by reducing the augmented matrix "
        "$[A \\mid b]$ into reduced row echelon form (RREF) $[I \\mid x]$, where "
        "$I$ is the $n \\times n$ identity matrix."
    )

    st.divider()
    st.subheader("2. Scale-Aware Gauss-Jordan Elimination with Partial Pivoting")

    with st.container(border=True):
        st.markdown("#### Step 1: Augmented Matrix Setup")
        st.markdown(
            "The coefficient matrix $A$ and constant vector $b$ are combined into an "
            "augmented matrix $M$ of dimension $n \\times (n+1)$:"
        )
        st.latex(r"M = [A \mid b]")

    with st.container(border=True):
        st.markdown("#### Step 2: Partial Pivoting")
        st.markdown(
            "To maintain numerical stability and avoid division by zero (or small "
            "numbers that introduce roundoff errors), row swaps are performed at each "
            "column $k$:"
        )
        st.latex(
            r"\text{Find } r \text{ such that } \lvert M_{r,k} \rvert = "
            r"\max_{i \ge k} \lvert M_{i,k} \rvert"
        )
        st.markdown("If $r \\neq k$, Row $k$ and Row $r$ are swapped.")

    with st.container(border=True):
        st.markdown("#### Step 3: Scale-Aware Singularity Detection")
        st.markdown(
            "Rather than relying on a rigid fixed tolerance, the solver establishes a "
            "relative singularity threshold based on the overall magnitude of the "
            "working matrix:"
        )
        st.latex(
            r"\text{threshold} = \max\left(\text{tol} \times \max(\lvert M \rvert),"
            r"\ 10^{-14}\right), \quad \text{tol} = 10^{-10}"
        )
        st.markdown(
            "If the pivot value $\\lvert M_{k,k} \\rvert < \\text{threshold}$, the "
            "system is flagged as singular or ill-conditioned, and execution halts "
            "cleanly with an informative explanation."
        )

    with st.container(border=True):
        st.markdown("#### Step 4: Row Normalization & Elementary Transformations")
        st.markdown(
            "**Normalize Pivot Row:** divide the entire pivot row by the diagonal "
            "entry $M_{k,k}$ so that $M_{k,k} = 1$:"
        )
        st.latex(
            r"M_{k,j} \leftarrow \frac{M_{k,j}}{M_{k,k}}"
            r"\quad \text{for } j = k, \dots, n"
        )
        st.markdown(
            "**Eliminate Non-Zero Column Entries:** zero out all off-diagonal entries "
            "in column $k$ across all other rows $i \\neq k$:"
        )
        st.latex(
            r"M_{i,j} \leftarrow M_{i,j} - (M_{i,k} \times M_{k,j})"
            r"\quad \text{for all } i \neq k"
        )
        st.markdown(
            "Upon completing elimination across all $n$ columns, the solution vector "
            "is directly read from the final column:"
        )
        st.latex(r"x_i = M_{i,n+1} \quad \text{for } i = 1, 2, \dots, n")

    st.divider()
    st.subheader("🔬 Key Technical Features")
    features_df = pd.DataFrame(
        {
            "Feature": [
                "Partial Pivoting",
                "Scale-Aware Singularity Checking",
                "Auditable Transformation Logging",
                "Flexible System Sizing",
            ],
            "Description": [
                "Interchanges rows to place the absolute largest available pivot on the diagonal.",
                "Evaluates pivots against a dynamic threshold based on matrix magnitude (tol × max|M|).",
                "Records every intermediate matrix state and human-readable row operation formula.",
                "Dynamically scales from 2×2 up to 8×8 linear systems via interactive UI controls.",
            ],
            "Engineering Benefit": [
                "Prevents division by zero and minimizes floating-point precision degradation.",
                "Prevents false singular rejections on high-magnitude matrices while catching ill-conditioned systems.",
                "Serves as an educational step-by-step calculation trace for verification.",
                "Accommodates a wide variety of structural and linear algebra problems.",
            ],
        }
    )
    st.dataframe(features_df, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# TAB 3: ROADMAP - UPCOMING ML CONCRETE STRENGTH PREDICTION
# -----------------------------------------------------------------------------
with tab3:
    st.markdown(
        '<span class="coming-soon-badge">🚧 NEXT UPDATE · COMING SOON</span>',
        unsafe_allow_html=True,
    )
    st.header("🚀 Roadmap: Machine Learning for Concrete Strength Prediction")
    st.markdown(
        "In the **next update**, this application will include a **Machine Learning "
        "module that predicts the compressive strength of concrete**."
    )

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        with st.container(border=True):
            st.markdown("### 🏗️ The Civil Engineering Problem")
            st.markdown(
                "In structural design and construction quality assurance, the "
                "**28-day compressive strength** of concrete is the standard measure "
                "of mix adequacy. Conventionally, this requires casting cylinder "
                "specimens and curing them in water tanks for nearly a month before "
                "destructive testing on a **Universal Testing Machine (UTM)**."
            )
            st.markdown(
                "The upcoming model aims to give an early screening estimate of "
                "compressive strength directly from batch mix proportions "
                "(e.g., **Cement Content**, **Water-Cement Ratio**, **Curing Age**) "
                "before casting."
            )
    with col_r2:
        with st.container(border=True):
            st.markdown("### ⚙️ Planned Approach")
            st.markdown(
                "The ML module will be built on top of the same first-principles "
                "Gauss-Jordan engine used in this app:"
            )
            st.markdown(
                "• **Multiple Linear Regression** fitted by least squares\n\n"
                "• **Normal Equations** solved by our own Gauss-Jordan solver: "
                "$(X^T X)\\beta = X^T y$\n\n"
                "• **Model validation** and an interactive mix-design predictor"
            )

    st.info(
        "ℹ️ This feature is **not yet available**. For now, the app focuses on the "
        "general-purpose Gauss-Jordan linear system solver in Tab 1."
    )

# -----------------------------------------------------------------------------
# 5. FOOTER
# -----------------------------------------------------------------------------
st.divider()
st.caption(
    f"🌐 Live app: [{LIVE_APP_URL.replace('https://', '')}]({LIVE_APP_URL}) · "
    "Davao Oriental State University · Faculty of Computing, Engineering and "
    "Technology · Department of Civil Engineering"
)
