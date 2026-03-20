import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches

st.set_page_config(page_title="Professional Nonlinear Pushover Dashboard", layout="wide")

# =========================================================
# HELPERS
# =========================================================
def hinge_label(val: float) -> str:
    if val < 0.10:
        return "Elastic"
    elif val < 0.35:
        return "IO"
    elif val < 0.60:
        return "LS"
    elif val < 0.85:
        return "CP"
    return "Failed"


def hinge_color(label: str) -> str:
    cmap = {
        "Elastic": "#4caf50",
        "IO": "#8bc34a",
        "LS": "#ffeb3b",
        "CP": "#ff9800",
        "Failed": "#f44336",
    }
    return cmap.get(label, "#9e9e9e")


def build_default_dataframe(n_storey: int) -> pd.DataFrame:
    rows = []
    for i in range(n_storey):
        rows.append(
            {
                "Storey": i + 1,
                "Height_m": 3.0,
                "Weight_kN": 800.0,
                "Columns": 4,
                "EI_col_kNm2": 2.0e7,
                "Mp_col_kNm": 250.0,
                "Mp_beam_kNm": 180.0,
                "n_bays": 3,
                "bay_width_m": 5.0,
                "col_width_m": 0.40,
                "beam_depth_m": 0.50,
            }
        )
    return pd.DataFrame(rows)


def compute_storey_stiffness(df: pd.DataFrame) -> np.ndarray:
    h = df["Height_m"].values.astype(float)
    ncol = df["Columns"].values.astype(float)
    ei = df["EI_col_kNm2"].values.astype(float)
    # Simplified equivalent lateral storey stiffness
    k = 12.0 * ei * ncol / np.maximum(h**3, 1e-9)
    return k


def compute_yield_story_shear(df: pd.DataFrame) -> np.ndarray:
    h = df["Height_m"].values.astype(float)
    mp_col = df["Mp_col_kNm"].values.astype(float)
    mp_beam = df["Mp_beam_kNm"].values.astype(float)
    vy = (mp_col + mp_beam) / np.maximum(h, 1e-9)
    return vy


def get_load_pattern(df: pd.DataFrame, pattern_name: str, user_pattern_text: str = "") -> np.ndarray:
    n = len(df)
    w = df["Weight_kN"].values.astype(float)
    hcum = np.cumsum(df["Height_m"].values.astype(float))

    if pattern_name == "Uniform":
        p = np.ones(n)
    elif pattern_name == "Triangular":
        p = np.arange(1, n + 1, dtype=float)
    elif pattern_name == "First-mode-like":
        p = w * hcum
    elif pattern_name == "User-defined":
        try:
            vals = [float(x.strip()) for x in user_pattern_text.split(",")]
            if len(vals) != n:
                st.warning(f"User-defined pattern must have {n} values. Using uniform instead.")
                p = np.ones(n)
            else:
                p = np.array(vals, dtype=float)
        except Exception:
            st.warning("Could not parse user-defined pattern. Using uniform instead.")
            p = np.ones(n)
    else:
        p = np.ones(n)

    if np.allclose(np.sum(p), 0.0):
        p = np.ones(n)

    return p / np.sum(p)


def run_pushover(
    df: pd.DataFrame,
    load_pattern: np.ndarray,
    n_steps: int,
    delta_lambda: float,
    post_yield_ratio: float,
    pdelta_factor: float,
):
    n = len(df)

    h = df["Height_m"].values.astype(float)
    k0 = compute_storey_stiffness(df)
    vy = compute_yield_story_shear(df)

    lambda_hist = []
    base_shear_hist = []
    roof_disp_hist = []
    drift_matrix = []
    hinge_state_history = []
    story_shear_history = []

    cumulative_damage = np.zeros(n)
    roof_disp = 0.0
    lam = 0.0

    for step in range(n_steps):
        lam += delta_lambda
        story_force = lam * load_pattern * np.sum(df["Weight_kN"].values.astype(float))
        story_shear = np.flip(np.cumsum(np.flip(story_force)))

        effective_k = np.zeros(n)
        hinge_states_numeric = np.zeros(n)

        for i in range(n):
            demand_ratio = story_shear[i] / max(vy[i], 1e-9)

            # Damage progression from demand ratio
            cumulative_damage[i] = max(cumulative_damage[i], min(demand_ratio, 1.25))
            dmg = cumulative_damage[i]
            hinge_states_numeric[i] = dmg

            # Stiffness degradation
            if dmg < 0.10:
                degradation = 1.00
            elif dmg < 0.35:
                degradation = 0.75
            elif dmg < 0.60:
                degradation = 0.45
            elif dmg < 0.85:
                degradation = max(post_yield_ratio, 0.10)
            else:
                degradation = max(post_yield_ratio * 0.35, 0.03)

            # P-Delta severity reduces effective stiffness
            degradation *= (1.0 - pdelta_factor * 0.35)
            degradation = max(degradation, 0.02)

            effective_k[i] = k0[i] * degradation

        story_drifts = story_shear / np.maximum(effective_k, 1e-9)
        floor_disp = np.cumsum(story_drifts)
        roof_disp = floor_disp[-1]
        base_shear = story_shear[0]

        lambda_hist.append(lam)
        base_shear_hist.append(base_shear)
        roof_disp_hist.append(roof_disp)
        drift_matrix.append(story_drifts / np.maximum(h, 1e-9))
        hinge_state_history.append(hinge_states_numeric.copy())
        story_shear_history.append(story_shear.copy())

        # stop when most storeys failed
        if np.sum(cumulative_damage >= 0.85) >= max(1, int(np.ceil(0.4 * n))):
            break

    results = {
        "lambda": np.array(lambda_hist),
        "base_shear": np.array(base_shear_hist),
        "roof_disp": np.array(roof_disp_hist),
        "drift_ratios": np.array(drift_matrix),
        "hinge_states_numeric": np.array(hinge_state_history),
        "story_shear": np.array(story_shear_history),
        "k0": k0,
        "vy": vy,
    }
    return results


def idealize_bilinear_curve(x: np.ndarray, y: np.ndarray):
    if len(x) < 3:
        return x, y, x[-1] if len(x) else 0.0

    vmax = np.max(y)
    ix_peak = np.argmax(y)
    dy = np.gradient(y, x, edge_order=1)
    k_initial = max(dy[0], 1e-9)

    vy = 0.60 * vmax
    dy_yield = vy / k_initial

    dxu = x[ix_peak]
    bil_x = np.array([0.0, dy_yield, dxu])
    bil_y = np.array([0.0, vy, vmax])

    target_disp = 0.8 * dxu
    return bil_x, bil_y, target_disp


def soft_storey_flags(k0: np.ndarray) -> np.ndarray:
    flags = np.zeros(len(k0), dtype=bool)
    for i in range(1, len(k0)):
        if k0[i] < 0.70 * k0[i - 1]:
            flags[i] = True
    return flags


def plot_capacity_curve(roof_disp, base_shear, bil_x, bil_y, target_disp):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(roof_disp, base_shear, linewidth=2, label="Pushover Curve")
    ax.plot(bil_x, bil_y, "--", linewidth=2, label="Idealized Bilinear")
    ax.axvline(target_disp, linestyle=":", linewidth=2, label="Target Roof Displacement")
    ax.set_xlabel("Roof Displacement (m)")
    ax.set_ylabel("Base Shear (kN)")
    ax.set_title("Capacity Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_drift_profile(df: pd.DataFrame, drift_ratios_final: np.ndarray):
    y = np.cumsum(df["Height_m"].values.astype(float))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(drift_ratios_final, y, marker="o", linewidth=2)
    ax.set_xlabel("Interstorey Drift Ratio")
    ax.set_ylabel("Elevation (m)")
    ax.set_title("Final Drift Profile")
    ax.grid(True, alpha=0.3)
    return fig


def plot_frame_elevation(df: pd.DataFrame, hinge_states=None, title="Frame Elevation"):
    n_storey = len(df)
    total_width = int(df["n_bays"].max()) * float(df["bay_width_m"].iloc[0])

    y_levels = [0.0]
    for h in df["Height_m"].values.astype(float):
        y_levels.append(y_levels[-1] + h)

    fig, ax = plt.subplots(figsize=(10, 7))

    for i in range(n_storey):
        n_bays = int(df.loc[i, "n_bays"])
        bay_width = float(df.loc[i, "bay_width_m"])
        col_width = float(df.loc[i, "col_width_m"])
        beam_depth = float(df.loc[i, "beam_depth_m"])

        y_bot = y_levels[i]
        y_top = y_levels[i + 1]

        state = "Elastic"
        if hinge_states is not None and i < len(hinge_states):
            state = hinge_states[i]
        color = hinge_color(state)

        # Columns
        for j in range(n_bays + 1):
            x = j * bay_width
            ax.add_patch(
                patches.Rectangle(
                    (x - col_width / 2, y_bot),
                    col_width,
                    y_top - y_bot,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1.0,
                    alpha=0.85,
                )
            )

        # Beam
        for j in range(n_bays):
            x1 = j * bay_width
            ax.add_patch(
                patches.Rectangle(
                    (x1, y_top - beam_depth / 2),
                    bay_width,
                    beam_depth,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1.0,
                    alpha=0.85,
                )
            )

    # Labels
    for i in range(n_storey):
        y_mid = 0.5 * (y_levels[i] + y_levels[i + 1])
        ax.text(total_width + 0.8, y_mid, f"S{i+1}", va="center", fontsize=9)

    legend_handles = [
        patches.Patch(color=hinge_color("Elastic"), label="Elastic"),
        patches.Patch(color=hinge_color("IO"), label="IO"),
        patches.Patch(color=hinge_color("LS"), label="LS"),
        patches.Patch(color=hinge_color("CP"), label="CP"),
        patches.Patch(color=hinge_color("Failed"), label="Failed"),
    ]

    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_xlim(-1, total_width + 3)
    ax.set_ylim(0, y_levels[-1] + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Frame Width (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25)
    return fig


def create_download_bundle(inputs_df, summary_df, curve_df, hinge_df):
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("inputs.csv", inputs_df.to_csv(index=False))
        zf.writestr("summary.csv", summary_df.to_csv(index=False))
        zf.writestr("capacity_curve.csv", curve_df.to_csv(index=False))
        zf.writestr("hinge_states.csv", hinge_df.to_csv(index=False))
    mem_zip.seek(0)
    return mem_zip


# =========================================================
# UI
# =========================================================
st.title("🏢 Professional Nonlinear Pushover Dashboard")
st.caption("Teaching-grade simplified pushover simulator for frame buildings up to 10 storeys.")

with st.sidebar:
    st.header("Global Controls")
    n_storey = st.slider("Number of Storeys", 1, 10, 5)
    pattern_name = st.selectbox(
        "Lateral Load Pattern",
        ["Uniform", "Triangular", "First-mode-like", "User-defined"],
    )
    user_pattern_text = ""
    if pattern_name == "User-defined":
        user_pattern_text = st.text_input(
            "User Pattern (comma-separated, bottom to top)",
            value="1,2,3,4,5",
        )

    n_steps = st.slider("Pushover Steps", 20, 300, 100, 10)
    delta_lambda = st.number_input("Load Increment Factor", min_value=0.001, value=0.02, step=0.005)
    post_yield_ratio = st.slider("Post-yield Stiffness Ratio", 0.01, 0.30, 0.08, 0.01)
    pdelta_level = st.selectbox("P-Delta Severity", ["Off", "Low", "Moderate", "High"])
    pdelta_map = {"Off": 0.00, "Low": 0.20, "Moderate": 0.45, "High": 0.70}
    pdelta_factor = pdelta_map[pdelta_level]

if "frame_df" not in st.session_state or len(st.session_state["frame_df"]) != n_storey:
    st.session_state["frame_df"] = build_default_dataframe(n_storey)

st.subheader("Storey Input Table")
edited_df = st.data_editor(
    st.session_state["frame_df"],
    num_rows="fixed",
    use_container_width=True,
    key="frame_editor",
)
st.session_state["frame_df"] = edited_df.copy()

c_run, c_reset = st.columns([1, 1])
run_btn = c_run.button("Run Pushover Analysis", use_container_width=True)
reset_btn = c_reset.button("Reset Defaults", use_container_width=True)

if reset_btn:
    st.session_state["frame_df"] = build_default_dataframe(n_storey)
    st.rerun()

# =========================================================
# ANALYSIS
# =========================================================
if run_btn:
    df = st.session_state["frame_df"].copy()

    required_cols = [
        "Storey", "Height_m", "Weight_kN", "Columns", "EI_col_kNm2",
        "Mp_col_kNm", "Mp_beam_kNm", "n_bays", "bay_width_m",
        "col_width_m", "beam_depth_m"
    ]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            st.stop()

    df = df.sort_values("Storey").reset_index(drop=True)

    load_pattern = get_load_pattern(df, pattern_name, user_pattern_text)
    results = run_pushover(
        df=df,
        load_pattern=load_pattern,
        n_steps=n_steps,
        delta_lambda=delta_lambda,
        post_yield_ratio=post_yield_ratio,
        pdelta_factor=pdelta_factor,
    )

    roof_disp = results["roof_disp"]
    base_shear = results["base_shear"]
    hinge_num_final = results["hinge_states_numeric"][-1]
    hinge_labels_final = [hinge_label(v) for v in hinge_num_final]
    drift_final = results["drift_ratios"][-1]
    k0 = results["k0"]
    vy = results["vy"]

    bil_x, bil_y, target_disp = idealize_bilinear_curve(roof_disp, base_shear)
    soft_flags = soft_storey_flags(k0)

    # Summary metrics
    max_base_shear = float(np.max(base_shear))
    max_roof_disp = float(np.max(roof_disp))
    final_roof_disp = float(roof_disp[-1])
    max_drift = float(np.max(drift_final))
    failed_count = int(np.sum(np.array(hinge_labels_final) == "Failed"))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peak Base Shear (kN)", f"{max_base_shear:,.2f}")
    m2.metric("Max Roof Disp. (m)", f"{max_roof_disp:,.4f}")
    m3.metric("Target Disp. (m)", f"{target_disp:,.4f}")
    m4.metric("Max Drift Ratio", f"{max_drift:.4f}")

    st.divider()

    left, right = st.columns([1.1, 1.0])

    with left:
        st.pyplot(plot_capacity_curve(roof_disp, base_shear, bil_x, bil_y, target_disp))

    with right:
        st.pyplot(plot_drift_profile(df, drift_final))

    st.divider()

    st.subheader("Frame Visualization")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Initial Frame**")
        fig_init = plot_frame_elevation(
            df,
            hinge_states=["Elastic"] * len(df),
            title="Initial Frame",
        )
        st.pyplot(fig_init)

    with c2:
        st.markdown("**Frame After Pushover**")
        fig_dmg = plot_frame_elevation(
            df,
            hinge_states=hinge_labels_final,
            title="Frame Hinge State",
        )
        st.pyplot(fig_dmg)

    st.divider()

    result_table = pd.DataFrame({
        "Storey": df["Storey"],
        "Height_m": df["Height_m"],
        "Weight_kN": df["Weight_kN"],
        "Initial_Stiffness_kN_per_m": k0,
        "Yield_Story_Shear_kN": vy,
        "Final_Drift_Ratio": drift_final,
        "Soft_Storey_Flag": soft_flags,
        "Final_Hinge_State": hinge_labels_final,
    })

    st.subheader("Storey Results")
    st.dataframe(result_table, use_container_width=True)

    st.subheader("Load Pattern")
    load_pattern_df = pd.DataFrame({
        "Storey": df["Storey"],
        "Normalized_Load_Pattern": load_pattern,
    })
    st.dataframe(load_pattern_df, use_container_width=True)

    st.subheader("Interpretation")
    notes = []
    if np.any(soft_flags):
        soft_storeys = ", ".join([str(int(s)) for s in df["Storey"][soft_flags].tolist()])
        notes.append(f"Soft-storey tendency detected at storey/storeys: {soft_storeys}.")
    if failed_count > 0:
        notes.append(f"{failed_count} storey/storeys reached Failed hinge state.")
    if max_drift > 0.02:
        notes.append("Maximum drift ratio exceeded 2%, which may indicate severe nonlinear response.")
    elif max_drift > 0.01:
        notes.append("Maximum drift ratio exceeded 1%, indicating significant inelastic action.")
    else:
        notes.append("Drift remained relatively moderate in this simplified analysis.")
    notes.append("This dashboard is a simplified surrogate model for teaching and concept studies, not a replacement for full nonlinear FEM.")

    for n in notes:
        st.write(f"- {n}")

    # Download data
    curve_df = pd.DataFrame({
        "Roof_Displacement_m": roof_disp,
        "Base_Shear_kN": base_shear,
    })

    hinge_df = pd.DataFrame({
        "Storey": df["Storey"],
        "Final_Hinge_State": hinge_labels_final,
        "Final_Hinge_Numeric": hinge_num_final,
    })

    summary_df = pd.DataFrame([{
        "Peak_Base_Shear_kN": max_base_shear,
        "Final_Roof_Displacement_m": final_roof_disp,
        "Max_Roof_Displacement_m": max_roof_disp,
        "Target_Roof_Displacement_m": target_disp,
        "Max_Drift_Ratio": max_drift,
        "Failed_Storey_Count": failed_count,
        "PDelta_Level": pdelta_level,
        "PostYield_Stiffness_Ratio": post_yield_ratio,
        "Load_Pattern": pattern_name,
    }])

    bundle = create_download_bundle(df, summary_df, curve_df, hinge_df)

    st.download_button(
        "Download Results ZIP",
        data=bundle,
        file_name="pushover_results.zip",
        mime="application/zip",
        use_container_width=True,
    )

else:
    st.info("Set your frame properties, then click **Run Pushover Analysis**.")
