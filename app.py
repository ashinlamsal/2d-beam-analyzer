import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from solver import Load, Support, sample_diagrams, solve_beam

st.set_page_config(page_title="2D Beam Analyzer", layout="wide")
st.title("2D Beam Analyzer")
st.caption("Direct Stiffness Method (DSM) finite-element analysis. Positive transverse loads are upward; positive moment is sagging.")


def diagram_summary(data):
    labels = {"shear": "Shear force, V", "moment": "Bending moment, M", "axial": "Axial force, N", "deflection": "Deflection, v"}
    rows = []
    for key, name in labels.items():
        values = data[key]
        maximum, minimum, absolute = int(np.argmax(values)), int(np.argmin(values)), int(np.argmax(np.abs(values)))
        rows.append({"Result": name, "Maximum": values[maximum], "x at maximum": data["x"][maximum], "Minimum": values[minimum], "x at minimum": data["x"][minimum], "Maximum absolute": values[absolute], "x at |max|": data["x"][absolute]})
    return pd.DataFrame(rows)


def clamp_positions_to_span():
    """Keep any retained Streamlit position inputs within a newly entered span."""
    span = st.session_state.span
    for key in st.session_state:
        if key.startswith(("sx", "pos", "start", "end")) and isinstance(st.session_state[key], (int, float)):
            st.session_state[key] = min(max(st.session_state[key], 0.0), span)


with st.sidebar:
    st.header("Beam properties")
    st.caption("The span is user-defined. Support and load positions use this entered span.")
    L = st.number_input("Beam span L", min_value=0.001, value=6.0, step=0.5, format="%.3f", key="span", on_change=clamp_positions_to_span)
    E = st.number_input("Elastic modulus E", min_value=0.001, value=200e9, format="%.6g")
    I = st.number_input("Moment of inertia I", min_value=1e-9, value=8e-6, format="%.8g")
    A = st.number_input("Cross-sectional area A", min_value=1e-9, value=0.01, format="%.8g")
    st.header("Supports")
    n_supports = st.number_input("Number of supports", 1, 8, 2)
    supports = []
    for i in range(n_supports):
        columns = st.columns(2)
        x = columns[0].number_input(f"Support {i + 1} position", 0.0, float(L), 0.0 if i == 0 else float(L), key=f"sx{i}")
        kind = columns[1].selectbox(f"Support {i + 1} type", ["pinned", "roller", "fixed"], key=f"sk{i}")
        supports.append(Support(x, kind))
    st.header("Loads")
    n_loads = st.number_input("Number of loads", 0, 12, 1)
    loads = []
    for i in range(n_loads):
        kind = st.selectbox(f"Load {i + 1} type", ["point", "udl", "uvl", "axial"], key=f"type{i}")
        if kind in {"point", "axial"}:
            columns = st.columns(2)
            magnitude = columns[0].number_input(f"Load {i + 1} magnitude", value=-10.0, key=f"mag{i}")
            x = columns[1].number_input(f"Load {i + 1} position", 0.0, float(L), float(L) / 2, key=f"pos{i}")
            loads.append(Load(kind, magnitude, x))
        else:
            q0 = st.number_input(f"Load {i + 1} start magnitude", value=-5.0, key=f"q0{i}")
            q1 = q0 if kind == "udl" else st.number_input(f"Load {i + 1} end magnitude", value=-10.0, key=f"q1{i}")
            columns = st.columns(2)
            start = columns[0].number_input(f"Load {i + 1} start", 0.0, float(L), 0.0, key=f"start{i}")
            end = columns[1].number_input(f"Load {i + 1} end", 0.0, float(L), float(L), key=f"end{i}")
            loads.append(Load(kind, q0, start=start, end=end, magnitude_end=q1))
    run = st.button("Analyze beam", type="primary", use_container_width=True)

if not run:
    st.info("Set the span, supports, and loads in the sidebar, then select Analyze beam.")
    st.stop()

try:
    result = solve_beam(L, E, I, A, supports, loads)
except ValueError as error:
    st.error(str(error))
    st.stop()

data = sample_diagrams(result)
summary = diagram_summary(data)
results_tab, max_tab, reactions_tab = st.tabs(["Diagrams & free-body diagram", "Maximum forces", "Support reactions"])

with results_tab:
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.055, subplot_titles=("Free-body diagram", "Shear force diagram", "Bending moment diagram", "Axial force diagram", "Deflection curve"))
    fig.add_trace(go.Scatter(x=[0, L], y=[0, 0], mode="lines", line=dict(color="#263238", width=7), hoverinfo="skip"), 1, 1)
    for support in supports:
        label, symbol = {"pinned": ("Pin", "triangle-up"), "roller": ("Roller", "circle"), "fixed": ("Fixed", "square")}[support.kind]
        if support.kind == "fixed": fig.add_vline(x=support.x, line_width=7, line_color="#1565c0", row=1, col=1)
        else: fig.add_trace(go.Scatter(x=[support.x], y=[-0.12], mode="markers+text", marker=dict(symbol=symbol, size=18, color="#1565c0"), text=[label], textposition="bottom center", hovertemplate=f"{label}<br>x=%{{x:.3g}}<extra></extra>"), 1, 1)
    for load in loads:
        if load.kind in {"point", "axial"}:
            vertical = load.kind == "point"; color = "#d32f2f" if vertical else "#6a1b9a"; y = 0.24 if vertical else 0.06
            symbol = "arrow-down" if vertical and load.magnitude < 0 else "arrow-up"
            fig.add_trace(go.Scatter(x=[load.x], y=[y], mode="markers+text", marker=dict(symbol=symbol, size=20, color=color), text=[f"{load.kind.title()} {load.magnitude:g}"], textposition="top center", hovertemplate=f"{load.kind.title()} load<br>x=%{{x:.3g}}<extra></extra>"), 1, 1)
        else:
            xs = np.linspace(load.start, load.end, 7); avg = (load.magnitude + load.magnitude_end) / 2
            fig.add_trace(go.Scatter(x=xs, y=np.full(7, .25), mode="markers+lines", marker=dict(symbol="arrow-down" if avg < 0 else "arrow-up", size=14, color="#d32f2f"), line=dict(color="#d32f2f", width=2), text=[load.kind.upper()] * 7, hovertemplate="%{text}<br>x=%{x:.3g}<extra></extra>"), 1, 1)
    for row, key, name, color in [(2,"shear","V","#1976d2"),(3,"moment","M","#ef6c00"),(4,"axial","N","#6a1b9a"),(5,"deflection","v","#00897b")]:
        values = data[key]; index = int(np.argmax(np.abs(values)))
        fig.add_trace(go.Scatter(x=data["x"], y=values, fill="tozeroy", name=name, line=dict(color=color), hovertemplate="x=%{x:.5g}<br>value=%{y:.5g}<extra></extra>"), row, 1)
        fig.add_trace(go.Scatter(x=[data["x"][index]], y=[values[index]], mode="markers+text", marker=dict(color=color, size=8), text=[f"|max| = {values[index]:.4g}"], textposition="top center", showlegend=False), row, 1)
    fig.update_yaxes(range=[-0.35, .45], visible=False, row=1, col=1)
    fig.update_layout(height=1180, showlegend=False, hovermode="x unified", margin=dict(t=90))
    fig.update_xaxes(title_text="Position along beam", row=5, col=1)
    st.plotly_chart(fig, use_container_width=True)

with max_tab:
    st.subheader("Maximum and minimum developed actions")
    st.caption("Values are sampled along the beam. Maximum absolute retains the sign of the governing action.")
    st.dataframe(summary.style.format({column: "{:.6g}" for column in summary.columns if column != "Result"}), use_container_width=True, hide_index=True)
    metrics = st.columns(4)
    for column, (_, row) in zip(metrics, summary.iterrows()): column.metric(row["Result"], f"{row['Maximum absolute']:.5g}", f"at x = {row['x at |max|']:.5g}")

with reactions_tab:
    indices = [i for i, value in enumerate(result.reactions) if abs(value) > 1e-8]
    st.subheader("Support reactions")
    st.dataframe(pd.DataFrame({"DOF": [f"Node {i // 3 + 1}: {['u', 'v', 'θ'][i % 3]}" for i in indices], "Reaction": [result.reactions[i] for i in indices]}), use_container_width=True, hide_index=True)
