from __future__ import annotations

import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from bengal_election.pipeline import (
    build_scenario_frame,
    build_training_frame,
    evaluate_bundle,
    load_bundle,
    load_election_data,
    load_legacy_bundle,
    predict_scenario,
)

st.set_page_config(
    page_title="Bengal Election 2026 Forecast Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "AITC": "#1b7f5d",
    "BJP": "#d46a1f",
    "INC": "#2d5baf",
    "CPIM": "#b23a48",
    "AIFB": "#6d597a",
    "CPI": "#7a3e2b",
    "RSP": "#4c6a92",
    "SUCI": "#5f0f40",
    "IND": "#6b7280",
}

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(196, 168, 119, 0.12), transparent 26%),
            linear-gradient(180deg, #f5f1e8 0%, #fbfaf6 48%, #f4efe6 100%);
        color: #18212f;
        font-family: "Aptos", "Segoe UI", sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1320px;
    }
    .hero {
        background: linear-gradient(135deg, #18212f 0%, #24364d 60%, #34536b 100%);
        padding: 2rem 2.2rem;
        border-radius: 24px;
        color: #f6efe2;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 22px 60px rgba(24, 33, 47, 0.16);
        margin-bottom: 1.25rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.03em;
    }
    .hero p {
        margin: 0.65rem 0 0;
        max-width: 62rem;
        line-height: 1.6;
        color: rgba(246, 239, 226, 0.82);
    }
    .summary-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(246,241,232,0.94));
        border: 1px solid rgba(24, 33, 47, 0.08);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        min-height: 132px;
        box-shadow: 0 18px 40px rgba(40, 52, 68, 0.06);
    }
    .summary-card .label {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.75rem;
        color: #5a6676;
        margin-bottom: 0.55rem;
    }
    .summary-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #18212f;
        line-height: 1.1;
    }
    .summary-card .note {
        margin-top: 0.55rem;
        color: #566273;
        font-size: 0.92rem;
    }
    .section-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: #18212f;
        margin-bottom: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_prepared_data() -> pd.DataFrame:
    return build_training_frame(load_election_data())


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    bundle = load_bundle()
    if bundle is not None:
        return bundle
    return load_legacy_bundle()


def format_pct(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "Unavailable"
    return f"{value * 100:.1f}%" if value <= 1 else f"{value:.1f}%"


def build_metric_card(label: str, value: str, note: str) -> str:
    return f"""
    <div class="summary-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="note">{note}</div>
    </div>
    """


df = get_prepared_data()
bundle = get_model_bundle()

st.markdown(
    """
    <div class="hero">
        <h1>Bengal Election 2026 Forecast Studio</h1>
        <p>
            A constituency-level election forecasting workspace built for scenario analysis, historical comparison,
            and model transparency. The interface prioritizes disciplined presentation, richer context, and safer
            fallbacks over decorative dashboard behavior.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if df.empty:
    st.error("The election dataset could not be loaded from `data/raw/West_Bengal_AE.csv`.")
    st.stop()

if bundle is None:
    st.error("No model artifact is available. Run `python train_model.py` to generate the forecast bundle.")
    st.stop()

latest_year = int(df["Year"].max())
latest_df = df[df["Year"] == latest_year].copy()
constituencies = sorted(latest_df["Constituency_Name"].dropna().unique())
metadata = evaluate_bundle(bundle)
metrics = metadata.get("metrics", {})

with st.sidebar:
    st.markdown("## Scenario Controls")
    selected_constituency = st.selectbox("Constituency", constituencies, index=0)
    constituency_snapshot = latest_df[latest_df["Constituency_Name"] == selected_constituency].copy()
    constituency_snapshot = constituency_snapshot.sort_values("Vote_Share_Percentage", ascending=False)
    top_parties = constituency_snapshot["Party"].head(5).tolist()
    st.markdown("Adjust swing assumptions relative to the latest observed vote share.")
    swing_adjustments: dict[str, float] = {}
    for party in top_parties:
        baseline_share = float(
            constituency_snapshot.loc[constituency_snapshot["Party"] == party, "Vote_Share_Percentage"].iloc[0]
        )
        swing_adjustments[party] = st.slider(
            f"{party} swing",
            min_value=-15.0,
            max_value=15.0,
            value=0.0,
            step=0.5,
            help=f"Latest baseline vote share: {baseline_share:.2f}%",
        )
    st.markdown("---")
    artifact_label = "Full reliability bundle" if bundle.get("artifact_version") == 2 else "Legacy fallback bundle"
    st.caption(f"Loaded artifact: {artifact_label}")

scenario_df = build_scenario_frame(
    df,
    constituency_name=selected_constituency,
    swing_adjustments=swing_adjustments,
    tracked_parties=top_parties,
)
prediction_df = predict_scenario(bundle, scenario_df)
if prediction_df.empty:
    st.error(
        "The current artifact could not score any parties for this constituency. Train the new bundle with `python train_model.py` to enable full coverage."
    )
    st.stop()

display_df = prediction_df.sort_values("Win_Probability", ascending=False).copy()
tracked_prediction_df = display_df[display_df["Tracked"]].copy() if "Tracked" in display_df.columns else display_df.copy()
if tracked_prediction_df.empty:
    tracked_prediction_df = display_df.copy()
winner = display_df.iloc[0]
runner_up = display_df.iloc[1] if len(display_df) > 1 else None

vote_share_chart = prediction_df[["Party", "Base_Vote_Share", "Vote_Share_Percentage"]].copy()
vote_share_chart = vote_share_chart.rename(
    columns={"Base_Vote_Share": "Latest Observed Share", "Vote_Share_Percentage": "Scenario Share"}
)
vote_share_chart = vote_share_chart.melt(id_vars="Party", var_name="Series", value_name="Vote Share")
probability_chart = tracked_prediction_df[["Party", "Win_Probability"]].copy()
probability_chart["Win_Probability"] = probability_chart["Win_Probability"] * 100

historical_df = df[
    (df["Constituency_Name"] == selected_constituency)
    & (df["Year"] >= 2001)
    & (~df["Party"].isin({"NOTA"}))
].copy()
historical_df = historical_df.sort_values(["Year", "Vote_Share_Percentage"], ascending=[True, False])
winners_history = historical_df[historical_df["Won"] == 1][
    ["Year", "Party", "Candidate", "Vote_Share_Percentage", "Margin_Percentage"]
]

metric_columns = st.columns(4)
with metric_columns[0]:
    st.markdown(
        build_metric_card("Projected winner", str(winner["Party"]), f"Scenario initialized from the {latest_year} election."),
        unsafe_allow_html=True,
    )
with metric_columns[1]:
    st.markdown(
        build_metric_card("Win probability", f"{winner['Win_Probability'] * 100:.1f}%", "Probability assigned to the top-ranked party."),
        unsafe_allow_html=True,
    )
with metric_columns[2]:
    lead_note = "No second ranked party available."
    lead_value = "Unavailable"
    if runner_up is not None:
        lead_value = f"{(winner['Win_Probability'] - runner_up['Win_Probability']) * 100:.1f} pts"
        lead_note = f"Probability lead over {runner_up['Party']}."
    st.markdown(build_metric_card("Forecast lead", lead_value, lead_note), unsafe_allow_html=True)
with metric_columns[3]:
    baseline_turnout = constituency_snapshot["Turnout_Percentage"].dropna().max()
    st.markdown(
        build_metric_card("Latest turnout", format_pct(float(baseline_turnout) if pd.notna(baseline_turnout) else None), f"Observed in the {latest_year} result."),
        unsafe_allow_html=True,
    )

tab1, tab2, tab3, tab4 = st.tabs(["Scenario Lab", "Constituency Profile", "Model Reliability", "Data Table"])

with tab1:
    left, right = st.columns([1.08, 0.92], gap="large")

    with left:
        st.markdown('<div class="section-title">Win Probability Under Current Scenario</div>', unsafe_allow_html=True)
        fig_probability = px.bar(
            probability_chart.sort_values("Win_Probability"),
            x="Win_Probability",
            y="Party",
            orientation="h",
            color="Party",
            color_discrete_map=PALETTE,
            labels={"Win_Probability": "Win probability (%)", "Party": ""},
        )
        fig_probability.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_probability, use_container_width=True)

    with right:
        st.markdown('<div class="section-title">Scenario Vote Share Mix</div>', unsafe_allow_html=True)
        fig_share = px.bar(
            vote_share_chart,
            x="Vote Share",
            y="Party",
            color="Series",
            orientation="h",
            barmode="group",
            color_discrete_sequence=["#b7c2d0", "#18212f"],
            labels={"Vote Share": "Vote share (%)", "Party": ""},
        )
        fig_share.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="",
        )
        st.plotly_chart(fig_share, use_container_width=True)

    insight_col, table_col = st.columns([0.92, 1.08], gap="large")
    with insight_col:
        st.markdown('<div class="section-title">Scenario Notes</div>', unsafe_allow_html=True)
        st.write(
            f"The simulator starts from the latest observed constituency contest in {latest_year}, applies your swing adjustments, "
            "and rescales vote shares to preserve the constituency total."
        )
        if bundle.get("artifact_version") == 1:
            st.warning(bundle["metadata"]["warning"])
        else:
            st.write(
                "The reliability bundle uses constituency-party lag features, prior winners, turnout context, and candidate continuity indicators."
            )

    with table_col:
        st.markdown('<div class="section-title">Party Ranking</div>', unsafe_allow_html=True)
        ranking_df = display_df[
            ["Party", "Base_Vote_Share", "Vote_Share_Percentage", "Win_Probability", "Candidate", "Position"]
        ].rename(
            columns={
                "Base_Vote_Share": f"{latest_year} vote share",
                "Vote_Share_Percentage": "Scenario vote share",
                "Win_Probability": "Win probability",
                "Candidate": f"{latest_year} candidate",
                "Position": f"{latest_year} finish",
            }
        )
        ranking_df["Win probability"] = (ranking_df["Win probability"] * 100).round(2)
        st.dataframe(ranking_df, use_container_width=True, hide_index=True)

with tab2:
    top_line_col, bottom_line_col = st.columns(2, gap="large")

    with top_line_col:
        st.markdown('<div class="section-title">Historical Vote Share Trend</div>', unsafe_allow_html=True)
        fig_history = px.line(
            historical_df[historical_df["Party"].isin(top_parties)],
            x="Year",
            y="Vote_Share_Percentage",
            color="Party",
            markers=True,
            color_discrete_map=PALETTE,
            labels={"Vote_Share_Percentage": "Vote share (%)", "Year": "Election year"},
        )
        fig_history.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_history, use_container_width=True)

    with bottom_line_col:
        st.markdown('<div class="section-title">Winning Margin Timeline</div>', unsafe_allow_html=True)
        fig_margin = go.Figure()
        fig_margin.add_trace(
            go.Bar(
                x=winners_history["Year"],
                y=winners_history["Margin_Percentage"],
                marker_color=[PALETTE.get(party, "#586272") for party in winners_history["Party"]],
                text=winners_history["Party"],
                textposition="outside",
            )
        )
        fig_margin.update_layout(
            height=420,
            xaxis_title="Election year",
            yaxis_title="Winning margin (%)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=10),
        )
        st.plotly_chart(fig_margin, use_container_width=True)

    st.markdown('<div class="section-title">Winning History</div>', unsafe_allow_html=True)
    st.dataframe(winners_history.sort_values("Year", ascending=False), use_container_width=True, hide_index=True)

with tab3:
    if not metrics:
        st.info("Detailed evaluation metrics will appear here after training the full bundle with `python train_model.py`.")
    else:
        metric_cards = st.columns(4)
        summary_metrics = [
            ("Validation year", str(metrics.get("validation_year", "Unavailable")), "Latest held-out election cycle."),
            ("Constituency accuracy", format_pct(metrics.get("constituency_accuracy")), "Top predicted party vs actual winner."),
            ("Row F1", format_pct(metrics.get("row_f1")), "Winner row classification quality."),
            ("ROC AUC", format_pct(metrics.get("roc_auc")), "Probability ranking quality."),
        ]
        for column, (label, value, note) in zip(metric_cards, summary_metrics):
            with column:
                st.markdown(build_metric_card(label, value, note), unsafe_allow_html=True)

        search_results = pd.DataFrame(metadata.get("candidate_search", []))
        feature_importance = pd.DataFrame(metadata.get("feature_importance", []))

        search_col, importance_col = st.columns(2, gap="large")
        with search_col:
            st.markdown('<div class="section-title">Model Selection Results</div>', unsafe_allow_html=True)
            st.dataframe(search_results, use_container_width=True, hide_index=True)
        with importance_col:
            st.markdown('<div class="section-title">Top Transformed Features</div>', unsafe_allow_html=True)
            if feature_importance.empty:
                st.info("Feature importance is unavailable for the current artifact.")
            else:
                fig_importance = px.bar(
                    feature_importance.sort_values("importance"),
                    x="importance",
                    y="feature",
                    orientation="h",
                    color="importance",
                    color_continuous_scale=["#b09b74", "#18212f"],
                    labels={"importance": "Importance", "feature": ""},
                )
                fig_importance.update_layout(
                    height=480,
                    coloraxis_showscale=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_importance, use_container_width=True)

with tab4:
    explorer_year = st.selectbox(
        "Reference election year",
        options=sorted(df["Year"].astype(int).unique().tolist(), reverse=True),
        index=0,
    )
    explorer_df = df[(df["Constituency_Name"] == selected_constituency) & (df["Year"] == explorer_year)].copy()
    explorer_df = explorer_df.sort_values(["Position", "Vote_Share_Percentage"], ascending=[True, False])
    st.dataframe(
        explorer_df[
            [
                "Year",
                "Party",
                "Candidate",
                "Votes",
                "Vote_Share_Percentage",
                "Position",
                "Margin_Percentage",
                "Turnout_Percentage",
                "Age",
                "Incumbent",
                "Recontest",
                "Turncoat",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    f"Data source: `data/raw/West_Bengal_AE.csv`. Latest election used for scenario initialization: {latest_year}. "
    "Run `python train_model.py` after installing dependencies to refresh the production forecast bundle."
)
