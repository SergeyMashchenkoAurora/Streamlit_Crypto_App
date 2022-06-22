import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(f"{SCRIPT_DIR}/../..")

from app.Monitoring import (
    st_read_data,
    DATA_PATH,
    set_sidebar_filters,
    distribution_by,
    display_min_mean_max,
)
from ml.predict import predict_anomalies

MODEL_PATH = SCRIPT_DIR / ".." / ".." / "ml" / "model.joblib"


def main() -> None:

    st.set_page_config(
        page_title="Rainbow Bridge Anomalies", layout="wide",
    )

    st.markdown("# Anomalies detection")
    st.markdown(
        "This page displays anomalies detected in the given period as well as statistics for them. Read more about the model in [the repository](https://github.com/kisialiou/rainbow-bridge-fraud-detection)."
    )

    data_init = (
        st_read_data(DATA_PATH)
        if "data" not in st.session_state
        else st.session_state["data"]
    )
    result_predictions = predict_anomalies(data_init.copy(deep=True), MODEL_PATH)
    predicted_values = data_init.merge(
        result_predictions, on="receiptId", how="inner"
    ).sort_values("blockHeight")
    predicted_values = set_sidebar_filters(predicted_values)

    fig = px.scatter(
        predicted_values,
        x="date",
        y="tokensInUSD",
        hover_data=[
            "date",
            "tokensInUSD",
            "tokenName",
            "signedBy",
            "verdict",
            "anomaly_score",
            "receiptId",
        ],
        color="verdict",
        color_discrete_map={"anomaly": "#FFA500", "ok": "#0000CD"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## List of anomalies")
    st.markdown(
        "List of all detected anomalies in the given period. The greater the `anomaly_score` the greater the confidence in 'anomality'."
    )
    st.dataframe(
        predicted_values.loc[
            predicted_values["verdict"] == "anomaly",
            [
                "receiptId",
                "date",
                "tokensInUSD",
                "tokenName",
                "signedBy",
                "anomaly_score",
            ],
        ].sort_values("anomaly_score", ascending=False)
    )
    anomalies = predicted_values.loc[predicted_values["verdict"] == "anomaly"]
    st.markdown("## Anomalies metrics")
    display_min_mean_max(anomalies)

    st.markdown("## Anomalies tokens distribution")
    st.markdown(
        "What are the most frequent tokens among anomalies? Which transaction value does every token have on average?"
    )
    distribution_by(anomalies, "tokenName")

    st.markdown("## Anomalies senders distribution")
    st.markdown(
        "What are the most frequent senders (`signedBy`) among anomalies? Which transaction value does every sender have on average?"
    )
    distribution_by(anomalies, "signedBy")

    st.markdown("## Anomalies recipients distribution")
    st.markdown(
        "What are the most frequent recipients among anomalies? Which transaction value does every recipient have on average?"
    )
    distribution_by(anomalies, "recipient")


if __name__ == "__main__":
    main()