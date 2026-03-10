"""Ghostfolio portfolio tracker embedded via iframe."""

import streamlit as st
import streamlit.components.v1 as components

ghostfolio_url = st.session_state.get("ghostfolio_url", "https://ghostfolio.renesis.pl")

st.markdown(
    """
    <style>
        /* Remove all Streamlit container padding - target all known versions */
        .main .block-container,
        [data-testid="stMainBlockContainer"],
        [data-testid="block-container"],
        section[data-testid="stMain"] > div,
        .stMainBlockContainer {
            padding: 0 !important;
            max-width: 100% !important;
        }
        /* Remove top header bar */
        [data-testid="stHeader"], header[data-testid="stHeader"] {
            display: none !important;
        }
        /* Remove the app view container padding */
        [data-testid="stAppViewContainer"] {
            padding-top: 0 !important;
        }
        /* Force iframe to fill everything */
        [data-testid="stIframe"] iframe,
        .stIframe iframe,
        iframe {
            width: 100% !important;
            height: calc(100vh - 10px) !important;
            border: none !important;
            display: block !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

components.iframe(ghostfolio_url, height=900, scrolling=True)
