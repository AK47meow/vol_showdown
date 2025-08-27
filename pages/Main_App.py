# pages/Main_App.py 
import time
import uuid
import unicodedata
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np

from logic import run_analysis
from utils import (
    get_firebase, log_event, tr,
    track_page_view, start_timer, stop_timer_seconds,
    event_with_dwell, finalize_session, init_session_tracking, init_page_session_vars, render_feedback_quizz_sidebar, is_advanced_level, is_beginner_level
)

# --- Page config ---
st.set_page_config(page_title="IVS — Main App", page_icon="🏦", layout="wide")

# ---------- Graphique ----------
def render_ivs_surface(xi, yi, zi):
    fig = go.Figure(data=[
        go.Surface(x=xi, y=yi, z=zi, colorscale="Viridis", showscale=True)
    ])
    fig.update_layout(
        title="Implied Volatility Surface",
        margin=dict(l=10, r=10, t=30, b=10),
        scene=dict(
            xaxis=dict(title="Moneyness (K/S)"),
            yaxis=dict(title="Time to Expiry (Years)"),
            zaxis=dict(title="Implied Volatility"),
        )
    )
    return fig

# --- Initialisation session_state ---
init_page_session_vars()


# ----------- Navigation --------------
_nav = st.session_state.get("_nav_to")
if _nav:
    st.session_state["_nav_to"] = None
    st.switch_page(_nav)

# --- Langue et métadonnées ---
LANG = st.session_state.language_code
ACQ = st.session_state.acq_src
LEVEL = st.session_state.expertise_level
PAGE_NAME = "Main_App"

# --- Mesurer le nb de modules ouverts ---
if "modules_opened" not in st.session_state:
    st.session_state.modules_opened = []

# --- Firebase + analytics ---
db, _ = get_firebase()
track_page_view(PAGE_NAME, {"lang": LANG})
init_session_tracking()
start_timer("page_timer", {"page_name": PAGE_NAME})


# --- Session status banner + CTAs (FR/EN) ---
feedback_done_now = bool(st.session_state.get("feedback_submitted_session"))

st.markdown("---")
if not feedback_done_now:
    st.info(tr(
        "👋 Before you leave, please complete the feedback to close your session.",
        "👋 Avant de partir, merci de compléter le feedback pour clôturer votre session."
    ))
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button(tr("📝 Go to Feedback", "📝 Aller au Feedback"), key="go_feedback_from_main"):
            seconds = stop_timer_seconds("page_timer", "page_leave", {"to": "Feedback", "from": PAGE_NAME})
            st.session_state["_nav_to"] = "pages/Feedback.py"
            st.rerun()
    with c2:
        st.caption(tr(
            "Your session will be finalized after you submit the feedback.",
            "Votre session sera finalisée après l’envoi du feedback."
        ))
else:
    st.success(tr(
        "✅ Feedback submitted. You can now close your session.",
        "✅ Feedback envoyé. Vous pouvez maintenant clôturer votre session."
    ))
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button(tr("✅ End my session", "✅ Clôturer ma session"), key="close_session_from_main"):
            try:
                finalize_session()
            except Exception:
                pass
            st.session_state["_nav_to"] = "Home.py"
            st.rerun()
    with c2:
        if st.button(tr("🏠 Back to Home", "🏠 Retour à l’accueil"), key="back_home_from_main"):
            seconds = stop_timer_seconds("page_timer", "page_leave", {"to": "Home", "from": PAGE_NAME})
            st.session_state["_nav_to"] = "Home.py"
            st.rerun()
st.markdown("---")

# --- Sidebar: Feedback + Missions --- #
with st.sidebar:
    st.markdown("## " + tr("🎮 Missions", "🎮 Missions"))

    # Ticker progress
    num_tickers = len(st.session_state.get("tickers_tested", []))
    ticker_score = min(num_tickers / 5, 1.0)

    # Module progress
    num_modules = len(st.session_state.get("modules_opened", []))
    module_score = min(num_modules / 2, 1.0)

    # ✅ Final quiz: rely on session flags set at submission time
    quiz_done = bool(st.session_state.get("quiz_final_done"))
    final_quiz_score = st.session_state.get("quiz_final_score")  # int or None
    quiz_score = 1.0 if quiz_done else 0.0

    # ✅ Feedback quiz: use session flag set when feedback is submitted
    feedback_done = bool(st.session_state.get("feedback_submitted_session"))
    feedback_score = 1.0 if feedback_done else 0.0

    # Global progress
    total = (ticker_score + module_score + quiz_score + feedback_score) / 4
    st.markdown(f"**{tr('Global progress', 'Progression globale')}**: `{int(total*100)}%`")
    st.progress(total)

    st.markdown("### " + tr("✅ Objectives", "✅ Objectifs"))
    st.markdown(tr(f"- 💹 Tested tickers: **{num_tickers}/5**", f"- 💹 Tickers testés : **{num_tickers}/5**"))
    st.markdown(tr(f"- 📚 Modules opened: **{num_modules}/2**", f"- 📚 Modules ouverts : **{num_modules}/2**"))
    st.markdown(tr(
        f"- 🧠 Final quiz: {'✅' if quiz_done else '❌'}" + (f"  (score {final_quiz_score}/5)" if final_quiz_score is not None else ""),
        f"- 🧠 Quizz final : {'✅' if quiz_done else '❌'}" + (f"  (score {final_quiz_score}/5)" if final_quiz_score is not None else "")
    ))
    st.markdown(tr(
        f"- 🗣 Feedback: {'✅' if feedback_done else '❌'}",
        f"- 🗣 Feedback : {'✅' if feedback_done else '❌'}"
    ))

# ---------- Traductions ----------
def T(lang):
    if lang == "fr":
        return dict(
            title="Surface de Volatilité Implicite (IVS)",
            desc="Visualisez la surface de volatilité implicite d’une action cotée.",
            ticker_label="Entrer un ticker (ex: AAPL)",
            option_type_label="Type d'option",
            load_button="Charger la chaîne d’options",
            loading_msg="Récupération des données de marché & calculs…",
            ready_msg="Analyse terminée — IVS prête ✅",
            error_msg="Impossible de calculer l’IVS. Réessaie.",
            not_enough_data="Pas assez de points de données pour construire une IVS significative.",
            open_modules="Ouvrir le module d'introduction",
            start_assess="Démarrer l'évaluation finale",
            df_title="Données de la chaîne d'options",
            df_note="Ces données proviennent de votre fonction run_analysis dans logic.py."
        )
    return dict(
        title="Implied Volatility Surface (IVS)",
        desc="Visualize the implied volatility surface for a listed stock.",
        ticker_label="Enter a ticker (e.g. AAPL)",
        option_type_label="Option type",
        load_button="Load Option Chain",
        loading_msg="Fetching market data & running calculations…",
        ready_msg="Analysis complete — IVS ready ✅",
        error_msg="Could not calculate IVS. Please try again.",
        not_enough_data="Not enough valid data points to build a meaningful IVS.",
        open_modules="Open Introduction module",
        start_assess="Start Final Assessment",
        df_title="Options Chain Data",
        df_note="These data come from your run_analysis function in logic.py."
    )

Txt = T(LANG)

# ---------- UI ----------
st.title(Txt["title"])
st.markdown(Txt["desc"])
st.markdown("---")

# --- Params de la page + Enregistrement KPI nb Tickers ---
ticker = st.text_input(Txt["ticker_label"], value="AAPL")
if "tickers_tested" not in st.session_state:
    st.session_state.tickers_tested = set()
st.session_state.tickers_tested.add(ticker.upper())

option_type = st.selectbox(Txt["option_type_label"], options=["c", "p"], index=0)
col1, col2 = st.columns(2)

# ---------- Analyse IVS ----------
if col1.button(Txt["load_button"]):
    st.session_state.ivs_status = "loading"
    try:
        with st.spinner(Txt["loading_msg"]):
            contracts, xi, yi, zi = run_analysis(ticker, option_type)
        st.session_state.ivs_data = {"contracts": contracts, "xi": xi, "yi": yi, "zi": zi}
        st.session_state.ivs_status = "ready"
        if "tickers_tested" not in st.session_state:
            st.session_state.tickers_tested = set()

        log_event("ivs_loaded", PAGE_NAME, {
            "ok": True, "ticker": ticker, "option_type": option_type
        })
    except Exception as e:
        st.session_state.ivs_status = "error"
        st.error(f"Erreur : {e}")
        log_event("ivs_loaded", PAGE_NAME, {
            "ok": False, "ticker": ticker, "option_type": option_type,
            "error": str(e)
        })

# ---------- Affichage résultat ----------
status = st.session_state.ivs_status
ivs_data = st.session_state.ivs_data

if status == "ready":
    st.success(Txt["ready_msg"])
    if ivs_data and ivs_data["contracts"] is not None and not ivs_data["contracts"].empty:
        contracts_df = ivs_data["contracts"]
        st.subheader(Txt["df_title"])
        st.dataframe(ivs_data["contracts"], use_container_width=True)
        st.info(Txt["df_note"])
        csv_bytes = contracts_df.to_csv(index=False).encode("utf-8")

# ---------- Download Options data CSV pour étudiants --------------------        
        st.download_button(
            label="⬇️ Download options data (CSV)",
            data=csv_bytes,
            file_name=f"options_{ticker.upper()}_{option_type}.csv",
            mime="text/csv",
            key="download_options_csv"
        )

    if ivs_data and ivs_data["zi"] is not None:
        fig = render_ivs_surface(ivs_data["xi"], ivs_data["yi"], ivs_data["zi"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(Txt["not_enough_data"])

elif status == "loading":
    st.info(Txt["loading_msg"])
elif status == "error":
    st.warning(Txt["error_msg"])

# ---------- Navigation ----------
st.markdown("---")
ASSESSMENT_ADV = "pages/Final_Quizz_Advanced.py"
ASSESSMENT_BEG = "pages/Final_Quizz_Beginner.py"

col1, col2 = st.columns(2)

with col1:
    # Module button (route by level, FR/EN-safe using helpers)
    if is_beginner_level(LEVEL):
        if st.button(tr("📚 Open Introduction module", "📚 Ouvrir le module d'introduction")):
            stop_timer_seconds("page_timer", "page_leave", {"to": "Introduction_to_IVS", "from": "Main_App", "reason": "open_module"})
            st.session_state["_nav_to"] = "pages/Introduction to IVS.py"
            st.rerun()
    else:  # advanced
        if st.button(tr("📚 Open Advanced module", "📚 Ouvrir le module avancé")):
            stop_timer_seconds("page_timer", "page_leave", {"to": "Implied_Volatility_Surfaces", "from": "Main_App", "reason": "open_module"})
            st.session_state["_nav_to"] = "pages/Implied Volatility Surfaces.py"
            st.rerun()

with col2:
    # Quiz button (route by level)
    if is_beginner_level(LEVEL):
        if st.button(tr("🧠 Start Beginner Quiz", "🧠 Démarrer le Quizz Débutant")):
            stop_timer_seconds("page_timer", "page_leave", {"to": "Quizz/Beginner", "from": "Main_App", "reason": "start_assess"})
            log_event("assessment_start_click", PAGE_NAME, {"level": LEVEL, "acq_src": ACQ})
            st.session_state["_nav_to"] = ASSESSMENT_BEG
            st.rerun()
    else:
        if st.button(tr("🧠 Start Advanced Quiz", "🧠 Démarrer le Quizz Avancé")):
            stop_timer_seconds("page_timer", "page_leave", {"to": "Quizz/Advanced", "from": "Main_App", "reason": "start_assess"})
            log_event("assessment_start_click", PAGE_NAME, {"level": LEVEL, "acq_src": ACQ})
            st.session_state["_nav_to"] = ASSESSMENT_ADV
            st.rerun()

# ----------- Navigation --------------
_nav = st.session_state.get("_nav_to")
if _nav:
    st.session_state["_nav_to"] = None
    st.switch_page(_nav)
