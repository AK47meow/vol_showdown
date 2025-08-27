import streamlit as st
from utils import (
    init_page_session_vars, track_page_view,
    render_feedback_quizz_sidebar, finalize_session, tr
)

# --- Page config doit être TOUT EN HAUT
st.set_page_config(page_title="Feedback", page_icon="💬")

# --- Early redirect handler
_nav = st.session_state.get("_nav_to")
if _nav:
    st.session_state["_nav_to"] = None
    st.switch_page(_nav)

# Init session
init_page_session_vars()

# Track page view
track_page_view("Feedback")

# Titre
st.title("📝 Feedback")
st.markdown(tr(
    "Please rate the platform and help us improve! 🙏",
    "Merci d’évaluer la plateforme et de nous aider à l’améliorer ! 🙏"
))

# Render le quiz
render_feedback_quizz_sidebar()

# ✅ Bouton central pour clôturer la session (optionnel, double sécurité)
if st.button(tr("✅ End my session", "✅ Clôturer ma session"), key="close_session_main"):
    finalize_session()
    st.success(tr(
        "✅ Session closed successfully!",
        "✅ Session clôturée avec succès !"
    ))
    st.switch_page("Home.py")