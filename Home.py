# Home.py — Registration + Source capture (EN/FR)
print("DEBUG: Home.py - New version loaded. Confirming Firebase functions are present.")

import re
import uuid
import streamlit as st
from firebase_admin import firestore
from utils import (
    get_firebase, log_event, save_user_profile,
    tr, event_with_dwell, track_page_view, finalize_session, init_session_tracking, mark_module_opened, start_timer, stop_timer_seconds, init_page_session_vars, render_feedback_quizz_sidebar
)

# --- Initialisation session_state ---
init_page_session_vars()

# --- Mesurer le nb de modules ouverts ---
if "modules_opened" not in st.session_state:
    st.session_state.modules_opened = []

# --- Définir LANG maintenant que language_code est dans la session ---
PAGE_NAME = "Home"
LANG = st.session_state.get("language_code", "en")
ACQ = st.session_state.acq_src
LEVEL = st.session_state.expertise_level

# --- Initialisation Firebase ---
db, _ = get_firebase()
init_session_tracking()
uid = st.session_state.logging_user_id
track_page_view(PAGE_NAME, {"lang": LANG})
start_timer("page_timer", {"page_name": PAGE_NAME})

# --- Logique de redirection après inscription ---
if st.session_state.registration_complete:
    st.session_state.registration_complete = False
    st.switch_page("pages/Main_App.py")

# ---------- Texts i18n ----------
def T(lang):
    if lang == "fr":
        return dict(
            title="Bienvenue sur Volatility Showdown",
            subtitle="Inscription rapide",
            language="Langue de l’interface",
            levels_title="Niveau d'expertise",
            levels=["Débutant", "Avancé"],
            email_label="Adresse e-mail",
            email_ph="votre.email@exemple.com",
            acq_label="Comment nous avez-vous connus ?",
            acq_opts=["Université", "LinkedIn", "J'ai été forcé(e)", "E-mail", "Conférence", "Recherche web", "Autre"],
            acq_details="Précisez (facultatif)",
            consent="J’accepte la politique de confidentialité.",
            enter="🚀 Entrer sur la Plateforme",
            email_err="Veuillez entrer une adresse e-mail valide.",
            consent_err="Vous devez accepter la politique de confidentialité.",
            saved="Inscription réussie ! Redirection..."
        )
    return dict(
        title="Welcome to Volatility Showdown",
        subtitle="Quick Registration",
        language="Interface Language",
        levels_title="Expertise Level",
        levels=["Beginner", "Advanced"],
        email_label="Email Address",
        email_ph="your.email@example.com",
        acq_label="How did you hear about us?",
        acq_opts=["University", "LinkedIn", "I was forced", "Email", "Conference", "Web Search", "Other"],
        acq_details="Please specify (optional)",
        consent="I accept the privacy policy.",
        enter="🚀 Enter the Platform",
        email_err="Please enter a valid email address.",
        consent_err="You must accept the privacy policy.",
        saved="Registration successful! Redirecting..."
    )

Txt = T(LANG)

# ---------- Helpers ----------
def _is_valid_email(email: str) -> bool:
    return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

def _email_domain(email: str) -> str:
    return email.split('@')[-1] if '@' in email else ""

def _acq_src_label(acq: str, details: str) -> str:
    return f"{acq}: {details}" if acq in ("Autre", "Other") and details else acq

# ---------- Interface ----------
st.set_page_config(page_title="Volatility Showdown — Home", layout="centered")
st.title(Txt["title"])
st.subheader(Txt["subtitle"])

with st.expander("Parameters"):
    col1, col2 = st.columns(2)

    with col1:
        selected_lang = st.selectbox(
            Txt["language"],
            ["English", "Français"],
            index=1 if st.session_state.language_code == "fr" else 0
        )
        st.session_state.language_code = "fr" if selected_lang == "Français" else "en"
        st.session_state.language_name = selected_lang

    with col2:
        available_levels = Txt["levels"]
        current_level = st.session_state.expertise_level

        # Conversion FR/EN si incohérence avec la langue active
        if LANG == "fr" and current_level == "Advanced":
            current_level = "Avancé"
        elif LANG == "fr" and current_level == "Beginner":
            current_level = "Débutant"
        elif LANG == "en" and current_level == "Avancé":
            current_level = "Advanced"
        elif LANG == "en" and current_level == "Débutant":
            current_level = "Beginner"

        st.session_state.expertise_level = current_level

        selected_level = st.selectbox(
            Txt["levels_title"],
            options=available_levels,
            index=available_levels.index(current_level)
        )
        st.session_state.expertise_level = selected_level

    acq = st.selectbox(Txt["acq_label"], options=Txt["acq_opts"])
    acq_details = st.text_input(Txt["acq_details"])
    st.session_state.acq_src = _acq_src_label(acq, acq_details)

# ---------- Formulaire ----------
with st.form("registration_form"):
    email = st.text_input(Txt["email_label"], placeholder=Txt["email_ph"])
    consent = st.checkbox(Txt["consent"])
    submit = st.form_submit_button(Txt["enter"])

if submit:
    if not _is_valid_email(email):
        st.error(Txt["email_err"])
        st.stop()
    if not consent:
        st.error(Txt["consent_err"])
        st.stop()

    domain = _email_domain(email)
    profile = {
        "email": email,
        "email_domain": domain,
        "has_email": True,
        "acquisition_source": st.session_state.acq_src,
        "language": st.session_state.language_name,
        "expertise_level": st.session_state.expertise_level,
        "session_id": st.session_state.session_id,
        "created_at": firestore.SERVER_TIMESTAMP,
        "last_active": firestore.SERVER_TIMESTAMP,
    }

    if db:
        save_user_profile(uid, profile)

    log_event("registration_submit", PAGE_NAME, {
        "lang": LANG,
        "level": st.session_state.expertise_level,
        "acq_src": st.session_state.acq_src,
        "has_email": 1,
        "email_domain": domain,
        "user_id": uid,
        "session_id": st.session_state.session_id
    })

    st.session_state.user_email = email
    st.session_state.registration_complete = True
    st.success(Txt["saved"])
    finalize_session()
    st.rerun()

with st.expander("🧭 Platform Guide / Guide de la plateforme"):
    lang = st.session_state.language_code

    if lang == "en":
        st.markdown("### 🎮 Main Quest")
        st.markdown("- ✅ Take the Final Quiz to prove your knowledge!")

        st.markdown("### 🧩 Side Quests")
        st.markdown("- 🗣 Leave your opinion in the Feedback Quiz (sidebar)")
        st.markdown("- 📚 Read the educational modules")

        st.markdown("### 🛡 Missions")
        st.markdown("- 💹 Test different tickers (e.g., AAPL, TSLA, GOOG...)")
        st.markdown("- 📥 Download the options data (CSV)")

        st.markdown("### 📊 Progress Tracking")

    else:
        st.markdown("### 🎮 Quête Principale")
        st.markdown("- ✅ Passez le Quiz final pour valider vos connaissances !")

        st.markdown("### 🧩 Quêtes annexes")
        st.markdown("- 🗣 Donnez votre avis via le quiz de feedback (barre latérale)")
        st.markdown("- 📚 Lisez les modules pédagogiques")

        st.markdown("### 🛡 Missions")
        st.markdown("- 💹 Testez différents tickers (ex: AAPL, TSLA, GOOG...)")
        st.markdown("- 📥 Téléchargez les données d'options (CSV)")

        st.markdown("### 📊 Suivi de progression")


# ---------- Lien direct si déjà enregistré ----------
if st.session_state.user_email:
    st.markdown(f"✅ Welcome back, **{st.session_state.user_email}**!")
    if st.button("Go to Main App"):
        finalize_session()
        st.switch_page("pages/Main_App.py")
