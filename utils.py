# utils.py — Firebase logging, timers, and end-of-session tracking
from __future__ import annotations
import uuid
import time
from typing import Optional, Tuple, Dict, Any

import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud import firestore as gcf

FIREBASE_APP_NAME = "default_app_cached"

# --- Initialisation variables DRY ---
def init_page_session_vars():
    defaults = {
        "language_code": "en",
        "language_name": "English",
        "expertise_level": "Beginner",
        "logging_user_id": str(uuid.uuid4()),
        "session_id": str(uuid.uuid4()),
        "current_page": None,
        "acq_src": "",
        "registration_complete": False,
        "user_email": "",
        "final_assess_started": False,
        "modules_opened": [],
        "tickers_tested": set(),
        "feedback_submitted": False,
        "feedback_submitted_session": False,

        "ivs_status": "idle",
        "ivs_data": None,

        "page_timer": {"start": None, "data": {}},
        "page_durations": {},
        "app_version": "2.0",

        "_adv_explore_logged": False,
        "module_timer": {"start": None, "module_id": None},
        "module_durations": {},     
        "module_opens": {},         

        "crosslevel": {
            "beg_on_adv": {"seconds": 0.0, "count": 0},
            "adv_on_beg": {"seconds": 0.0, "count": 0},
        },
        "_adv_beg_counted": False,
    }

    for var, default in defaults.items():
        if var not in st.session_state:
            st.session_state[var] = default

# -------- Langue --------
def get_lang() -> str:
    return st.session_state.get("language_code", "en")

def tr(en_text: str, fr_text: str) -> str:
    return fr_text if get_lang() == "fr" else en_text

# -------- Level helpers (routing only; do NOT normalize what we log) --------
def is_advanced_level(level: str) -> bool:
    """
    Return True if the user's level (in EN/FR) is 'advanced'.
    We intentionally do not normalize the value we store in Firebase.
    """
    s = (level or "").strip().lower()
    return ("avanc" in s) or ("advanced" in s)

def is_beginner_level(level: str) -> bool:
    """
    Return True if the user's level (in EN/FR) is 'beginner'.
    """
    s = (level or "").strip().lower()
    return ("début" in s) or ("debut" in s) or ("begin" in s)

# -------- Analytics stubs --------
def inject_analytics(page_name: str, user_props: dict | None = None): return
def analytics_event(name: str, params: dict | None = None): return

# -------- Firebase init --------
@st.cache_resource(show_spinner=False)
def get_firebase() -> Tuple[Optional["firebase_admin.App"], Optional["firestore.Client"]]:
    if firebase_admin._apps and FIREBASE_APP_NAME in firebase_admin._apps:
        return firebase_admin.get_app(FIREBASE_APP_NAME), firestore.client(app=firebase_admin.get_app(FIREBASE_APP_NAME))

    fb_conf = st.secrets.get("firebase", None)
    if not fb_conf:
        st.warning("Firebase secrets missing.")
        return None, None

    try:
        parsed = dict(fb_conf)
        parsed["private_key"] = parsed["private_key"].replace("\\n", "\n")
        cred = credentials.Certificate(parsed)
        app = firebase_admin.initialize_app(cred, name=FIREBASE_APP_NAME)
        db = firestore.client(app=app)
        print("Firebase initialized.")
        return app, db
    except Exception as e:
        st.error(f"Firebase error: {e}")
        return None, None

# -------- Session-level logs --------
def init_session_tracking():
    if "session_start_time" not in st.session_state:
        st.session_state["session_start_time"] = time.time()
    st.session_state.setdefault("_timers", {})
    st.session_state.setdefault("page_durations", {})

    st.session_state.setdefault("crosslevel", {
        "adv_on_beg": {"count": 0, "seconds": 0.0},
        "beg_on_adv": {"count": 0, "seconds": 0.0},
    })

    # ✅ Normalise d'anciens formats dict -> float
    pd = st.session_state.get("page_durations", {})
    for k, v in list(pd.items()):
        if isinstance(v, dict):
            pd[k] = float(v.get("seconds", 0.0))
        else:
            pd[k] = float(v or 0.0)
    st.session_state["page_durations"] = pd


def finalize_session() -> None:
    _, db = get_firebase()
    if not db:
        return

    user_id = st.session_state.get("logging_user_id", "anonymous")
    session_id = st.session_state.get("session_id", "unknown")
    start_time = st.session_state.get("session_start_time", time.time())
    session_seconds = round(time.time() - start_time, 2)

    # 1) Event "session_duration" dans "events"
    try:
        db.collection("events").add({
            "timestamp": firestore.SERVER_TIMESTAMP,
            "event_name": "session_duration",
            "duration": session_seconds,
            "user_id": user_id,
            "session_id": session_id,
            "app_version": st.session_state.get("app_version", "1.0"),
            "tickers_tested": list(st.session_state.get("tickers_tested", []))
        })
        print(f"Session duration logged: {session_seconds}s")
    except Exception as e:
        print(f"Error logging session duration: {e}")

    # 2) Agrégats dans "users/{user}"
    page_metrics = st.session_state.get("page_durations", {})
    modules_opened = st.session_state.get("modules_opened", [])
    tickers_tested = list(st.session_state.get("tickers_tested", set()))

    # -- by_page: robuste aux deux formats (float OU {"seconds": ...})
    by_page = {}
    for page, data in page_metrics.items():
        if isinstance(data, dict):
            secs = float(data.get("seconds", 0.0))
        else:
            # ex. data est un float déjà cumulé
            secs = float(data or 0.0)
        by_page[page] = {
            "seconds": round(secs, 2),
            "last_ts": firestore.SERVER_TIMESTAMP
        }

    # -- by_module: 1 open par module vu durant la session
    by_module = {}
    for mod in modules_opened:
        by_module[mod] = {
            "opens": 1,
            "last_ts": firestore.SERVER_TIMESTAMP
        }

    # -- crosslevel: symétrique et consolidé au même endroit
    cross = st.session_state.get("crosslevel", {})
    crosslevel_payload = {
        "adv_on_beg": {
            "seconds": round(float(cross.get("adv_on_beg", {}).get("seconds", 0.0)), 2),
            "count": int(cross.get("adv_on_beg", {}).get("count", 0)),
            "last_ts": firestore.SERVER_TIMESTAMP,
        },
        "beg_on_adv": {
            "seconds": round(float(cross.get("beg_on_adv", {}).get("seconds", 0.0)), 2),
            "count": int(cross.get("beg_on_adv", {}).get("count", 0)),
            "last_ts": firestore.SERVER_TIMESTAMP,
        },
    }

    profile_update = {
        "session_id": session_id,
        "session_seconds": session_seconds,
        "tickers_tested": tickers_tested,
        "feedback_submitted": st.session_state.get("feedback_submitted", False),
        "last_active": firestore.SERVER_TIMESTAMP,
        "metrics": {
            "by_page": by_page,
            "by_module": by_module,
            "crosslevel": crosslevel_payload,  # ⬅️ UNE seule clé, symétrique
        },
    }

    try:
        db.collection("users").document(user_id).set(profile_update, merge=True)
        print(f"[DEBUG] Session finalized for {user_id}")
    except Exception as e:
        print(f"[ERROR] Finalizing session: {e}")

# -------- Timer & page-level logs --------
def log_event(event_name: str, page_name: str, event_data: dict | None = None):
    if event_data is None:
        event_data = {}

    _, db = get_firebase()
    if not db:
        print(f"Firebase unavailable for log_event {event_name}")
        return

    try:
        user_id = st.session_state.get("logging_user_id", "anonymous")
        session_id = st.session_state.get("session_id", "unknown")

        event_data.update({
            "timestamp": firestore.SERVER_TIMESTAMP,
            "user_id": user_id,
            "session_id": session_id,
            "page_name": page_name,
            "event_name": event_name,
            "app_version": st.session_state.get("app_version", "1.0"),
            "acq_src": st.session_state.get("acq_src", "unknown"),
        })

        db.collection("events").add(event_data)
        print(f"Logged '{event_name}' on '{page_name}' for {user_id}")
    except Exception as e:
        print(f"ERROR logging event '{event_name}': {e}")

def mark_module_opened(module_id: str, lang: str, level: str, topic: str):
    if "modules_opened" not in st.session_state:
        st.session_state.modules_opened = []

    if module_id not in st.session_state.modules_opened:
        st.session_state.modules_opened.append(module_id)

    st.session_state.last_module_open = {
        "id": module_id,
        "lang": lang,
        "level": level,
        "topic": topic,
        "ts": firestore.SERVER_TIMESTAMP
    }

    # Optional immediate log
    log_event("module_opened", st.session_state.get("current_page", "unknown"), {
        "module_id": module_id,
        "lang": lang,
        "level": level,
        "topic": topic
    })

def start_timer(key: str, initial_data: dict | None = None):
    """Démarre un chrono. initial_data peut contenir {'page_name': PAGE_NAME, ...}"""
    # init containers (idempotent)
    if "page_durations" not in st.session_state:
        st.session_state["page_durations"] = {}
    # start
    st.session_state[key] = {
        "start": time.time(),
        "data": (initial_data or {}).copy()
    }
    # optionnel : garder une trace de la page courante pour le logging par défaut
    if initial_data and "page_name" in initial_data:
        st.session_state["current_page"] = initial_data["page_name"]

def stop_timer_seconds(key: str, event_name: str, log_data: dict | None = None) -> Optional[float]:
    """
    Stoppe le chrono 'key' et renvoie les secondes.
    - Met à jour page_durations[page_name]
    - Ecrit un log_event(event_name, page_name, {..., 'seconds': secs})
    """
    timer = st.session_state.get(key)
    if not timer or timer.get("start") is None:
        return None

    start = float(timer["start"])
    end = time.time()
    seconds = round(end - start, 2)
    # marquer stoppé
    st.session_state[key]["start"] = None

    # payload = data passée à start_timer + log_data additionnel
    payload = dict(timer.get("data") or {})
    if log_data:
        payload.update(log_data)
    payload["seconds"] = seconds

    # déterminer la page
    page_name = payload.get("page_name") or timer.get("data", {}).get("page_name") \
                or st.session_state.get("current_page") or "unknown_page"

    # cumuler la durée par page
    pd = st.session_state.get("page_durations", {})
    pd[page_name] = float(pd.get(page_name, 0.0)) + float(seconds)
    st.session_state["page_durations"] = pd

    # log de l'événement lié à l'arrêt du timer
    try:
        log_event(event_name, page_name, payload)
    except Exception:
        pass

    return seconds

def track_page_view(page_name: str, context: dict | None = None) -> None:
    # 1) log l'event
    log_event("page_view", page_name, context)

    now = time.time()
    prev_page = st.session_state.get("current_page")
    running = st.session_state.get("page_timer", {}).get("start")

    # 2) si on change de page et qu'un timer tournait, crédite l'ancienne page
    if prev_page and prev_page != page_name and running:
        try:
            elapsed = max(0.0, now - float(running))
        except Exception:
            elapsed = 0.0

        durations = st.session_state.get("page_durations", {})
        base = float(durations.get(prev_page, 0.0))
        durations[prev_page] = round(base + elapsed, 2)
        st.session_state["page_durations"] = durations

    # 3) (re)start du timer pour la nouvelle page
    start_timer("page_timer", {"page_name": page_name})
    st.session_state["current_page"] = page_name

def event_with_dwell(event_name: str, page_name: str | None = None, extra: dict | None = None) -> Optional[float]:
    key = "page_timer"
    if key not in st.session_state or st.session_state[key].get("start") is None:
        return None

    seconds = round(time.time() - st.session_state[key]["start"], 2)
    payload = st.session_state[key].get("data", {}).copy()
    if extra:
        payload.update(extra)
    payload["seconds"] = seconds

    log_event(event_name, page_name or st.session_state.get("current_page", "unknown_page"), payload)
    return seconds

def save_user_profile(user_id: str, profile_data: dict):
    _, db = get_firebase()
    if not db:
        return
    try:
        db.collection("users").document(user_id).set(profile_data, merge=True)
        print(f"User profile saved: {user_id}")
    except Exception as e:
        print(f"Error saving user profile: {e}")

def save_final_score(score: int) -> None:
    _, db = get_firebase()
    if not db:
        return

    user_id = st.session_state.get("logging_user_id", "anonymous")
    try:
        db.collection("users").document(user_id).set({
            "final_score": score,
            "final_score_ts": firestore.SERVER_TIMESTAMP
        }, merge=True)
        print(f"[DEBUG] Final quiz score logged for {user_id}")
    except Exception as e:
        print(f"[ERROR] Logging final score: {e}")

def save_quiz_score(level: str, score: int) -> None:
    _, db = get_firebase()
    if not db:
        return

    user_id = st.session_state.get("logging_user_id", "anonymous")
    lvl = (level or "").strip().lower()
    if is_advanced_level(level):
        field_score = "advanced_final_score"
        field_ts    = "advanced_final_score_ts"
    else:
        field_score = "beginner_final_score"
        field_ts    = "beginner_final_score_ts"

    try:
        db.collection("users").document(user_id).set({
            field_score: int(score),
            field_ts: firestore.SERVER_TIMESTAMP
        }, merge=True)
        # Optional: mirror to events stream too
        log_event("quiz_score_saved", st.session_state.get("current_page","unknown"), {
            "level": level, "score": int(score), "field": field_score}
        )
    except Exception as e:
        print(f"[ERROR] save_quiz_score: {e}")

def log_feedback(data: dict):
    _, db = get_firebase()
    if not db:
        return
    try:
        db.collection("feedback").add({
            **data,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "user_id": st.session_state.get("logging_user_id", "anonymous"),
            "session_id": st.session_state.get("session_id", "unknown"),
        })
        st.session_state.feedback_submitted = True
        st.session_state.feedback_submitted_session = True
        print(f"[DEBUG] Feedback logged")
    except Exception as e:
        print(f"[ERROR] Logging feedback: {e}")

def render_feedback_quizz_sidebar():
    Txt = {
        "en": {
            "q1": "How useful was the platform?",
            "q2": "How clear were the explanations?",
            "q3": "How would you rate the design and usability?",
            "q4": "Would you recommend this platform to others?",
            "submit": "Submit feedback",
            "thank_you": "✅ Thank you for your feedback!",
            "cta_missions": "🎯 Finish your missions",
            "cta_finished": "✅ Finished",
        },
        "fr": {
            "q1": "Dans quelle mesure la plateforme était-elle utile ?",
            "q2": "Dans quelle mesure les explications étaient-elles claires ?",
            "q3": "Comment évalueriez-vous le design et l'ergonomie ?",
            "q4": "Recommanderiez-vous cette plateforme à d'autres ?",
            "submit": "Envoyer le feedback",
            "thank_you": "✅ Merci pour votre retour !",
            "cta_missions": "🎯 Reprendre vos missions",
            "cta_finished": "✅ Terminer",
        }
    }

    lang = get_lang()
    labels = Txt.get(lang, Txt["en"])

    q1 = st.slider(labels["q1"], 1, 5, key="fb_q1")
    q2 = st.slider(labels["q2"], 1, 5, key="fb_q2")
    q3 = st.slider(labels["q3"], 1, 5, key="fb_q3")
    q4 = st.slider(labels["q4"], 1, 5, key="fb_q4")

    if st.button(labels["submit"], key="submit_feedback"):
        feedback = {
            "q1": q1,
            "q2": q2,
            "q3": q3,
            "q4": q4,
            "lang": lang,
        }

        # ---- Calcul d'un score global (moyenne sur 5) ----
        feedback_score = (q1 + q2 + q3 + q4) / 4.0

        # ---- Log côté événements ----
        log_event(
            "feedback_submitted",
            st.session_state.get("current_page", "Feedback"),
            {**feedback, "avg_score": feedback_score},
        )

        # ---- Écriture Firestore au bon endroit ----
        _, db = get_firebase()
        if db:
            user_id = st.session_state.get("logging_user_id", "anonymous")
            doc_ref = db.collection("users").document(user_id)
            doc_ref.set(
                {
                    "feedback_submitted": True,
                    "feedback_score": feedback_score,
                    "feedback_details": feedback,
                    "feedback_ts": firestore.SERVER_TIMESTAMP,
                },
                merge=True,
            )

        # ---- État local ----
        st.session_state.feedback_submitted = True
        st.session_state.feedback_submitted_session = True
        st.session_state.feedback_score = feedback_score

        st.success(labels["thank_you"])

        # ---- ✅ nouveaux boutons après envoi ----
        c1, c2 = st.columns(2)
        with c1:
            if st.button(labels["cta_missions"], key="fb_go_main"):
                st.switch_page("pages/Main_App.py")
        with c2:
            if st.button(labels["cta_finished"], key="fb_finish_all"):
                try:
                    finalize_session()
                except Exception:
                    pass
                st.switch_page("Home.py")

def log_crosslevel(content_level: str, seconds: float | None, where: str):
    """
    Log + cumule un événement cross-level si l'utilisateur consulte un contenu d'un autre niveau.
    - content_level : 'Beginner' ou 'Advanced' (niveau DU CONTENU, en EN)
    - seconds : durée (peut être None)
    - where   : page_name
    """
    user_level = st.session_state.expertise_level
    sec = float(seconds or 0.0)

    payload = {
        "user_level": user_level,       # on garde tel quel (FR/EN) pour la trace
        "content_level": content_level, # 'Beginner' / 'Advanced'
        "seconds": sec,
        "count": 1,
        "page": where,
        "lang": st.session_state.language_code,
    }

    key = None
    if is_advanced_level(user_level) and content_level == "Beginner":
        key = "adv_on_beg"
        log_event("adv_on_beg", where, payload)
    elif is_beginner_level(user_level) and content_level == "Advanced":
        key = "beg_on_adv"
        log_event("beg_on_adv", where, payload)
    else:
        # petit debug utile si rien ne matche
        print(f"[crosslevel] no match: user_level={user_level!r} content_level={content_level!r}")

    if key:
        cross = st.session_state.setdefault("crosslevel", {
            "adv_on_beg": {"count": 0, "seconds": 0.0},
            "beg_on_adv": {"count": 0, "seconds": 0.0},
        })
        cross[key]["count"] += 1
        cross[key]["seconds"] += sec
        st.session_state["crosslevel"] = cross
