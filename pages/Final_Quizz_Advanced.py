# pages/Final_Quizz_Advanced.py
import streamlit as st
from utils import (
    init_page_session_vars, init_session_tracking, get_firebase,
    track_page_view, start_timer, stop_timer_seconds, event_with_dwell,
    log_event, tr, save_quiz_score, is_advanced_level, finalize_session, log_crosslevel
)

st.set_page_config(page_title="Final Quiz — Advanced", page_icon="🧠", layout="wide")
init_page_session_vars()


# ---- Session / tracking
init_session_tracking()

LANG  = st.session_state.language_code
LEVEL = st.session_state.expertise_level
PAGE_NAME = "Quizz/Advanced"


track_page_view(PAGE_NAME, {"lang": LANG, "level": LEVEL})
start_timer("page_timer", {"page_name": PAGE_NAME, "level": LEVEL})

# ---- Text dictionary
def T(lang: str):
    if lang == "fr":
        return dict(
            title="Évaluation Finale — Avancé",
            intro="Questions orientées calibration, greeks et IVS.",
            softgate_msg="Ce quizz est destiné aux utilisateurs **Avancé**. Vous pouvez continuer si vous le souhaitez.",
            softgate_continue="Continuer quand même",
            softgate_go_beg="Aller au quizz Débutant",
            submit="Valider mes réponses",
            must_answer="Répondez à toutes les questions avant de valider.",
            score_prefix="Votre score",
            out_of="sur",
            nav_main="🏠 Retour à l’application principale",
            nav_module="📚 Revoir le module avancé",
            q=[
                dict(q="La méthode de Newton–Raphson met à jour σ via…", choices=[
                    "σ ← σ − (Prix(σ) − Prix_mkt)/Vega(σ)",
                    "σ ← σ + (Prix(σ) − Prix_mkt)×Vega(σ)"
                ], correct="σ ← σ − (Prix(σ) − Prix_mkt)/Vega(σ)"),
                dict(q="Un ‘guess’ initial efficace pour σ peut être…", choices=[
                    "Le point d’inflexion de f(σ)=BSM(σ)−Prix_mkt",
                    "Toujours 10%"
                ], correct="Le point d’inflexion de f(σ)=BSM(σ)−Prix_mkt"),
                dict(q="Dans un hedge delta discret, le P&L dépend surtout de…", choices=[
                    "la convexité (gamma) et du chemin",
                    "uniquement du taux sans risque"
                ], correct="la convexité (gamma) et du chemin"),
                dict(q="Le ‘skew’ négatif typique sur actions reflète…", choices=[
                    "une prime de protection pour les baisses",
                    "une erreur de marché sans fondement"
                ], correct="une prime de protection pour les baisses"),
                dict(q="La structure par terme (term structure) de l’IV…", choices=[
                    "peut varier avec le régime de volatilité et le risque d’événement",
                    "est toujours plate"
                ], correct="peut varier avec le régime de volatilité et le risque d’événement"),
            ]
        )
    return dict(
        title="Final Quiz — Advanced",
        intro="Questions on calibration, greeks, and IV surfaces.",
        softgate_msg="This quiz is intended for **Advanced** users. You can still continue if you wish.",
        softgate_continue="Continue anyway",
        softgate_go_beg="Go to Beginner quiz",
        submit="Submit answers",
        must_answer="Please answer all questions before submitting.",
        score_prefix="Your score",
        out_of="out of",
        nav_main="🏠 Back to Main App",
        nav_module="📚 Review Advanced module",
        q=[
            dict(q="Newton–Raphson updates σ via…", choices=[
                "σ ← σ − (Price(σ) − P_mkt)/Vega(σ)",
                "σ ← σ + (Price(σ) − P_mkt)×Vega(σ)"
            ], correct="σ ← σ − (Price(σ) − P_mkt)/Vega(σ)"),
            dict(q="A good initial ‘guess’ for σ can be…", choices=[
                "The inflection point of f(σ)=BSM(σ)−Price_mkt",
                "Always 10%"
            ], correct="The inflection point of f(σ)=BSM(σ)−Price_mkt"),
            dict(q="In discrete delta hedging, P&L depends strongly on…", choices=[
                "convexity (gamma) and the path",
                "only the risk-free rate"
            ], correct="convexity (gamma) and the path"),
            dict(q="The typical negative equity skew reflects…", choices=[
                "a protection premium against downside",
                "a baseless market error"
            ], correct="a protection premium against downside"),
            dict(q="The IV term structure…", choices=[
                "can vary with vol regime and event risk",
                "is always flat"
            ], correct="can vary with vol regime and event risk"),
        ]
    )

TXT = T(LANG)

# ---- Soft gate (beginners landing here)
if not is_advanced_level(LEVEL) and not st.session_state.get("_adv_quiz_override", False):
    st.info(TXT["softgate_msg"])
    c1, c2 = st.columns(2)
    with c1:
        if st.button(TXT["softgate_continue"], key="adv_soft_continue"):
            st.session_state["_adv_quiz_override"] = True
            log_event("advanced_quiz_opened_by_beginner", PAGE_NAME, {"level": LEVEL, "lang": LANG})
            st.rerun()
    with c2:
        if st.button(TXT["softgate_go_beg"], key="adv_soft_go_beg"):
            stop_timer_seconds("page_timer", "page_leave", {"to": "Quizz/Beginner", "from": "Quizz/Advanced"})
            st.session_state["_nav_to"] = "pages/Final_Quizz_Beginner.py"
            st.rerun()
    st.stop()

# One-shot log that quiz page was opened
if not st.session_state.get("_adv_quiz_open_logged"):
    log_event("quiz_start", PAGE_NAME, {"level": LEVEL, "lang": LANG})
    st.session_state["_adv_quiz_open_logged"] = True

# ---- UI
st.title(TXT["title"])
st.caption(TXT["intro"])
st.markdown("---")

with st.form("advanced_quiz_form"):
    answers = []
    for i, item in enumerate(TXT["q"], start=1):
        ans = st.radio(f"{i}. {item['q']}", item["choices"], index=None, key=f"adv_q{i}")
        answers.append(ans)
    submitted = st.form_submit_button(TXT["submit"])

if submitted:
    if any(a is None for a in answers):
        st.warning(TXT["must_answer"])
    else:
        # per-question logs + score
        correct_flags = []
        for i, item in enumerate(TXT["q"], start=1):
            ok = (answers[i-1] == item["correct"])
            correct_flags.append(ok)
            log_event("quiz_answer", PAGE_NAME, {
                "i": i, "question": item["q"], "answer": answers[i-1],
                "correct": ok, "level": LEVEL, "lang": LANG
            })
        score = sum(1 for x in correct_flags if x)

        # Save in session for safe reuse
        st.session_state["_adv_quiz_score"] = score
        st.session_state["_adv_quiz_completed"] = True

        # dwell time and persist
        seconds = event_with_dwell("quiz_submit", PAGE_NAME, {
            "score": score, "total": len(TXT["q"]), "level": LEVEL, "lang": LANG
        })
        save_quiz_score("Advanced", score)

        # --- KPI cross-level ---
        log_event("quiz_final", PAGE_NAME, {
            "level": LEVEL, "final_score": score, "total": len(TXT["q"]), "lang": LANG
        })

        # Stop timer => durée de la page
        page_secs = stop_timer_seconds("page_timer", "page_submit_end", {
            "from": "Quizz/Advanced", "score": score, "total": len(TXT["q"])
        })

        # Cross-level dérivé : contenu Advanced
        log_crosslevel(content_level="Advanced", seconds=page_secs or seconds, where=PAGE_NAME)

        try:
            finalize_session()
        except Exception as e:
            print(f"[WARN] finalize_session after adv quiz: {e}")

        st.success(f"{TXT['score_prefix']}: **{score} {TXT['out_of']} {len(TXT['q'])}**")
        if seconds is not None:
            st.caption(tr(f"Time on quiz: {seconds:.1f}s", f"Temps passé : {seconds:.1f}s"))

        # ------ Navigation (safe) -----------
st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    if st.button(TXT["nav_main"], key="adv_nav_main"):
        nav_secs = stop_timer_seconds("page_timer", "page_leave",
                                      {"to": "Main_App", "from": "Quizz/Advanced"})

        log_crosslevel(content_level="Advanced", seconds=nav_secs, where=PAGE_NAME)

        log_event("adv_quiz_nav_main", PAGE_NAME, {"level": LEVEL, "lang": LANG})
        st.switch_page("pages/Main_App.py")

with c2:
    if st.button(TXT["nav_module"], key="adv_nav_module"):
        nav_secs = stop_timer_seconds("page_timer", "page_leave",
                                      {"to": "Implied_Volatility_Surfaces", "from": "Quizz/Advanced"})
        from utils import log_crosslevel
        log_crosslevel(content_level="Advanced", seconds=nav_secs, where=PAGE_NAME)

        log_event("adv_quiz_nav_module", PAGE_NAME, {"level": LEVEL, "lang": LANG})
        st.switch_page("pages/Implied Volatility Surfaces.py")
