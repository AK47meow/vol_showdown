# pages/Final_Quizz_Beginner.py
import streamlit as st
from utils import (
    init_page_session_vars, init_session_tracking, get_firebase,
    track_page_view, start_timer, stop_timer_seconds, event_with_dwell,
    log_event, tr, save_quiz_score, finalize_session, log_crosslevel
)

st.set_page_config(page_title="Final Quiz — Beginner", page_icon="🧩", layout="wide")

# ---- Session / tracking
init_page_session_vars()
init_session_tracking()

LANG  = st.session_state.language_code
LEVEL = st.session_state.expertise_level
PAGE_NAME = "Quizz/Beginner"

# --- Safe nav guard: jump out early if a nav target is set

track_page_view(PAGE_NAME, {"lang": LANG, "level": LEVEL})
start_timer("page_timer", {"page_name": PAGE_NAME, "level": LEVEL})

# ---- Text dictionary
def T(lang: str):
    if lang == "fr":
        return dict(
            title="Évaluation Finale — Débutant",
            intro="Testez les notions clés vues dans le module d’introduction.",
            submit="Valider mes réponses",
            must_answer="Répondez à toutes les questions avant de valider.",
            score_prefix="Votre score",
            out_of="sur",
            nav_main="🏠 Retour à l’application principale",
            nav_module="📚 Revoir le module d’introduction",
            nav_adv_module="🚀 Passer au module avancé (oui tu as été élu, t'as l'air de maîtriser ce que tu fais)",
            q=[
                dict(q="Une option est-elle une obligation d’acheter/vendre ?", choices=["Oui", "Non"], correct="Non"),
                dict(q="La volatilité implicite reflète surtout…", choices=["des profits passés", "les anticipations du marché"], correct="les anticipations du marché"),
                dict(q="Un put OTM coûte souvent plus cher en régime de 'smirk' baissier.", choices=["Faux", "Vrai"], correct="Vrai"),
                dict(q="Le risque perçu et le risque mesuré coïncident toujours.", choices=["Vrai", "Faux"], correct="Faux"),
                dict(q="L’‘IVS’ cartographie l’IV selon K/S et…", choices=["la maturité T", "le spread bid-ask"], correct="la maturité T"),
            ]
        )
    return dict(
        title="Final Quiz — Beginner",
        intro="Check the core ideas from the introduction module.",
        submit="Submit answers",
        must_answer="Please answer all questions before submitting.",
        score_prefix="Your score",
        out_of="out of",
        nav_main="🏠 Back to Main App",
        nav_module="📚 Review Introduction module",
        nav_adv_module="🚀 Go to Advanced module (I wasn't expecting that from you)",
        q=[
            dict(q="Is an option a *mandatory* buy/sell contract?", choices=["Yes", "No"], correct="No"),
            dict(q="Implied volatility mostly reflects…", choices=["past profits", "market expectations"], correct="market expectations"),
            dict(q="In a bearish ‘smirk’, OTM puts are typically pricier.", choices=["False", "True"], correct="True"),
            dict(q="Perceived risk and measured risk always coincide.", choices=["True", "False"], correct="False"),
            dict(q="The IVS maps IV by moneyness K/S and…", choices=["maturity T", "bid-ask spread"], correct="maturity T"),
        ]
    )

TXT = T(LANG)

# ---- UI
st.title(TXT["title"])
st.caption(TXT["intro"])
st.markdown("---")

# One-shot log that quiz page was opened
if not st.session_state.get("_beg_quiz_open_logged"):
    log_event("quiz_start", PAGE_NAME, {"level": LEVEL, "lang": LANG})
    st.session_state["_beg_quiz_open_logged"] = True

with st.form("beginner_quiz_form"):
    answers = []
    for i, item in enumerate(TXT["q"], start=1):
        ans = st.radio(f"{i}. {item['q']}", item["choices"], index=None, key=f"beg_q{i}")
        answers.append(ans)
    submitted = st.form_submit_button(TXT["submit"])

if submitted:
    # Validate
    if any(a is None for a in answers):
        st.warning(TXT["must_answer"])
    else:
        # Score + per-question logs
        correct_flags = []
        for i, item in enumerate(TXT["q"], start=1):
            ok = (answers[i-1] == item["correct"])
            correct_flags.append(ok)
            log_event("quiz_answer", PAGE_NAME, {
                "i": i, "question": item["q"], "answer": answers[i-1],
                "correct": ok, "level": LEVEL, "lang": LANG
            })
        score = sum(1 for x in correct_flags if x)

        # Time on quiz (dwell)
        seconds = event_with_dwell("quiz_submit", PAGE_NAME, {
            "score": score, "total": len(TXT["q"]), "level": LEVEL, "lang": LANG
        })

        # Persist score in users/{uid}
        save_quiz_score("Beginner", score)
# --- Pivot event (same name as Advanced quiz) for cross-level KPIs by level ---
        log_event("quiz_final", PAGE_NAME, {
            "level": LEVEL, "final_score": score, "total": len(TXT["q"]), "lang": LANG
        })

        # --- Stop the page timer to get accurate duration ---
        page_secs = stop_timer_seconds("page_timer", "page_submit_end", {
            "from": PAGE_NAME, "score": score, "total": len(TXT["q"])
        })

        # --- Cross-level for Beginner content (if user is Advanced, emits adv_on_beg) ---
        log_crosslevel(content_level="Beginner", seconds=(page_secs or seconds), where=PAGE_NAME)

        # --- Sidebar flags so the UI can immediately show the quiz checkmark ---
        st.session_state["quiz_final_done"]  = True
        st.session_state["quiz_final_score"] = int(score)
        st.session_state["_sidebar_nonce"]   = st.session_state.get("_sidebar_nonce", 0) + 1


        try:
            finalize_session()
        except Exception as e:
            print(f"[WARN] finalize_session after beg quiz: {e}")

        st.success(f"{TXT['score_prefix']}: **{score} {TXT['out_of']} {len(TXT['q'])}**")
        if seconds is not None:
            st.caption(tr(f"Time on quiz: {seconds:.1f}s", f"Temps passé : {seconds:.1f}s"))
        st.session_state["_beg_quiz_completed"] = True
        st.session_state["_beg_quiz_score"] = int(score)

        # ------- Navigation (toujours hors du if submitted) ------------
if st.session_state.get("_beg_quiz_completed", False):
    st.markdown("---")
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button(TXT["nav_main"], key="beg_nav_main"):
            nav_secs = stop_timer_seconds("page_timer", "page_leave",
                       {"to": "Main_App", "from": PAGE_NAME}) or 0.0
            log_crosslevel(content_level="Beginner", seconds=nav_secs, where=PAGE_NAME)
            log_event("beg_quiz_nav_main", PAGE_NAME, {"level": LEVEL, "lang": LANG})
            st.switch_page("pages/Main_App.py")

    with c2:
        if st.button(TXT["nav_module"], key="beg_nav_module"):
            nav_secs = stop_timer_seconds("page_timer", "page_leave",
                       {"to": "Introduction_to_IVS", "from": PAGE_NAME}) or 0.0
            log_crosslevel(content_level="Beginner", seconds=nav_secs, where=PAGE_NAME)
            log_event("beg_quiz_nav_module", PAGE_NAME, {"level": LEVEL, "lang": LANG})
            st.switch_page("pages/Introduction to IVS.py")

    with c3:
        if st.session_state.get("_beg_quiz_score", 0) >= 4:
            if st.button(TXT["nav_adv_module"], key="beg_nav_adv_module"):
                nav_secs = stop_timer_seconds("page_timer", "page_leave",
                           {"to": "Implied_Volatility_Surfaces", "from": PAGE_NAME}) or 0.0
                # contenu = Advanced → si user=Beginner, ça log beg_on_adv
                log_crosslevel(content_level="Advanced", seconds=nav_secs, where=PAGE_NAME)
                log_event("beg_quiz_nav_adv_module", PAGE_NAME,
                          {"level": LEVEL, "lang": LANG, "score": st.session_state["_beg_quiz_score"]})
                st.switch_page("pages/Implied Volatility Surfaces.py")
        else:
            st.button(TXT["nav_adv_module"], key="beg_nav_adv_module_disabled", disabled=True)