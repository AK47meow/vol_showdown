# pages/Introduction to IVS.py
import uuid
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from utils import (
    get_firebase, log_event, tr,
    track_page_view, start_timer, stop_timer_seconds,
    event_with_dwell, finalize_session, init_session_tracking,
    mark_module_opened, init_page_session_vars, log_crosslevel
)

# --- Early redirect handler (top of the page)
_nav = st.session_state.get("_nav_to")
if _nav:
    st.session_state["_nav_to"] = None
    st.switch_page(_nav)

st.set_page_config(page_title="Module — Introduction to IVS", page_icon="📚", layout="wide")

# ---------------- Session / tracking ----------------
init_page_session_vars()
init_session_tracking()  # set session_start_time once

LANG  = st.session_state.language_code
ACQ   = st.session_state.acq_src
LEVEL = st.session_state.expertise_level
PAGE_NAME = "Modules/Introduction_to_IVS"  


mark_module_opened(
    module_id=("fr_beg_implied_volatility" if LANG == "fr" else "en_beg_implied_volatility"),
    lang=LANG,
    level=LEVEL,
    topic="Implied Volatility"
)

if "modules_opened" not in st.session_state:
    st.session_state.modules_opened = []

_, db = get_firebase()
track_page_view(PAGE_NAME)
start_timer("page_timer", {"page_name": PAGE_NAME})

# --- Promotion eligibility (>= 4/5 at Beginner final quiz, from session_state) ---
beg_final_ok = bool(st.session_state.get("quiz_final_done")) and int(st.session_state.get("quiz_final_score", 0)) >= 4
st.session_state["beg_promo_eligible"] = beg_final_ok

# --- One-shot impression of the promo CTA when eligible
if beg_final_ok and not st.session_state.get("_beg_promo_impression_logged", False):
    log_event("promo_to_advanced_impression", PAGE_NAME, {
        "level": LEVEL, "lang": LANG,
        "score": int(st.session_state.get("quiz_final_score", 0)),
        "eligible": True
    })
    st.session_state["_beg_promo_impression_logged"] = True

# ---- Cross-level: if an Advanced user opens the Beginner module, count 1 visit (once per session)
if (LEVEL == "Advanced") and (not st.session_state.get("_beg_open_counted", False)):
    log_crosslevel(content_level="Beginner", seconds=0.0, where=PAGE_NAME)
    st.session_state["_beg_open_counted"] = True
    log_event("crosslevel_advanced_open_beginner", PAGE_NAME, {"level": LEVEL, "lang": LANG})

if st.session_state.get("_switch_page_main"):
    st.session_state._switch_page_main = False
    st.switch_page("pages/Main_App.py")


# ---------------- Textes (FR + ENG skeleton) ----------------
def T(lang: str):
    if lang == "fr":
        return dict(
            title="Introduction à la Surface de Volatilité Implicite (SVI / IVS)",
            intro=(
                "Bienvenue ! Ce module pose des bases **simples et utiles** : pourquoi les marchés existent, "
                "ce qu’est **une option (S, K, T)**, comment penser **risque & volatilité**, et comment lire une "
                "**surface de volatilité implicite (IVS)** — tout en restant vigilant face aux **arnaques courantes**. "
                "Le but n’est pas de faire des calculs : on veut **comprendre les idées** et relier ces notions à la réalité."
            ),

            # 1) Finance de marché
            s1_title="1) Pourquoi s'intéresser à la finance de marché ? (Le B.A.-BA)",
            s1_paragraph=(
                "La finance de marché est un **moteur de l’économie**. Elle permet :\n\n"
                "- **Financer l’économie réelle** : les entreprises lèvent des fonds via des **actions** (parts de propriété) "
                "et des **obligations** (emprunts). Les États financent routes, hôpitaux, transition énergétique, etc.\n"
                "- **Gérer les risques** : compagnies aériennes, agriculteurs, exportateurs… se couvrent contre les variations "
                "défavorables (prix de l’énergie, taux de change, etc.).\n"
                "- **Donner un prix à l’incertitude** : les prix reflètent l’agrégation d’**informations, d’anticipations** et "
                "de **préférences de risque** de millions d’acteurs."
            ),
            s1_quiz_q="Vrai ou faux ? Les marchés ne servent qu’aux spéculateurs.",
            s1_quiz_choices=["Vrai", "Faux"],
            s1_quiz_feedback_ok="✅ Faux : ils financent l’économie, aident à gérer des risques et produisent des signaux d’information.",

            # 2) Options
            s2_title="2) Comprendre les options : un outil puissant (et complexe !)",
            s2_paragraph=(
                "Une **option** est un **contrat** qui donne à l’acheteur un **droit (pas une obligation)** sur un actif **sous-jacent S**.\n\n"
                "**Paramètres clés** :\n"
                "- **S** : prix du sous-jacent (action, indice, pétrole…)\n"
                "- **K** : **prix d’exercice** (*strike*)\n"
                "- **T** : **échéance** (date limite)\n\n"
                "**Types** :\n"
                "- **Call** : droit **d’acheter** S à K\n"
                "- **Put**  : droit **de vendre** S à K\n\n"
                "**Style d’exercice** :\n"
                "- **Européenne** : exerçable **à l’échéance**\n"
                "- **Américaine** : exerçable **à tout moment** jusqu’à l’échéance\n\n"
                "**À quoi ça sert ?** Effet de **levier** (attention aux pertes), **couverture** (*hedging*), et stratégies flexibles "
                "selon les scénarios de marché. Le prix d’une option **révèle aussi** ce que le marché **anticipe** sur l’incertitude future."
            ),
            s2_quiz_q="Une option vous oblige-t-elle à acheter/vendre ?",
            s2_quiz_choices=["Oui", "Non"],
            s2_quiz_feedback_ok="✅ Exact : une option est un **droit**, pas une obligation.",
            s2_tip="🤯 L’effet de levier amplifie **gains** et **pertes** — à manier avec précaution.",

            # 3) Risque, perception, volatilité
            s3_title="3) Risque, perception… et volatilité",
            s3_paragraph=(
                "Le **risque** est la possibilité d’un résultat défavorable. La **perception** varie : beaucoup craignent l’avion alors que "
                "le **risque routier** est statistiquement plus élevé. En finance :\n\n"
                "- **Averse au risque** : prêt à **payer** pour réduire l’incertitude\n"
                "- **Neutre au risque** : raisonne au **rendement attendu**\n"
                "- **Preneur de risque** : accepte plus d’incertitude pour un gain potentiel plus élevé\n\n"
                "La **volatilité** quantifie l’**amplitude** des variations de prix (incertitude). Plus les rendements s’écartent de leur "
                "moyenne, plus la volatilité est élevée (idée d’**écart-type**)."
            ),
            s3_formula=r"\\sigma = \\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}(r_i - \\mu)^2}",
            s3_chart_title="Comparer deux environnements : calme vs agité",
            s3_quiz_q="La série la plus risquée est celle…",
            s3_quiz_choices=["…où le prix bouge peu", "…où le prix varie fortement"],
            s3_quiz_feedback_ok="✅ Correct : plus la dispersion est forte, plus la volatilité (et donc le risque perçu) est élevée.",

            # 4) IVS
            s4_title="4) L’IVS : une carte 3D des anticipations du marché",
            s4_paragraph=(
                "La **volatilité implicite (IV)** est **déduite des prix d’options** : c’est une mesure **tournée vers l’avenir**. "
                "La **surface de volatilité implicite (IVS)** cartographie l’IV selon deux axes : **moneyness K/S** et **maturité T**.\n\n"
                "Formes à repérer :\n"
                "- **Smile 😀** : IV plus élevée pour les options très ITM/OTM → marché attentif aux **mouvements extrêmes**.\n"
                "- **Smirk 🤨** (skew baissier) : IV plus élevée pour les **puts OTM** → **peur des baisses** (prime de protection)."
            ),
            s4_chart_title="Surface IVS (démo synthétique)",
            s4_quiz_q="Une IVS représente surtout…",
            s4_quiz_choices=["Des profits passés", "Les anticipations (incertitude) du marché"],
            s4_quiz_feedback_ok="✅ Exact : l’IVS reflète **les anticipations** et préférences de risque agrégées.",

            # 5) Prévention arnaques
            s5_title="5) Prévention : attention aux arnaques courantes",
            s5_paragraph=(
                "Restez **vigilant** face aux promesses trop belles pour être vraies :\n"
                "- **Copy-trading / signaux miracles** : gains “garantis” en copiant un “expert”\n"
                "- **Options binaires** : produits **très risqués**, souvent opaques et non régulés\n"
                "- **Promos trompeuses** : “+300% par semaine”, “sans risque” → **drapeaux rouges**\n\n"
                "👉 Vérifiez toujours la **régulation** et la **réputation** d’une plateforme."
            ),
            s5_quiz_q="On vous promet +10%/jour “sans risque” en copiant un pro. Crédible ?",
            s5_quiz_choices=["Oui", "Non"],
            s5_quiz_feedback_ok="✅ Non : rendement et risque vont **toujours** ensemble. C’est presque sûrement une arnaque.",

            # Recap
            recap_title="📝 Récapitulatif",
            recap_points=(
                "- Options = **droits** (Call/Put) sur **S, K, T**\n"
                "- Le risque est **perçu** différemment → impacte les prix\n"
                "- Volatilité = **dispersion** des rendements\n"
                "- **IVS** = **carte** des anticipations et peurs du marché\n"
                "- Attention aux **arnaques** : promesses de gains faciles\n"
            ),
            mark_read="Marquer comme lu et passer à l’application",
        )
    # EN minimal scaffold (on complètera ensuite si besoin)
    return dict(
        title="Introduction to Implied Volatility Surface (IVS)",
        intro="This module explains markets, options (S, K, T), risk & volatility, and how to read an IVS — plus scam prevention.",
        s1_title="1) Why markets?",
        s1_paragraph="- Finance the real economy\n- Manage risks\n- Price uncertainty via expectations",
        s1_quiz_q="True/False: Markets are only for speculators.",
        s1_quiz_choices=["True","False"],
        s1_quiz_feedback_ok="✅ False.",
        s2_title="2) Options 101",
        s2_paragraph="Option = right (not obligation). S, K, T. Call/Put. European/American. Uses: leverage, hedging, strategies.",
        s2_quiz_q="Does an option force you to buy/sell?",
        s2_quiz_choices=["Yes","No"],
        s2_quiz_feedback_ok="✅ Correct: a right, not an obligation.",
        s2_tip="🤯 Leverage amplifies gains and losses.",
        s3_title="3) Risk, perception & volatility",
        s3_paragraph="Risk attitudes differ; volatility = dispersion of returns.",
        s3_formula=r"\\sigma = \\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}(r_i - \\mu)^2}",
        s3_chart_title="Calm vs choppy environments",
        s3_quiz_q="Which is riskier?",
        s3_quiz_choices=["Small moves","Large moves"],
        s3_quiz_feedback_ok="✅ Correct.",
        s4_title="4) IVS: a 3D map of expectations",
        s4_paragraph="IV is inferred from option prices. IVS vs K/S and T. Smile and smirk.",
        s4_chart_title="Synthetic IVS surface",
        s4_quiz_q="An IVS mostly shows…",
        s4_quiz_choices=["Past profits","Market expectations"],
        s4_quiz_feedback_ok="✅ Market expectations.",
        s5_title="5) Scam prevention",
        s5_paragraph="Beware guaranteed returns, miracle signals, binary options. Check regulation.",
        s5_quiz_q="10%/day risk-free by copying a pro. Credible?",
        s5_quiz_choices=["Yes","No"],
        s5_quiz_feedback_ok="✅ No.",
        recap_title="📝 Recap",
        recap_points="- Options rights; Risk perception; Volatility; IVS; Scams.",
        mark_read="Mark as read and proceed to the app",
    )

TXT = T(LANG)

# ---------------- Helpers ----------------
def quiz_tf(question: str, choices: list[str], correct: str, feedback_ok: str, *, key: str):
    """
    Quiz True/False ou Vrai/Faux multi-langue sans présélection.
    Affiche un bouton de validation et du feedback seulement après clic.
    """
    # Aucune sélection initiale (Streamlit >= 1.27)
    rep = st.radio(question, choices, index=None, key=f"{key}_radio")
    validate_label = "Valider" if LANG == "fr" else "Submit"
    validate = st.button(validate_label, key=f"{key}_submit")

    if not validate:
        return None

    if rep is None:
        st.warning("Sélectionne une réponse." if LANG=="fr" else "Select an answer.")
        return None

    ok = (rep == correct)
    if ok:
        st.success(feedback_ok)
    else:
        st.error("❌ " + ("Mauvaise réponse." if LANG=="fr" else "Wrong answer."))
    log_event("quiz_feedback", PAGE_NAME, {"q": key, "correct": ok})
    return ok

# ---------------- UI ----------------
st.title(TXT["title"])
st.write(TXT["intro"])
st.markdown("---")

# ---- 1) Finance de marché
st.header(TXT["s1_title"])
st.markdown(TXT["s1_paragraph"])
quiz_tf(
    TXT["s1_quiz_q"],
    TXT["s1_quiz_choices"],
    correct=TXT["s1_quiz_choices"][-1],  # "Faux" / "False"
    feedback_ok=TXT["s1_quiz_feedback_ok"],
    key="s1"
)
st.markdown("---")

# ---- 2) Options + payoff interactif
st.header(TXT["s2_title"])
st.markdown(TXT["s2_paragraph"])

c1, c2, c3 = st.columns([1,1,2])
with c1:
    opt_type = st.selectbox("Type d’option" if LANG=="fr" else "Option type",
                            ["Call","Put"], key="opt_type")
with c2:
    K = st.slider("Strike K", 50.0, 150.0, 100.0, 1.0, key="strike")
with c3:
    Smax = st.slider("Plage de prix S max" if LANG=="fr" else "Max spot range",
                     80.0, 200.0, 160.0, 5.0, key="smax")

S_vals = np.linspace(0, Smax, 400)
payoff = np.maximum(S_vals - K, 0.0) if opt_type=="Call" else np.maximum(K - S_vals, 0.0)

fig_payoff = go.Figure()
fig_payoff.add_trace(go.Scatter(x=S_vals, y=payoff, name=(f"Payoff {opt_type} à l’échéance" if LANG=="fr" else f"{opt_type} payoff at expiry")))
fig_payoff.add_vline(x=K, line_dash="dot", annotation_text=f"K={K:.0f}")
fig_payoff.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=320,
                         xaxis_title=("Prix du sous-jacent S à l’échéance" if LANG=="fr" else "Underlying price S at expiry"),
                         yaxis_title=("Payoff" if LANG=="fr" else "Payoff"))
st.plotly_chart(fig_payoff, use_container_width=True)
log_event("chart_interaction", PAGE_NAME, {"chart":"payoff", "type":opt_type, "K":K, "Smax":Smax})

quiz_tf(
    TXT["s2_quiz_q"],
    TXT["s2_quiz_choices"],
    correct=("Non" if LANG=="fr" else "No"),
    feedback_ok=TXT["s2_quiz_feedback_ok"],
    key="s2"
)

st.info(TXT["s2_tip"])
st.markdown("---")

# ---- 3) Risque, perception & volatilité + graph
st.header(TXT["s3_title"])
st.markdown(TXT["s3_paragraph"])
st.latex(TXT["s3_formula"])

c1, c2 = st.columns(2)
with c1:
    vol_low = st.slider("Volatilité 'calme'" if LANG=="fr" else "Calm volatility",
                        0.05, 0.40, 0.12, 0.01, key="vol_low_slider")
with c2:
    vol_high = st.slider("Volatilité 'agitée'" if LANG=="fr" else "Choppy volatility",
                         0.10, 1.00, 0.45, 0.01, key="vol_high_slider")

N = 200
np.random.seed(7)
r_low  = np.random.normal(0, vol_low/np.sqrt(252), N)
r_high = np.random.normal(0, vol_high/np.sqrt(252), N)
S0 = 100
path_low  = S0 * np.exp(np.cumsum(r_low))
path_high = S0 * np.exp(np.cumsum(r_high))

fig_prices = go.Figure()
fig_prices.add_trace(go.Scatter(y=path_low,  name="Faible vol" if LANG=="fr" else "Low vol"))
fig_prices.add_trace(go.Scatter(y=path_high, name="Forte vol"  if LANG=="fr" else "High vol"))
fig_prices.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=320,
                         xaxis_title=("Jours" if LANG=="fr" else "Days"),
                         yaxis_title=("Prix simulé" if LANG=="fr" else "Simulated price"))
st.subheader(TXT["s3_chart_title"])
st.plotly_chart(fig_prices, use_container_width=True)
log_event("chart_interaction", PAGE_NAME, {"chart":"vol_paths", "vol_low":vol_low, "vol_high":vol_high})

quiz_tf(
    TXT["s3_quiz_q"],
    TXT["s3_quiz_choices"],
    correct=TXT["s3_quiz_choices"][-1],
    feedback_ok=TXT["s3_quiz_feedback_ok"],
    key="s3"
)
st.markdown("---")

# ---- 4) IVS + surface
st.header(TXT["s4_title"])
st.markdown(TXT["s4_paragraph"])

colA, colB, colC = st.columns(3)
with colA:
    shape = st.selectbox("Forme" if LANG=="fr" else "Shape", ["Smile 😀","Smirk 🤨"], key="ivs_shape_box")
with colB:
    base = st.slider("Niveau de base (IV)" if LANG=="fr" else "Base level (IV)", 0.05, 0.80, 0.25, 0.01, key="ivs_base_slider")
with colC:
    skew = st.slider("Skew (asymétrie)" if LANG=="fr" else "Skew (asymmetry)", -0.80, 0.80, -0.35, 0.01, key="ivs_skew_slider")

k = np.linspace(0.6, 1.4, 40)   # moneyness K/S
t = np.linspace(0.05, 1.0, 30)  # maturités (années)
Kgrid, Tgrid = np.meshgrid(k, t)

if "Smile" in shape:
    IV = base + 0.6*(Kgrid-1.0)**2 * (0.6*np.sqrt(Tgrid)+0.4)
else:
    IV = base + 0.25*(Kgrid-1.0)**2 + skew*(Kgrid-1.0) * (0.5+0.5*np.sqrt(Tgrid))
IV = np.clip(IV, 0.01, 2.0)

fig_ivs = go.Figure(data=[go.Surface(x=Kgrid, y=Tgrid, z=IV, colorscale="Viridis", showscale=True)])
fig_ivs.update_layout(
    margin=dict(l=10, r=10, t=30, b=10),
    height=420,
    scene=dict(
        xaxis_title="Moneyness K/S",
        yaxis_title=("Maturité T (années)" if LANG=="fr" else "Maturity T (years)"),
        zaxis_title=("Volatilité implicite" if LANG=="fr" else "Implied Volatility")
    )
)
st.subheader(TXT["s4_chart_title"])
st.plotly_chart(fig_ivs, use_container_width=True)
log_event("chart_interaction", PAGE_NAME, {"chart":"ivs_surface", "shape":shape, "base":base, "skew":skew})

quiz_tf(
    TXT["s4_quiz_q"],
    TXT["s4_quiz_choices"],
    correct=TXT["s4_quiz_choices"][-1],
    feedback_ok=TXT["s4_quiz_feedback_ok"],
    key="s4"
)
st.markdown("---")

# ---- 5) Prévention arnaques
st.header(TXT["s5_title"])
st.markdown(TXT["s5_paragraph"])
quiz_tf(
    TXT["s5_quiz_q"],
    TXT["s5_quiz_choices"],
    correct=("Non" if LANG=="fr" else "No"),
    feedback_ok=TXT["s5_quiz_feedback_ok"],
    key="s5"
)
st.markdown("---")

# ---- Récap + Mark as read
st.header(TXT["recap_title"])
st.markdown(TXT["recap_points"])

st.markdown("---")
c1, c2 = st.columns(2)
# 1) Back to Main — finish missions
with c1:
    if st.button(("🏠 Retour à l’accueil" if LANG=="fr" else "🏠 Back to Main App"), key="beg_mod_nav_main"):
        secs = stop_timer_seconds("page_timer", "page_leave", {"to": "Main_App", "from": PAGE_NAME})
        # Cross-level for Beginner content: if user is Advanced, this emits adv_on_beg; if Beginner, nothing
        log_crosslevel(content_level="Beginner", seconds=secs, where=PAGE_NAME)

        log_event("beg_module_nav_main", PAGE_NAME, {
            "level": LEVEL, "lang": LANG, "topic": "Introduction", "seconds": secs
        })
        st.switch_page("pages/Main_App.py")

# 2) Go to Advanced module (only if eligible: Beginner final quiz score >= 4/5)
with c2:
    if st.session_state.get("beg_promo_eligible", False):
        if st.button(("🚀 Passer au module avancé" if LANG=="fr" else "🚀 Go to Advanced Module"),
                     key="beg_mod_nav_advanced"):
            secs = stop_timer_seconds("page_timer", "page_leave", {"to": "Implied_Volatility_Surfaces", "from": PAGE_NAME})
            log_crosslevel(content_level="Beginner", seconds=secs, where=PAGE_NAME)
            log_event("promo_to_advanced_click", PAGE_NAME, {
                "level": LEVEL, "lang": LANG,
                "score": int(st.session_state.get("quiz_final_score", 0)),
                "eligible": True, "seconds": secs
            })
            st.switch_page("pages/Implied Volatility Surfaces.py")
    else:
        st.button(("🚀 Passer au module avancé (score ≥ 4/5 requis)" if LANG=="fr" else "🚀 Go to Advanced (need ≥ 4/5)"),
                  key="beg_mod_nav_advanced_disabled", disabled=True)