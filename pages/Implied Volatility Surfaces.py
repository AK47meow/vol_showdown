# pages/Implied Volatility Surfaces.py
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from math import sqrt
from scipy.stats import norm

from utils import (
    init_page_session_vars, init_session_tracking,
    track_page_view, start_timer, stop_timer_seconds,
    log_event, mark_module_opened, get_firebase,
    is_advanced_level, event_with_dwell, log_crosslevel
)

# --- Early redirect handler (top of the page)
_nav = st.session_state.get("_nav_to")
if _nav:
    st.session_state["_nav_to"] = None
    st.switch_page(_nav)

st.set_page_config(page_title="Implied Volatility Surfaces — Advanced", page_icon="🧠", layout="wide")

# ---------------- Session / tracking ----------------
init_page_session_vars()
init_session_tracking()

LANG   = st.session_state.language_code
LEVEL  = st.session_state.expertise_level
PAGE_NAME = "Modules/Implied_Volatility_Surfaces"

# Mark module opened (do this ONCE)
mark_module_opened(
    module_id=("fr_adv_implied_vol_surface" if LANG == "fr" else "en_adv_implied_vol_surface"),
    lang=LANG,
    level=LEVEL,
    topic="Implied Volatility Surfaces",
)

# Track + start page timer
track_page_view(PAGE_NAME)
start_timer("page_timer", {"page_name": PAGE_NAME})

# ---- Cross-level: si un débutant ouvre le module avancé, compter une visite (1x par session)
if (not is_advanced_level(LEVEL)) and (not st.session_state.get("_adv_beg_counted", False)):
    log_crosslevel(content_level="Advanced", seconds=0.0, where=PAGE_NAME)  # ⬅️ compte +0s
    st.session_state["_adv_beg_counted"] = True
    log_event("crosslevel_beginner_open_advanced", PAGE_NAME, {"level": LEVEL, "lang": LANG})

# ---------------- Textes FR/EN ----------------
def T(lang: str):
    if lang == "fr":
        return dict(
            title="Surfaces de Volatilité Implicite — Avancé",
            recap="On s’appuie sur le module débutant : volatilité, BSM et IVS sont réutilisés.",
            banner_info="👀 Ce module est recommandé pour le niveau **Avancé**. Vous pouvez explorer librement sans interrompre votre session.",
            # Bloc BSM
            bsm_block_title="1) Volatilité & Modèle de Black–Scholes–Merton (rappel)",
            vol_line="- **Volatilité** : quantité instantanée non observable directement, distincte de la volatilité historique.",
            bsm_line="- **BSM** : sous hypothèses (log-normalité, pas d’arbitrage, couverture continue, vol constante), formule fermée pour options européennes.",
            s_label="S (spot)",
            k_label="K (strike)",
            t_label="T (années)",
            sigma_label="σ (vol)",
            r_label="r (taux sans risque)",
            q_label="q (dividende / taux forward)",
            type_label="Type d’option",
            price_metric="Prix BSM",
            greeks_title="Greeks",
            # Inverse/Calibration
            inverse_title="2) Problème inverse & Calibration (Newton–Raphson)",
            inverse_line=("On **connaît le prix de marché**, on **cherche σ implicite** tel que "
                          "BSM(S,K,r,q,σ,T)=Prix_mkt. La méthode **Newton–Raphson** itère "
                          "σ ← σ − (Price(σ)−P_mkt)/Vega(σ)."),
            mkt_price="Prix de marché",
            sigma_guess="σ initial (guess)",
            btn_compute_iv="Calculer la VI (Newton)",
            iv_success="Volatilité implicite : ",
            pro_tip_title="💡 Pro tip — bon ‘guess’ initial",
            pro_tip_body=("Initialisez σ au **point d’inflexion** de f(σ)=BSM(σ)−Prix_mkt sur un intervalle raisonnable "
                          "(ex. [1%, 200%]) puis passez ce σ_guess à Newton–Raphson pour accélérer la convergence. "
                          "Si `logic.py` implémente déjà cette stratégie, elle sera utilisée automatiquement."),
            # Risk / Greeks / Hedging
            risk_title="3) Profils de risque & Greeks — Delta-hedging",
            risk_body=("- **Profils** : aversion (assurance chère), neutralité (centré sur l’espérance), recherche de risque (convexité).  \n"
                       "- **Greeks** :  \n"
                       "  - **Delta** : sensibilité au spot (couverture directionnelle)  \n"
                       "  - **Gamma** : convexité (coût de rebalancement)  \n"
                       "  - **Vega** : sensibilité à σ (exposition aux régimes de volatilité)  \n"
                       "  - **Theta** : portage temporel (carry)  \n"
                       "  - **Rho** : sensibilité aux taux  \n"
                       "**Delta-hedging discret** : neutraliser le delta avec une position en S, rééquilibrer ; le P&L dépend de la convexité (gamma) et du chemin."),
            sim_caption="Simulation : marche géométrique, hedge quotidien ; P&L = option + hedge (sans frais).",
            days_label="Jours",
            true_sigma_label="σ ‘vraie’ (trajet)",
            seed_label="Seed",
            pnl_option="P&L Option",
            pnl_hedge="Flux Hedge",
            pnl_total="P&L Total",
            # IV Surface
            ivs_title="4) IV Surface — skew & structure par terme",
            base_vol="Vol de base",
            smile_int="Intensité Smile",
            skew_slope="Pente Skew",
            term_slope="Pente Terme",
            scene_x="K/S",
            scene_y="T (années)",
            scene_z="VI",
            cta_main="🏠 Retour à l'accueil",
            cta_quiz="🧠 Passer l'évaluation finale",
        )
    return dict(
        title="Implied Volatility Surfaces — Advanced",
        recap="We build on the beginner module: volatility, BSM, and IVS notions are reused.",
        banner_info="👀 This module is recommended for **Advanced** level. You can explore freely without interrupting your session.",
        # BSM block
        bsm_block_title="1) Volatility & Black–Scholes–Merton (recap)",
        vol_line="- **Volatility**: instantaneous, unobservable quantity distinct from historical vol.",
        bsm_line="- **BSM**: under assumptions (log-normality, no-arb, continuous hedge, constant vol), closed-form European option pricing.",
        s_label="S (spot)",
        k_label="K (strike)",
        t_label="T (years)",
        sigma_label="σ (vol)",
        r_label="r (risk-free)",
        q_label="q (dividend / forward yield)",
        type_label="Option type",
        price_metric="BSM price",
        greeks_title="Greeks",
        # Inverse/Calibration
        inverse_title="2) Inverse Problem & Calibration (Newton–Raphson)",
        inverse_line=("Given a **market option price**, find **σ_imp** such that "
                      "BSM(S,K,r,q,σ,T)=Price_mkt. **Newton–Raphson** updates "
                      "σ ← σ − (Price(σ)−P_mkt)/Vega(σ)."),
        mkt_price="Market price",
        sigma_guess="σ guess",
        btn_compute_iv="Compute IV (Newton)",
        iv_success="Implied vol: ",
        pro_tip_title="💡 Pro tip — good initial ‘guess’",
        pro_tip_body=("Initialize σ at the **inflection point** of f(σ)=BSM(σ)−Price_mkt over a reasonable bracket "
                      "(e.g. [1%,200%]), then pass σ_guess to Newton–Raphson to improve convergence. "
                      "If your `logic.py` already implements this, it will be used automatically."),
        # Risk / Greeks / Hedging
        risk_title="3) Risk profiles & Greeks — Delta hedging",
        risk_body=("- **Profiles**: risk-averse (insurance pricey), neutral (mean-based), risk-seeking (convexity).  \n"
                   "- **Greeks**:  \n"
                   "  - **Delta**: sensitivity to spot (directional hedge)  \n"
                   "  - **Gamma**: convexity (rebalancing cost)  \n"
                   "  - **Vega**: sensitivity to σ (vol regimes)  \n"
                   "  - **Theta**: time carry  \n"
                   "  - **Rho**: sensitivity to rates  \n"
                   "**Discrete delta-hedging**: neutralize delta with a position in S, rebalance; P&L depends on convexity (gamma) and the path."),
        sim_caption="Simulation: geometric walk, daily hedge; P&L = option + hedge (no fees).",
        days_label="Days",
        true_sigma_label="True σ (path)",
        seed_label="Seed",
        pnl_option="Option P&L",
        pnl_hedge="Hedge Cashflow",
        pnl_total="Total P&L",
        # IV Surface
        ivs_title="4) IV Surface — skew & term structure",
        base_vol="Base vol",
        smile_int="Smile intensity",
        skew_slope="Skew slope",
        term_slope="Term slope",
        scene_x="K/S",
        scene_y="T (y)",
        scene_z="IV",
        cta_main="🏠 Back to Main App",
        cta_quiz="🧠 Take Final Quiz",
    )

TXT = T(LANG)

# ---- KPI: beginner exploring advanced (log only once per session)
if (not is_advanced_level(LEVEL)) and not st.session_state.get("_adv_explore_logged", False):
    log_event("advanced_explored_by_beginner", PAGE_NAME, {"level": LEVEL, "lang": LANG})
    st.session_state["_adv_explore_logged"] = True

# ------------------------------------------------------------
# 0) Newton–Raphson import (logic.py) + fallback
# ------------------------------------------------------------
IMPL_FUNC = None
try:
    # expected signature: implied_vol_newton(price, S, K, r, q, T, sigma_guess=None, max_iter=100, tol=1e-8)
    from logic import implied_vol_newton as IMPL_FUNC
except Exception:
    IMPL_FUNC = None

def bs_price(S, K, r, q, sigma, T, option_type="call"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S-K, 0.0) if option_type=="call" else max(K-S, 0.0)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == "call":
        return float(S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2))
    else:
        return float(K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1))

def bs_greeks(S, K, r, q, sigma, T, option_type="call"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return dict(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    pdf_d1 = 1/np.sqrt(2*np.pi)*np.exp(-0.5*d1*d1)
    if option_type == "call":
        delta = np.exp(-q*T)*norm.cdf(d1)
    else:
        delta = -np.exp(-q*T)*norm.cdf(-d1)
    gamma = np.exp(-q*T) * pdf_d1 / (S*sigma*np.sqrt(T))
    vega  = S*np.exp(-q*T) * pdf_d1 * np.sqrt(T)
    if option_type == "call":
        theta = (-S*np.exp(-q*T)*pdf_d1*sigma/(2*np.sqrt(T))
                 + q*S*np.exp(-q*T)*norm.cdf(d1)
                 - r*K*np.exp(-r*T)*norm.cdf(d2))
        rho   = K*T*np.exp(-r*T)*norm.cdf(d2)
    else:
        theta = (-S*np.exp(-q*T)*pdf_d1*sigma/(2*np.sqrt(T))
                 - q*S*np.exp(-q*T)*norm.cdf(-d1)
                 + r*K*np.exp(-r*T)*norm.cdf(-d2))
        rho   = -K*T*np.exp(-r*T)*norm.cdf(-d2)
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

def implied_vol_fallback(price_mkt, S, K, r, q, T, sigma_guess=None, max_iter=100, tol=1e-8, option_type="call"):
    if sigma_guess is None:
        sigma_guess = 0.2
    sigma = float(sigma_guess)
    for _ in range(max_iter):
        price = bs_price(S, K, r, q, sigma, T, option_type)
        vega  = bs_greeks(S, K, r, q, sigma, T, option_type)["vega"]
        diff  = price - price_mkt
        if abs(diff) < tol:
            return max(1e-4, sigma)
        if abs(vega) < 1e-8:
            break
        sigma -= diff/vega
        if sigma <= 0:
            sigma = 1e-4
    return max(1e-4, sigma)

def implied_vol(price_mkt, S, K, r, q, T, sigma_guess=None, option_type="call"):
    if IMPL_FUNC is not None:
        try:
            return float(IMPL_FUNC(price_mkt, S, K, r, q, T, sigma_guess=sigma_guess))
        except Exception:
            pass
    return implied_vol_fallback(price_mkt, S, K, r, q, T, sigma_guess=sigma_guess, option_type=option_type)

# ---------------- UI ----------------
st.title(TXT["title"])
st.caption(TXT["recap"])

# 1) Vol & BSM
with st.expander(TXT["bsm_block_title"], expanded=True):
    st.markdown(TXT["vol_line"])
    st.markdown(TXT["bsm_line"])

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        S = st.number_input(TXT["s_label"], 1.0, 10000.0, 100.0, 1.0, key="adv_S")
    with c2:
        K = st.number_input(TXT["k_label"], 1.0, 10000.0, 100.0, 1.0, key="adv_K")
    with c3:
        T = st.slider(TXT["t_label"], 0.01, 3.0, 0.5, 0.01, key="adv_T")
    with c4:
        sigma = st.slider(TXT["sigma_label"], 0.01, 1.0, 0.25, 0.01, key="adv_sigma")
    with c5:
        r = st.slider(TXT["r_label"], 0.0, 0.10, 0.02, 0.005, key="adv_r")
    q = st.slider(TXT["q_label"], 0.0, 0.10, 0.00, 0.005, key="adv_q")
    otype = st.selectbox(TXT["type_label"], ["call","put"], index=0, key="adv_type")

    price = bs_price(S, K, r, q, sigma, T, option_type=otype)
    greeks = bs_greeks(S, K, r, q, sigma, T, option_type=otype)

    cP, cG = st.columns(2)
    with cP:
        st.metric(TXT["price_metric"], f"{price:,.4f}")
    with cG:
        st.write(TXT["greeks_title"])
        st.json({k: round(float(v), 6) for k, v in greeks.items()})

    log_event("adv_bsm_params", PAGE_NAME, {"S":S,"K":K,"T":T,"sigma":sigma,"r":r,"q":q,"type":otype})

# 2) Inverse & Calibration (Newton–Raphson)
st.subheader(TXT["inverse_title"])
st.markdown(TXT["inverse_line"])

c1, c2, c3 = st.columns(3)
with c1:
    price_mkt = st.number_input(TXT["mkt_price"], 0.0001, 1e6, max(0.01, price), 0.01, key="adv_mkt")
with c2:
    sigma_guess = st.slider(TXT["sigma_guess"], 0.01, 1.0, 0.25, 0.01, key="adv_guess")
with c3:
    btn_iv = st.button(TXT["btn_compute_iv"])

if btn_iv:
    iv = implied_vol(price_mkt, S, K, r, q, T, sigma_guess=sigma_guess, option_type=otype)
    st.success(f"{TXT['iv_success']}{iv:.6f}")
    log_event("adv_iv_solved", PAGE_NAME, {"iv": float(iv), "guess": float(sigma_guess)})

# Pro tip stays multilingual (text from TXT)
with st.expander(TXT["pro_tip_title"], expanded=False):
    st.markdown(TXT["pro_tip_body"])

# NEW: Show the actual Newton–Raphson implementation as a readable snippet (not executed)
with st.expander("📜 Newton–Raphson implementation (Python snippet)"):
    code_snippet = """def implied_vol_newton(C, S, K, r, T, tol, option_type):
    def inflexion_point(S, K, T, r):
        m = S / (K * np.exp(-r * T))
        return np.sqrt(2 * np.abs(np.log(m)) / T)

    guess = inflexion_point(S, K, T, r)
    max_iter = 300
    for i in range(max_iter):
        p = calculate_black_scholes(S, K, T, r, guess, option_type)
        v = calculate_vega(S, K, T, r, guess, option_type)
        if abs(v) < 1e-4:
            return None

        error = (p - C) / v
        guess = guess - error

        # if using numpy scalars
        if hasattr(error, "item"):
            if abs(error.item()) < tol:
                return guess
        else:
            if abs(error) < tol:
                return guess

    return None
"""
    st.code(code_snippet, language="python")
    st.caption("This is your original Newton–Raphson routine (shown for learning only). It is not executed here.")


# 3) Risk profiles & Greeks — Delta hedging
st.subheader(TXT["risk_title"])
st.markdown(TXT["risk_body"])
st.caption(TXT["sim_caption"])

h_col1, h_col2, h_col3 = st.columns(3)
with h_col1:
    days = st.slider(TXT["days_label"], 5, 252, 30, 1, key="adv_days")
with h_col2:
    true_sigma = st.slider(TXT["true_sigma_label"], 0.01, 1.0, 0.30, 0.01, key="adv_true_sigma")
with h_col3:
    seed = st.number_input(TXT["seed_label"], 0, 10_000, 7, 1, key="adv_seed")

np.random.seed(seed)
dt = 1/252
S_path = [S]
for t in range(days):
    dW = np.random.normal(0, sqrt(dt))
    S_next = S_path[-1] * np.exp((r - q - 0.5*true_sigma**2)*dt + true_sigma*dW)
    S_path.append(S_next)

# Hedging (discret)
shares = 0.0
cash = 0.0
opt_prices = []

opt0 = bs_price(S, K, r, q, sigma, T, option_type=otype)
opt_prices.append(opt0)
delta0 = bs_greeks(S, K, r, q, sigma, T, option_type=otype)["delta"]
shares = delta0
cash = -(shares * S)

for t in range(1, len(S_path)):
    remaining_T = max(1e-6, T - t*dt)
    St_ = S_path[t]
    opt_t = bs_price(St_, K, r, q, sigma, remaining_T, option_type=otype)
    opt_prices.append(opt_t)
    delta_t = bs_greeks(St_, K, r, q, sigma, remaining_T, option_type=otype)["delta"]
    d_shares = delta_t - shares
    cash -= d_shares * St_
    shares = delta_t

cash += shares * S_path[-1]
pnl_option = opt_prices[-1] - opt_prices[0]
pnl_total  = pnl_option + cash  # sans frais

dcol1, dcol2, dcol3 = st.columns(3)
with dcol1: st.metric(TXT["pnl_option"], f"{pnl_option:,.4f}")
with dcol2: st.metric(TXT["pnl_hedge"], f"{cash:,.4f}")
with dcol3: st.metric(TXT["pnl_total"], f"{pnl_total:,.4f}")

fig_path = go.Figure()
fig_path.add_trace(go.Scatter(y=S_path, name="S path"))
fig_path.update_layout(height=280, margin=dict(l=10,r=10,t=30,b=10), xaxis_title="Step", yaxis_title="S")
st.plotly_chart(fig_path, use_container_width=True)
log_event("adv_delta_hedge_run", PAGE_NAME, {"days":days,"true_sigma":true_sigma,"sigma_model":sigma})

# 4) IV Surface avancée
st.subheader(TXT["ivs_title"])
civ1, civ2, civ3, civ4 = st.columns(4)
with civ1:
    base_vol = st.slider(TXT["base_vol"], 0.01, 1.0, 0.25, 0.01, key="adv_base")
with civ2:
    smile_int = st.slider(TXT["smile_int"], 0.0, 1.0, 0.20, 0.01, key="adv_smile")
with civ3:
    skew_slope = st.slider(TXT["skew_slope"], -1.0, 1.0, -0.30, 0.01, key="adv_skew")
with civ4:
    term_slope = st.slider(TXT["term_slope"], -0.5, 0.5, 0.08, 0.01, key="adv_term")

k = np.linspace(0.6, 1.4, 41)
tgrid = np.linspace(0.05, 2.0, 30)
Kgrid, Tgrid = np.meshgrid(k, tgrid)

def iv_parametric(m, T, base, smile, skew, term):
    return np.clip(base + smile*(m-1.0)**2 + skew*(m-1.0) + term*np.log(1+T), 0.01, 2.0)

IV = iv_parametric(Kgrid, Tgrid, base_vol, smile_int, skew_slope, term_slope)

fig_ivs = go.Figure(data=[go.Surface(x=Kgrid, y=Tgrid, z=IV, colorscale="Viridis", showscale=True)])
fig_ivs.update_layout(height=460, margin=dict(l=10,r=10,t=30,b=10),
                      scene=dict(xaxis_title=TXT["scene_x"], yaxis_title=TXT["scene_y"], zaxis_title=TXT["scene_z"]))
st.plotly_chart(fig_ivs, use_container_width=True)
log_event("adv_ivs_surface", PAGE_NAME, {"base":base_vol,"smile":smile_int,"skew":skew_slope,"term":term_slope})

# -------- Navigation  ----------
st.divider()
c1, c2 = st.columns(2)

with c1:
    if st.button(TXT["cta_main"]):
        # 1) stoppe le timer et récupère la durée exacte sur la page
        secs = stop_timer_seconds("page_timer", "page_leave", {"to": "Main_App", "from": PAGE_NAME})
        # 2) cross-level (contenu Advanced) — si user=Beginner, ça émet 'beg_on_adv', sinon rien
        log_crosslevel(content_level="Advanced", seconds=secs, where=PAGE_NAME)
        # 3) logs existants (inchangés)
        log_event("adv_end_nav_main", PAGE_NAME, {"level": LEVEL, "lang": LANG})
        st.switch_page("pages/Main_App.py")

with c2:
    if st.button(TXT["cta_quiz"]):
        secs = stop_timer_seconds("page_timer", "page_leave", {"to": "Final_Quizz_Advanced", "from": PAGE_NAME})
        log_crosslevel(content_level="Advanced", seconds=secs, where=PAGE_NAME)
        log_event("adv_end_nav_quiz", PAGE_NAME, {"level": LEVEL, "lang": LANG})
        st.switch_page("pages/Final_Quizz_Advanced.py")