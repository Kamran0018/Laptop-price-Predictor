import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Initialize session state for tracking predictions, resetting values, and scrolling
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "predicted_price" not in st.session_state:
    st.session_state.predicted_price = 0
if "last_config" not in st.session_state:
    st.session_state.last_config = {}
if "scroll_to_top" not in st.session_state:
    st.session_state.scroll_to_top = False

# Reset function to clear inputs back to None and trigger scroll
def reset_all():
    keys_to_reset = [
        "company", "laptop_type", "ram", "cpu", "ssd", "hdd", 
        "gpu", "screen_size", "touchscreen", "resolution", "ips", 
        "weight", "os"
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            st.session_state[key] = None
    st.session_state.predicted = False
    st.session_state.predicted_price = 0
    st.session_state.last_config = {}
    st.session_state.scroll_to_top = True

# ── JavaScript Scroll to Top Injection ──
if st.session_state.scroll_to_top:
    st.markdown("""
    <script>
        var mainPage = document.querySelector('section.main') || window.document.querySelector('.main');
        if (mainPage) {
            mainPage.scrollTo({top: 0, behavior: 'smooth'});
        } else {
            window.scrollTo({top: 0, behavior: 'smooth'});
        }
    </script>
    """, unsafe_allow_html=True)
    st.session_state.scroll_to_top = False

# ── Optimized Split-Loading for Blazing Fast Page Load ─────────────────────────
@st.cache_resource
def load_dataframe():
    """Loads the small dataframe representation (125 KB) instantly on startup."""
    with open("df.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Loads the large ML pipeline model (19.2 MB) lazily on demand and caches it."""
    with open("pipe.pkl", "rb") as f:
        return pickle.load(f)

df = load_dataframe()

# ── Full CSS (Optimized: Removed blocking Google Font imports) ──────────────────
st.markdown("""
<style>
/* ── Reset & Base ── */
html, body, [class*="css"], .stApp {
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    background: #f8fafc !important;
}

/* ── Animated gradient background ── */
.stApp {
    background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 30%, #fae8ff 60%, #f0fdf4 100%) !important;
    background-size: 400% 400% !important;
    animation: gradientShift 15s ease infinite !important;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Floating orbs ── */
.orb-container {
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}
.orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(80px);
    opacity: 0.15;
    animation: floatOrb 10s ease-in-out infinite;
}
.orb1 { width:400px; height:400px; background:#4f46e5; top:-100px; left:-100px; animation-delay:0s; }
.orb2 { width:300px; height:300px; background:#db2777; top:50%; right:-80px; animation-delay:3s; }
.orb3 { width:350px; height:350px; background:#0891b2; bottom:-80px; left:25%; animation-delay:6s; }
@keyframes floatOrb {
    0%, 100% { transform: translateY(0px) scale(1); }
    50%       { transform: translateY(-40px) scale(1.1); }
}

/* ── Hero ── */
.hero-wrap {
    position: relative;
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    margin-bottom: 0.5rem;
    animation: heroSlide 0.8s cubic-bezier(0.22,1,0.36,1) both;
}
@keyframes heroSlide {
    from { opacity:0; transform: translateY(-25px); }
    to   { opacity:1; transform: translateY(0); }
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(90deg, #4f46e5, #db2777);
    color: white;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 0.35rem 1.2rem;
    border-radius: 50px;
    margin-bottom: 0.8rem;
    box-shadow: 0 4px 12px rgba(79,70,229,0.2);
}
.hero-title {
    font-size: clamp(2.2rem, 6vw, 3.2rem);
    font-weight: 800;
    background: linear-gradient(135deg, #0f172a 0%, #4f46e5 50%, #db2777 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin: 0 0 0.5rem;
}
.hero-sub {
    color: #475569;
    font-size: 1.05rem;
    font-weight: 400;
    margin: 0;
}

/* ── Section cards ── */
.card {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.7);
    border-radius: 24px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 10px 30px rgba(79,70,229,0.04), 0 1px 8px rgba(0,0,0,0.02);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.card:hover {
    box-shadow: 0 20px 40px rgba(79,70,229,0.08), 0 2px 12px rgba(0,0,0,0.04);
    transform: translateY(-2px);
    border-color: rgba(99,102,241,0.25);
}

.card-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.2rem;
    padding-bottom: 0.6rem;
    border-bottom: 1.5px dashed rgba(99,102,241,0.15);
}
.card-icon {
    width: 32px; height: 32px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
}
.icon-brand  { background: #e0e7ff; }
.icon-hw     { background: #ffe4e6; }
.icon-disp   { background: #dcfce7; }
.icon-other  { background: #fef9c3; }
.card-title {
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #1e1b4b;
}

/* ── Streamlit Input overrides ── */
.stSelectbox > label,
.stNumberInput > label {
    color: #1e293b !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.3px;
}
.stSelectbox > div > div {
    background: #f8fafc !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 12px !important;
    color: #0f172a !important;
    font-weight: 500 !important;
}
.stSelectbox > div > div:hover {
    border-color: #2563eb !important;
}
.stNumberInput > div > div > input {
    background: #f8fafc !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 12px !important;
    color: #0f172a !important;
    font-weight: 500 !important;
}
.stNumberInput > div > div > input:focus {
    border-color: #2563eb !important;
}

/* Predict Button & Predict Another Button base styling */
.stButton > button {
    width: 100% !important;
    border-radius: 16px !important;
    height: 3.2rem !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-sizing: border-box !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin-top: 1rem !important;
    margin-bottom: 2rem !important;
    border: none !important;
}

/* Predict Price button specific styling (Solid Blue Color #2563eb) */
div[data-testid="stHorizontalBlock"] div[data-testid="column"]:first-child .stButton > button {
    background: #2563eb !important;
    color: white !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
}
div[data-testid="stHorizontalBlock"] div[data-testid="column"]:first-child .stButton > button:hover {
    background: #1d4ed8 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45) !important;
}

/* Predict Another button specific styling (Solid Filled Secondary Light Blue #dbeafe) */
div[data-testid="stHorizontalBlock"] div[data-testid="column"]:last-child .stButton > button {
    background: #dbeafe !important;
    color: #1d4ed8 !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.12) !important;
}
div[data-testid="stHorizontalBlock"] div[data-testid="column"]:last-child .stButton > button:hover {
    background: #bfdbfe !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.20) !important;
}

/* ── Price result card ── */
.result-wrap {
    animation: resultPop 0.5s cubic-bezier(0.34,1.56,0.64,1) both;
    margin-bottom: 1.5rem;
}
@keyframes resultPop {
    from { opacity:0; transform: scale(0.95) translateY(15px); }
    to   { opacity:1; transform: scale(1) translateY(0); }
}
.result-card {
    background: linear-gradient(135deg, #eff6ff 0%, #f5f3ff 50%, #fdf2f8 100%);
    border: 2px solid rgba(99, 102, 241, 0.15);
    border-radius: 28px;
    padding: 2.2rem 1.8rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 16px 40px rgba(99, 102, 241, 0.12);
}
.result-card::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: conic-gradient(from 0deg, transparent 0deg, rgba(99, 102, 241, 0.04) 80deg, transparent 150deg);
    animation: spin 10s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

.result-amount {
    font-size: clamp(2.4rem, 6vw, 3.8rem);
    font-weight: 800;
    background: linear-gradient(135deg, #3730a3, #4f46e5, #db2777);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0.2rem 0;
}
.spec-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    justify-content: center;
    margin-top: 1rem;
}
.chip {
    background: rgba(255,255,255,0.85);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 50px;
    padding: 0.25rem 0.75rem;
    font-size: 0.72rem;
    color: #4f46e5;
    font-weight: 600;
    box-shadow: 0 2px 6px rgba(0,0,0,0.02);
}

/* ── Hide Streamlit Elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; max-width: 760px !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 0.78rem;
    padding: 2rem 0 0.5rem;
}
.footer span { color: #4f46e5; font-weight: 600; }
</style>

<!-- Floating backgrounds -->
<div class="orb-container">
  <div class="orb orb1"></div>
  <div class="orb orb2"></div>
  <div class="orb orb3"></div>
</div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge">✨ AI Pricing Engine</div>
  <h1 class="hero-title">💻 Laptop Price AI Predictor</h1>
  <p class="hero-sub">Configure specs below and click predict to calculate the estimated price</p>
</div>
""", unsafe_allow_html=True)

# ── Section 1: Identity & Class ──
st.markdown("""
<div class="card">
  <div class="card-header">
    <div class="card-icon icon-brand">🏷️</div>
    <div class="card-title">Brand &amp; Category</div>
  </div>
</div>
""", unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    company = st.selectbox(
        "🏢 Brand / Manufacturer", 
        sorted(df["Company"].unique()),
        key="company",
        index=None,
        placeholder="Choose manufacturer",
        help="Select the brand of the laptop manufacturer."
    )
with col_b:
    laptop_type = st.selectbox(
        "📂 Type Category", 
        df["TypeName"].unique(),
        key="laptop_type",
        index=None,
        placeholder="Choose category",
        help="Form factor / style of the device."
    )

st.markdown("<div style='margin-bottom:0.5rem'></div>", unsafe_allow_html=True)

# ── Section 2: Hardware Core ──
st.markdown("""
<div class="card">
  <div class="card-header">
    <div class="card-icon icon-hw">⚙️</div>
    <div class="card-title">Performance Specs</div>
  </div>
</div>
""", unsafe_allow_html=True)

col_c, col_d = st.columns(2)
with col_c:
    ram = st.selectbox(
        "🧠 System Memory (RAM)", 
        [2, 4, 6, 8, 12, 16, 24, 32, 64], 
        key="ram",
        index=None,
        placeholder="Choose RAM size",
        help="System RAM in gigabytes. More RAM allows more open applications."
    )
    cpu = st.selectbox(
        "🔬 Processor Core (CPU)", 
        df["Cpu brand"].unique(),
        key="cpu",
        index=None,
        placeholder="Choose processor core",
        help="The central processing unit brand and architecture."
    )
with col_d:
    ssd = st.selectbox(
        "⚡ Fast Storage (SSD)", 
        [0, 8, 128, 256, 512, 1024], 
        key="ssd",
        index=None,
        placeholder="Choose SSD capacity",
        help="Solid State Drive storage capacity. Higher is faster and holds more files."
    )
    hdd = st.selectbox(
        "💿 Mass Storage (HDD)", 
        [0, 128, 256, 512, 1024, 2048], 
        key="hdd",
        index=None,
        placeholder="Choose HDD capacity",
        help="Traditional Hard Disk Drive storage size."
    )
    
gpu = st.selectbox(
    "🎮 Graphics Processor (GPU)", 
    df["Gpu brand"].unique(),
    key="gpu",
    index=None,
    placeholder="Choose graphics processor",
    help="Graphics processing chip. Essential for rendering, gaming, and 3D."
)

st.markdown("<div style='margin-bottom:0.5rem'></div>", unsafe_allow_html=True)

# ── Section 3: Display Spec ──
st.markdown("""
<div class="card">
  <div class="card-header">
    <div class="card-icon icon-disp">🖥️</div>
    <div class="card-title">Screen Details</div>
  </div>
</div>
""", unsafe_allow_html=True)

col_e, col_f = st.columns(2)
with col_e:
    screen_size = st.number_input(
        "📐 Diagonal Size (inches)", 
        min_value=10.0, 
        max_value=20.0, 
        value=None,
        placeholder="e.g. 15.6",
        key="screen_size",
        step=0.1,
        help="Diagonal length of the display screen."
    )
    touchscreen = st.selectbox(
        "👆 Touchscreen Panel", 
        ["No", "Yes"],
        key="touchscreen",
        index=None,
        placeholder="Select choice",
        help="Does the display support finger or stylus touch?"
    )
with col_f:
    resolution = st.selectbox(
        "🔍 Standard Resolution",
        ["1920x1080", "1366x768", "1600x900", "3840x2160",
         "3200x1800", "2880x1800", "2560x1600", "2560x1440", "2304x1440"],
        key="resolution",
        index=None,
        placeholder="Choose screen resolution",
        help="Total pixels on display. Higher resolution means clearer images and text."
    )
    ips = st.selectbox(
        "🌈 IPS Panel technology", 
        ["No", "Yes"],
        key="ips",
        index=None,
        placeholder="Select choice",
        help="IPS panel offers wider viewing angles and richer colors."
    )

st.markdown("<div style='margin-bottom:0.5rem'></div>", unsafe_allow_html=True)

# ── Section 4: Portability & OS ──
st.markdown("""
<div class="card">
  <div class="card-header">
    <div class="card-icon icon-other">📋</div>
    <div class="card-title">Portability &amp; OS</div>
  </div>
</div>
""", unsafe_allow_html=True)

col_g, col_h = st.columns(2)
with col_g:
    weight = st.number_input(
        "⚖️ Net Weight (kg)", 
        min_value=0.5, 
        max_value=5.0, 
        value=None,
        placeholder="e.g. 1.8",
        key="weight",
        step=0.1,
        help="Total physical weight of the laptop chassis."
    )
with col_h:
    os = st.selectbox(
        "🖥️ OS / Software Environment", 
        df["os"].unique(),
        key="os",
        index=None,
        placeholder="Choose operating system",
        help="The primary preinstalled operating system."
    )

st.markdown("<div style='margin-bottom:1rem'></div>", unsafe_allow_html=True)

# Side-by-side Button columns
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    predict_btn = st.button("🔮 Predict Laptop Price", use_container_width=True)
with col_btn2:
    reset_btn = st.button("🔄 Predict Another", on_click=reset_all, use_container_width=True)

# ── Prediction Action ─────────────────────────────────────────────────────────
if predict_btn:
    inputs = [company, laptop_type, ram, cpu, ssd, hdd, gpu, screen_size, touchscreen, resolution, ips, weight, os]
    
    # Check if any input value is unselected/empty
    if any(v is None for v in inputs):
        st.warning("⚠️ Please fill in and select all options before predicting.")
    else:
        ts_val  = 1 if touchscreen == "Yes" else 0
        ips_val = 1 if ips == "Yes" else 0

        x_res, y_res = map(int, resolution.split("x"))
        screen_sz_clamped = max(screen_size, 10.0)
        ppi = ((x_res**2 + y_res**2) ** 0.5) / screen_sz_clamped

        query_df = pd.DataFrame({
            "Company":     [company],
            "TypeName":    [laptop_type],
            "Ram":         [ram],
            "Weight":      [weight],
            "Touchscreen": [ts_val],
            "Ips":         [ips_val],
            "ppi":         [ppi],
            "Cpu brand":   [cpu],
            "HDD":         [hdd],
            "SSD":         [ssd],
            "Gpu brand":   [gpu],
            "os":          [os],
        })

        try:
            with st.spinner("🤖 Running AI prediction…"):
                # Load pipeline lazily on first prediction action
                pipe = load_pipeline()
                predicted_val = np.exp(pipe.predict(query_df)[0])
                predicted_price = int(predicted_val)
            
            # Store in session state for persistence
            st.session_state.predicted = True
            st.session_state.predicted_price = predicted_price
            st.session_state.last_config = {
                "company": company,
                "laptop_type": laptop_type,
                "ram": ram,
                "ssd": ssd,
                "hdd": hdd,
                "resolution": resolution,
                "weight": weight,
                "os": os
            }
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# Render prediction if session state has predicted
if st.session_state.predicted:
    cfg = st.session_state.last_config
    st.markdown(f"""
    <div class="result-wrap">
      <div class="result-card">
        <span class="result-emoji">💎</span>
        <div style="font-size: 0.8rem; font-weight: 700; color: #4f46e5; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 0.2rem;">Estimated Market Price</div>
        <div class="result-amount">₹ {st.session_state.predicted_price:,}</div>
        <div style="font-size: 0.85rem; color: #475569; font-weight: 500;">Based on {cfg.get('company')} {cfg.get('laptop_type')} config</div>
        <div class="spec-chips">
          <span class="chip">{cfg.get('ram')} GB RAM</span>
          <span class="chip">{cfg.get('ssd') if cfg.get('ssd') else 0} GB SSD</span>
          <span class="chip">{cfg.get('resolution')}</span>
          <span class="chip">{cfg.get('weight')} kg</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Built with <span>Streamlit</span> &amp; <span>scikit-learn</span> · 
  Interactive configuration engine
</div>
""", unsafe_allow_html=True)
