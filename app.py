import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta, datetime
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set consistent color scheme
PRIMARY_COLOR = "#0a3d62"  # Dark blue
SECONDARY_COLOR = "#ffc857"  # Golden yellow
BACKGROUND_COLOR = "#f0f4f8"  # Light blue-gray
TEXT_COLOR = "#0a3d62"  # Dark blue
ACCENT_COLOR = "#1e5799"  # Medium blue
SUCCESS_COLOR = "#4CAF50"  # Green

STANDARD_FIGSIZE = (12, 6)
LARGE_FIGSIZE = (14, 8)
SMALL_FIGSIZE = (10, 6)

# Page configuration
st.set_page_config(
    page_title="Karoo Basin Water Intelligence",
    page_icon="🌵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force light mode with comprehensive CSS
st.markdown(f"""
<style>
    /* Base styles */
    :root {{
        --primary: {PRIMARY_COLOR};
        --secondary: {SECONDARY_COLOR};
        --background: {BACKGROUND_COLOR};
        --text: {TEXT_COLOR};
        --accent: {ACCENT_COLOR};
        --success: {SUCCESS_COLOR};
    }}
    
    /* Force light mode globally */
    html, body, [data-testid="stAppViewContainer"], 
    [data-testid="stSidebar"], .stApp {{
        background-color: {BACKGROUND_COLOR} !important;
        color: {TEXT_COLOR} !important;
    }}
    
    /* Text elements */
    h1, h2, h3, h4, h5, h6, p, div, span, li {{
        color: {TEXT_COLOR} !important;
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background-color: {BACKGROUND_COLOR} !important;
        border-right: 1px solid #d9e6f2;
    }}
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stFileUploader label {{
        color: {TEXT_COLOR} !important;
        font-weight: 600;
    }}
    
    /* Input controls */
    .stSelectbox select, .stMultiSelect select,
    .stTextInput input, .stTextArea textarea,
    .stSlider input, .stNumberInput input,
    .stDateInput input, .stTimeInput input,
    .stFileUploader > div > div {{
        background-color: white !important;
        color: {TEXT_COLOR} !important;
        border: 1px solid #d9e6f2 !important;
        border-radius: 8px !important;
    }}
    
    /* Slider value display */
    .stSlider div[data-testid="stMarkdownContainer"] {{
        color: {TEXT_COLOR} !important;
        font-weight: bold !important;
    }}
    
    /* Header */
    [data-testid="stHeader"] {{
        background-color: {PRIMARY_COLOR} !important;
    }}
    
    /* Metrics */
    [data-testid="stMetric"] {{
        background-color: white !important;
        border: 1px solid #d9e6f2 !important;
        border-radius: 12px !important;
        padding: 15px !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {TEXT_COLOR} !important;
    }}
    
    [data-testid="stMetricValue"] {{
        color: {PRIMARY_COLOR} !important;
    }}
    
    /* Tabs */
    .stTabs [role="tab"] {{
        background-color: #f8f9fa !important;
        color: {TEXT_COLOR} !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 16px !important;
        border: 1px solid #d9e6f2 !important;
        margin-right: 4px !important;
    }}
    
    .stTabs [role="tab"][aria-selected="true"] {{
        background-color: white !important;
        border-bottom: 3px solid {SECONDARY_COLOR} !important;
        color: {PRIMARY_COLOR} !important;
        font-weight: bold;
    }}
    
    /* Cover page styling */
    .cover-container {{
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {ACCENT_COLOR} 100%) !important;
        padding: 40px;
        border-radius: 15px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }}
    
    .cover-title {{
        font-size: 3.5em;
        font-weight: 800;
        text-align: center;
        margin-bottom: 10px;
        color: {SECONDARY_COLOR};
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }}
    
    .cover-subtitle {{
        font-size: 1.8em;
        text-align: center;
        margin-bottom: 40px;
        color: #f0f0f0;
    }}
    
    .stat-card {{
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        min-width: 180px;
        text-align: center;
        margin: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease;
    }}
    
    .stat-card:hover {{
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.15);
    }}
    
    .stat-number {{
        font-size: 2.5em;
        font-weight: 700;
        color: {SECONDARY_COLOR};
        margin-bottom: 10px;
    }}
    
    .stat-label {{
        font-size: 1.2em;
        color: #f0f0f0;
    }}
    
    /* Info cards */
    .info-card {{
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border-left: 4px solid {SECONDARY_COLOR};
        color: {TEXT_COLOR} !important;
    }}
    
    .info-card h4 {{
        color: {PRIMARY_COLOR} !important;
        margin-top: 0;
    }}
    
    .info-card-blue {{
        background: rgba(173, 216, 230, 0.3);
        border-left: 5px solid {ACCENT_COLOR};
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }}
    
    .info-card-blue h4 {{
        color: {ACCENT_COLOR} !important;
    }}
    
    .info-card-green {{
        background: rgba(76, 175, 80, 0.15);
        border-left: 5px solid {SUCCESS_COLOR};
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }}
    
    .info-card-green h4 {{
        color: {SUCCESS_COLOR} !important;
    }}
    
    /* Explanation boxes */
    .explanation-box {{
        background: #f8f9fa;
        border-left: 4px solid {PRIMARY_COLOR};
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 0.95em;
    }}
    
    .explanation-header {{
        color: {PRIMARY_COLOR};
        font-weight: bold;
        margin-bottom: 8px;
        font-size: 1.1em;
    }}
    
    /* Banner */
    .karoo-banner {{
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {ACCENT_COLOR} 100%);
        color: white;
        padding: 20px;
        border-radius: 14px;
        margin-bottom: 20px;
    }}
    
    .karoo-badge {{
        background: {SECONDARY_COLOR};
        color: {PRIMARY_COLOR};
        border-radius: 10px;
        padding: 6px 12px;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 10px;
    }}
    
    /* Team section */
    .team-attribution {{
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {ACCENT_COLOR} 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin-top: 30px;
        text-align: center;
    }}
    
    .team-member {{
        background: rgba(255,200,87,0.9);
        padding: 10px 15px;
        border-radius: 8px;
        margin: 8px;
        display: inline-block;
        color: {PRIMARY_COLOR};
        font-weight: bold;
    }}
    
    /* Footer */
    .footer {{
        text-align: center;
        font-size: 0.9em;
        color: #607d8b;
        padding: 16px 0 6px 0;
        margin-top: 20px;
    }}
    
    /* Image caption */
    .image-caption {{
        color: {SECONDARY_COLOR};
        margin-top: 8px;
        text-align: center;
        font-weight: bold;
    }}
    
    /* Ensure all text is visible */
    .stAlert, .stWarning, .stError, .stSuccess, .stInfo {{
        color: {TEXT_COLOR} !important;
    }}
</style>
<script>
// Force light mode with JavaScript as backup
function forceLightMode() {{
    // Remove any dark mode classes
    document.querySelectorAll('*').forEach(el => {{
        el.classList.remove('dark', 'Dark', 'theme-dark');
    }});
    
    // Set light theme attributes
    document.documentElement.setAttribute('data-theme', 'light');
    document.documentElement.style.setProperty('color-scheme', 'light');
    
    // Set meta theme color
    let metaTheme = document.querySelector('meta[name="theme-color"]');
    if (!metaTheme) {{
        metaTheme = document.createElement('meta');
        metaTheme.name = 'theme-color';
        document.head.appendChild(metaTheme);
    }}
    metaTheme.content = '{BACKGROUND_COLOR}';
}}

// Apply on load and continuously monitor
if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', forceLightMode);
}} else {{
    forceLightMode();
}}

// Set up mutation observer to catch any dynamic changes
const observer = new MutationObserver(forceLightMode);
observer.observe(document.body, {{ 
    childList: true, 
    subtree: true,
    attributes: true,
    attributeFilter: ['class', 'data-theme']
}});

// Periodic check as safety net
setInterval(forceLightMode, 1000);
</script>
""", unsafe_allow_html=True)

def show_cover_page():
    st.markdown("""
    <div class="cover-container">
        <div class="cover-title">🌵 Karoo Basin Water Intelligence</div>
        <div class="cover-subtitle">AI-Powered Sustainable Water Management for Fracking Operations</div>
        <div class="cover-stats">
            <div class="stat-card">
                <div class="stat-number">3-8M</div>
                <div class="stat-label">Gallons per well</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">🔥</div>
                <div class="stat-label">Water Scarcity</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">🤖</div>
                <div class="stat-label">AI Supervised</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="padding: 20px;">
        <h2 style="text-align: center; margin-bottom: 30px;">
            Addressing the Water-Energy Nexus in South Africa's Karoo Basin
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>⚡ What is Hydraulic Fracturing (Fracking)?</h4>
            <p>Fracking is a method of extracting natural gas and oil from deep underground by injecting 
            high-pressure fluid into shale rock formations to create fractures, allowing hydrocarbons to flow 
            to the surface.</p>
            <p><strong>Process involves:</strong></p>
            <ul>
                <li>Drilling vertically then horizontally into shale formations</li>
                <li>Injecting high-pressure fracking fluid (water, sand, chemicals)</li>
                <li>Collecting released natural gas or oil</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.image("https://images.unsplash.com/photo-1584697964358-3e14ca57658b?w=600&h=400&fit=crop", 
                 use_container_width=True)
        st.markdown('<p class="image-caption">Fracking Operation Diagram</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4>🌵 The Karoo Basin Context</h4>
        <p>The Karoo Basin is a vast semi-arid region in South Africa, spanning roughly 400,000 km² across the central and western parts of the country. 
        It is characterized by wide open plains, sparse vegetation, and a unique ecosystem supporting sheep farming and small-scale agriculture. 
        The region is known for its rich geological formations, including significant shale gas potential, making it a site of interest for hydraulic fracturing (fracking) operations.</p>
        <p><strong>Key challenges:</strong></p>
        <ul>
            <li>Annual rainfall is only 100–400 mm</li>
            <li>Water scarcity affects communities, agriculture, and industry</li>
            <li>Fracking requires 3–8 million gallons of water per well</li>
            <li>Competition between energy, agriculture, and local communities</li>
            <li>Groundwater contamination risks from chemicals and methane</li>
            <li>High evaporation rates due to arid climate</li>
            <li>Limited infrastructure for water storage and recycling</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <h3 style="margin-bottom: 20px;">🚨 Critical Challenges in the Karoo:</h3>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card-blue">
        <h4>💧 Water Scarcity & Competition</h4>
        <p>The Karoo region receives minimal rainfall (100-400 mm annually), creating intense competition for water between:</p>
        <ul>
            <li>Fracking operations requiring millions of gallons per well</li>
            <li>Agricultural needs for livestock and limited crops</li>
            <li>Community drinking water requirements</li>
            <li>Ecological systems needing minimum water flows</li>
        </ul>
    </div>
    
    <div class="info-card-blue">
        <h4>⚠️ Environmental Risks</h4>
        <p>Fracking poses several environmental concerns:</p>
        <ul>
            <li>Potential groundwater contamination from chemicals or methane migration</li>
            <li>Management of produced water (often saline and contaminated)</li>
            <li>Chemical spills and surface water pollution risks</li>
            <li>Earthquake risks from wastewater injection</li>
        </ul>
    </div>
    
    <div class="info-card-blue">
        <h4>🌾 Agricultural Impacts</h4>
        <p>Water diversion for fracking could threaten:</p>
        <ul>
            <li>Sheep farming (a primary Karoo industry)</li>
            <li>Limited crop production in already marginal conditions</li>
            <li>Rural livelihoods and food security</li>
            <li>Long-term soil and water quality</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <h3 style="margin-bottom: 20px;">🎯 Purpose of This Assignment</h3>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card-green">
        <h4>🤖 AI-Driven Water Management</h4>
        <p>This project tackles critical water challenges in fracking operations by leveraging AI and data analytics:</p>
        <ul>
            <li><strong>Water Optimization:</strong> AI algorithms minimize freshwater usage</li>
            <li><strong>Quality Monitoring:</strong> Real-time tracking of water quality</li>
            <li><strong>Leak Detection:</strong> Early warning systems for infrastructure issues</li>
            <li><strong>Carbon Accountability:</strong> Track and reduce emissions from water management</li>
            <li><strong>Predictive Analytics:</strong> Forecast water needs and treatment requirements</li>
        </ul>
    </div>
    
    <div class="info-card-green">
        <h4>🌍 Environmental & Community Impact</h4>
        <p>Our approach also considers long-term environmental and social benefits:</p>
        <ul>
            <li>Protecting community water supply and agricultural resources</li>
            <li>Reducing ecological disturbance from fracking operations</li>
            <li>Supporting sustainable energy production practices</li>
            <li>Integrating data-driven insights into water management policy</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Banner at the top of the app
st.markdown(f"""
<div class="karoo-banner">
    <div class="karoo-badge">Karoo Basin • Fracking Water Intelligence</div>
    <h2 style="margin:6px 0 0 0;color:white;">AI-Supervised Water Management & Carbon Accountability</h2>
    <div style="color:white;">Real-time monitoring • Predictive analytics • Adaptive decisions • ML Emissions baseline (16 months)</div>
    <div style="color:white;margin-top:6px;">Advance Chem Assignment Team</div>
</div>
""", unsafe_allow_html=True)

# Rest of your functions (ensure_datetime, safe_col, calculate_salinity, calculate_sar, 
# classify_water_quality, detect_leaks_advanced, calculate_carbon_footprint, 
# predict_water_demand_16months, generate_geological_data) remain unchanged

def ensure_datetime(df: pd.DataFrame, col="Timestamp"):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def safe_col(df: pd.DataFrame, name: str, default=np.nan):
    if name not in df.columns:
        df[name] = default
    return df

def calculate_salinity(ec_mS_cm):
    return ec_mS_cm * 640.0

def calculate_sar(na_mgL, ca_mgL, mg_mgL):
    na_meq = na_mgL / 23.0
    ca_meq = ca_mgL / 20.0
    mg_meq = mg_mgL / 12.2
    return na_meq / np.sqrt((ca_meq + mg_meq) / 2 + 1e-6)

def classify_water_quality(ec, sar, ph):
    if pd.isna(ec) or pd.isna(sar) or pd.isna(ph):
        return "Unknown"
    if ec < 0.25:
        salinity_class = "Very Low"
    elif ec < 0.75:
        salinity_class = "Low"
    elif ec < 2.25:
        salinity_class = "Medium"
    elif ec < 5.0:
        salinity_class = "High"
    else:
        salinity_class = "Very High"
    
    if sar < 10:
        sodicity_class = "Low"
    elif sar < 18:
        sodicity_class = "Medium"
    elif sar < 26:
        sodicity_class = "High"
    else:
        sodicity_class = "Very High"
    
    if 6.0 <= ph <= 8.5:
        ph_class = "Suitable"
    else:
        ph_class = "Unsuitable"
    
    return f"{salinity_class} Salinity, {sodicity_class} Sodicity, {ph_class} pH"

def detect_leaks_advanced(flow_data, pressure_data, timestamps, window_size=6):
    results = []
    if len(flow_data) < 10:
        return pd.DataFrame()
    
    flow_zscore = np.abs((flow_data - np.mean(flow_data)) / np.std(flow_data))
    pressure_zscore = np.abs((pressure_data - np.mean(pressure_data)) / np.std(pressure_data))
    
    X = np.column_stack([flow_data, pressure_data])
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outliers = iso_forest.fit_predict(X)
    
    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(X)
    
    for i in range(len(flow_data)):
        leak_score = 0
        
        if flow_zscore[i] > 2.5 or pressure_zscore[i] > 2.5:
            leak_score += 0.3
        
        if outliers[i] == -1:
            leak_score += 0.4
        
        if clusters[i] == -1:
            leak_score += 0.3
        
        if leak_score > 0.6:
            severity = "Critical" if leak_score > 0.8 else "Warning"
            results.append({
                'timestamp': timestamps[i],
                'leak_score': leak_score,
                'severity': severity,
                'flow': flow_data[i],
                'pressure': pressure_data[i]
            })
    
    return pd.DataFrame(results)

def calculate_carbon_footprint(water_volume, treatment_kwh_m3=0.8, transport_km=20, 
                              grid_carbon_intensity=0.5, transport_intensity=0.15):
    treatment_emissions = water_volume * treatment_kwh_m3 * grid_carbon_intensity
    transport_emissions = water_volume * transport_km * transport_intensity
    operations_emissions = water_volume * 0.12  
    total_emissions = treatment_emissions + transport_emissions + operations_emissions
    
    return {
        'total_emissions': total_emissions,
        'treatment_emissions': treatment_emissions,
        'transport_emissions': transport_emissions,
        'operations_emissions': operations_emissions
    }

def predict_water_demand_16months(data, stream, months=16):
    if len(data) < 100:
        return None, None, None
    
    df = data.copy()
    df = ensure_datetime(df)
    df = df[df['Stream'] == stream]
    df = df.set_index('Timestamp')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df.resample('D')[numeric_cols].mean(numeric_only=True).ffill()
    
    df['day_of_year'] = df.index.dayofyear
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    
    last_date = df.index.max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                               periods=months*30, freq='D')
    
    if 'Temperature_C' in df.columns:
        X = df[['day_of_year', 'month', 'quarter', 'year', 'Temperature_C']].dropna()
    else:
        X = df[['day_of_year', 'month', 'quarter', 'year']].dropna()
    
    y = df['Flow_m3_h'].loc[X.index]
    
    if len(X) < 50:
        return None, None, None
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    future_features = pd.DataFrame({
        'day_of_year': future_dates.dayofyear,
        'month': future_dates.month,
        'quarter': future_dates.quarter,
        'year': future_dates.year,
        'Temperature_C': df['Temperature_C'].mean() if 'Temperature_C' in df.columns else 0  
    })
    
    predictions = model.predict(future_features)
    
    return future_dates, predictions, model.score(X, y)

def generate_geological_data():
    np.random.seed(42)
    n_samples = 200
    depth = np.random.uniform(500, 3500, n_samples)
    
    mineral_content = {
        'Na_mg_L': 50 + 0.1 * depth + np.random.normal(0, 15, n_samples),
        'Cl_mg_L': 40 + 0.1 * depth + np.random.normal(0, 15, n_samples),
        'Ca_mg_L': 30 + 0.1 * depth + np.random.normal(0, 15, n_samples),
        'Mg_mg_L': 20 + 0.1 * depth + np.random.normal(0, 15, n_samples),
        'SO4_mg_L': 10 + 0.1 * depth + np.random.normal(0, 15, n_samples),
        'TDS_mg_L': 60 + 0.1 * depth + np.random.normal(0, 15, n_samples)
    }
    
    porosity = 25 - 0.004 * depth + np.random.normal(0, 3, n_samples)
    porosity = np.clip(porosity, 2, 40)
    
    permeability = 0.5 * porosity**2 + np.random.normal(0, 50, n_samples)
    permeability = np.clip(permeability, 0.1, 2000)
    
    lithology_types = ['Sandstone', 'Shale', 'Siltstone', 'Limestone']
    lithology = np.random.choice(lithology_types, n_samples, p=[0.4, 0.3, 0.2, 0.1])
    
    mineral_content_df = pd.DataFrame(mineral_content)
    return pd.concat([mineral_content_df, pd.DataFrame({
        'Depth_m': depth,
        'Porosity_pct': porosity,
        'Permeability_mD': permeability,
        'Lithology': lithology
    })], axis=1)

# Sidebar configuration
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], 
                                   help="Upload water quality data with required columns")
    
    st.header("🔧 Analysis Mode")
    analysis_mode = st.selectbox(
        "Select Analysis Focus",
        ["Cover Page", "Water Management", "Inspection", "Leak Detection", "Predictive Analytics", 
         "Carbon Footprint", "Geological Analysis"]
    )
    
    st.header("⚙️ Time Settings")
    analysis_period = st.slider("Analysis Period (months)", 1, 24, 16, 1,
                              help="Set the period for analysis and prediction")

@st.cache_data
def load_and_preprocess_data(file):
    if file is None:
        return pd.DataFrame()
    
    df = pd.read_csv(file)
    df = ensure_datetime(df, "Timestamp")
    
    required_cols = ["Stream", "Timestamp", "Flow_m3_h", "Pressure_kPa", "EC_mS_cm", "pH", "Temperature_C"]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            return pd.DataFrame()
    
    optional_cols = ["Na_mg_L", "Cl_mg_L", "Ca_mg_L", "Mg_mg_L", "SO4_mg_L", "TDS_mg_L"]
    for col in optional_cols:
        df = safe_col(df, col, np.nan)
    
    if all(col in df.columns for col in ["Na_mg_L", "Ca_mg_L", "Mg_mg_L"]):
        df["SAR"] = calculate_sar(df["Na_mg_L"], df["Ca_mg_L"], df["Mg_mg_L"])
    
    if "EC_mS_cm" in df.columns:
        df["Salinity_mg_L"] = calculate_salinity(df["EC_mS_cm"])
    
    df["Quality_Classification"] = df.apply(
        lambda row: classify_water_quality(
            row.get("EC_mS_cm", np.nan),
            row.get("SAR", np.nan),
            row.get("pH", np.nan)
        ), axis=1
    )
    
    return df

if uploaded_file is not None:
    data = load_and_preprocess_data(uploaded_file)
else:
    geological_data = generate_geological_data()
    st.info("👆 Please upload a CSV file to begin analysis. Showing synthetic geological data for demonstration.")
    data = pd.DataFrame()

available_streams = data["Stream"].unique() if not data.empty else ["Synthetic_Data"]
selected_streams = st.sidebar.multiselect(
    "Select Streams for Analysis",
    options=available_streams,
    default=available_streams[:min(2, len(available_streams))],
    help="Select water streams to analyze"
)

if not data.empty:
    filtered_data = data[data["Stream"].isin(selected_streams)].copy()
    if analysis_period:
        latest_time = filtered_data["Timestamp"].max()
        cutoff_time = latest_time - pd.DateOffset(months=analysis_period)
        filtered_data = filtered_data[filtered_data["Timestamp"] >= cutoff_time]
else:
    filtered_data = pd.DataFrame()

tab1, tab2 = st.tabs(["Analysis", "Raw Data"])

with tab1:
    if analysis_mode == "Cover Page":
        show_cover_page()
        
    elif analysis_mode == "Water Management":
        st.header("💧 Water Management Module")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            current_flow = filtered_data["Flow_m3_h"].iloc[-1] if not filtered_data.empty else 0
            st.metric("Current Flow", f"{current_flow:.1f} m³/h")
        with col2:
            current_ec = filtered_data["EC_mS_cm"].iloc[-1] if not filtered_data.empty else 0
            st.metric("Current EC", f"{current_ec:.2f} mS/cm")
        with col3:
            current_pressure = filtered_data["Pressure_kPa"].iloc[-1] if not filtered_data.empty else 0
            st.metric("System Pressure", f"{current_pressure:.0f} kPa")
        with col4:
            total_volume = filtered_data["Flow_m3_h"].sum() if not filtered_data.empty else 0
            st.metric("Total Volume", f"{total_volume:,.0f} m³")
        
        st.subheader("Water Quality Assessment")
        if "Quality_Classification" in filtered_data.columns:
            quality_counts = filtered_data["Quality_Classification"].value_counts()
            fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
            quality_counts.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title("Water Quality Distribution")
            ax.set_ylabel("Count")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
            
            st.markdown("""
            <div class="explanation-box">
            <div class="explanation-header">Interpretation:</div>
            This chart shows the distribution of water quality classifications across all samples. 
            The classification is based on Electrical Conductivity (EC), Sodium Adsorption Ratio (SAR), and pH levels.
            <ul>
            <li><strong>High Quality</strong>: Suitable for irrigation with minimal treatment</li>
            <li><strong>Medium Quality</strong>: May require some treatment before agricultural use</li>
            <li><strong>Low Quality</strong>: Requires significant treatment or should be avoided for irrigation</li>
            </ul>
            Monitoring these distributions helps in planning water treatment requirements and ensuring
            compliance with agricultural water standards.
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("Real-time Monitoring")
        fig, axes = plt.subplots(2, 2, figsize=LARGE_FIGSIZE)
        parameters = [('Flow_m3_h', 'Flow Rate'), ('EC_mS_cm', 'Electrical Conductivity'), 
                     ('Pressure_kPa', 'System Pressure'), ('pH', 'pH Level')]
        
        for i, (param, title) in enumerate(parameters):
            ax = axes[i//2, i%2]
            for stream in selected_streams:
                stream_data = filtered_data[filtered_data["Stream"] == stream]
                ax.plot(stream_data["Timestamp"], stream_data[param], label=stream)
            ax.set_title(title)
            ax.set_ylabel(param)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add detailed interpretations for each graph
        st.markdown("""
        <div class="explanation-box">
        <div class="explanation-header">Flow Rate Interpretation:</div>
        The flow rate graph shows the volume of water moving through the system over time. 
        Consistent patterns indicate normal operation, while sudden drops may signal:
        <ul>
        <li>Equipment malfunctions or pump failures</li>
        <li>Potential leaks in the distribution system</li>
        <li>Changes in water demand patterns</li>
        <li>Maintenance activities or scheduled shutdowns</li>
        </ul>
        Regular monitoring helps identify operational issues early and maintain system efficiency.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="explanation-box">
        <div class="explanation-header">Electrical Conductivity Interpretation:</div>
        Electrical Conductivity (EC) measures water's ability to conduct electricity, which correlates with mineral content.
        <ul>
        <li>Low EC values indicate pure water with minimal dissolved salts</li>
        <li>High EC values suggest mineral-rich water that may require treatment</li>
        <li>Sudden spikes could indicate contamination events</li>
        <li>Gradual increases may suggest concentrating minerals due to evaporation</li>
        </ul>
        Monitoring EC helps determine appropriate water treatment needs for agricultural use.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="explanation-box">
        <div class="explanation-header">System Pressure Interpretation:</div>
        System pressure monitoring is critical for pipeline integrity and efficient operation.
        <ul>
        <li>Consistent pressure indicates stable system operation</li>
        <li>Pressure drops may signal leaks or blockages in the system</li>
        <li>Pressure spikes could indicate valve malfunctions or pump issues</li>
        <li>Gradual decreases might suggest deteriorating infrastructure</li>
        </ul>
        Maintaining optimal pressure ensures efficient water delivery and minimizes energy consumption.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="explanation-box">
        <div class="explanation-header">pH Level Interpretation:</div>
        pH measures the acidity or alkalinity of water, which affects its suitability for different uses.
        <ul>
        <li>Neutral pH (6.5-8.5) is ideal for most agricultural applications</li>
        <li>Low pH (acidic water) can corrode pipes and equipment</li>
        <li>High pH (alkaline water) can cause scaling and reduce treatment efficiency</li>
        <li>pH fluctuations may indicate changing water sources or treatment issues</li>
        </ul>
        Maintaining appropriate pH levels ensures water quality and protects infrastructure.
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Temperature vs Time")
        fig_temp, ax_temp = plt.subplots(figsize=STANDARD_FIGSIZE)
        for stream in selected_streams:
            stream_data = filtered_data[filtered_data["Stream"] == stream]
            ax_temp.plot(stream_data["Timestamp"], stream_data["Temperature_C"], label=stream)
        ax_temp.set_title("Temperature Over Time")
        ax_temp.set_ylabel("Temperature (°C)")
        ax_temp.legend()
        ax_temp.grid(True, alpha=0.3)
        st.pyplot(fig_temp)
        
        st.markdown("""
        <div class="explanation-box">
        <div class="explanation-header">Temperature Interpretation:</div>
        This graph illustrates the trend of temperature variations over time for selected water streams. 
        Elevated temperatures may indicate potential equipment issues or changes in incoming water sources.
        <ul>
        <li><strong>Monitoring Peaks</strong>: Sudden increases may require further investigation</li>
        <li><strong>Consistency</strong>: Stable temperature readings indicate healthy system operation</li>
        <li><strong>Outlier Considerations</strong>: Unusual patterns may signal the need for inspections</li>
        </ul>
        Regular monitoring of temperature helps ensure system efficiency and informs treatment processes.
        </div>
        """, unsafe_allow_html=True)
    
    elif analysis_mode == "Inspection":
        st.header("🔍 Inspection Module")
        st.subheader("Anomaly Detection & Inspection Scheduling")
        
        if not filtered_data.empty and all(col in filtered_data.columns for col in ["Flow_m3_h", "Pressure_kPa"]):
            X = filtered_data[["Flow_m3_h", "Pressure_kPa", "EC_mS_cm", "pH"]].dropna()
            
            if len(X) > 10:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                anomalies = iso_forest.fit_predict(X_scaled)
                anomaly_scores = iso_forest.decision_function(X_scaled)
                
                normal_count = np.sum(anomalies == 1)
                anomaly_count = np.sum(anomalies == -1)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Normal Readings", normal_count)
                with col2:
                    st.metric("Anomalies Detected", anomaly_count)
                
                if anomaly_count > 0:
                    st.warning(f"🚨 {anomaly_count} anomalies detected. Schedule inspections for these time periods.")
                    
                    st.subheader("Anomaly Detection Visualization")
                    fig1, ax1 = plt.subplots(figsize=STANDARD_FIGSIZE)
                    ax1.plot(filtered_data["Timestamp"], filtered_data["Flow_m3_h"], 
                            label="Flow Rate", color='blue', alpha=0.7)
                    ax1.set_ylabel("Flow (m³/h)", color='blue')
                    ax1.tick_params(axis='y', labelcolor='blue')
                    
                    ax2 = ax1.twinx()
                    ax2.plot(filtered_data["Timestamp"], filtered_data["Pressure_kPa"], 
                            label="Pressure", color='red', alpha=0.7)
                    ax2.set_ylabel("Pressure (kPa)", color='red')
                    ax2.tick_params(axis='y', labelcolor='red')
                    
                    anomaly_indices = np.where(anomalies == -1)[0]
                    anomaly_times = filtered_data["Timestamp"].iloc[anomaly_indices]
                    anomaly_flows = filtered_data["Flow_m3_h"].iloc[anomaly_indices]
                    
                    ax1.scatter(anomaly_times, anomaly_flows, color='orange', s=100, 
                               label='Detected Anomalies', marker='X', zorder=5)
                    
                    ax1.set_title("Flow Rate and Pressure with Detected Anomalies")
                    ax1.legend(loc='upper left')
                    ax2.legend(loc='upper right')
                    ax1.grid(True, alpha=0.3)
                    st.pyplot(fig1)
                    
                    fig2, ax3 = plt.subplots(figsize=STANDARD_FIGSIZE)
                    ax3.plot(filtered_data["Timestamp"].iloc[:len(anomaly_scores)], anomaly_scores, 
                            label="Anomaly Score", color='purple', alpha=0.7)
                    ax3.axhline(y=0, color='red', linestyle='--', label='Anomaly Threshold')
                    ax3.scatter(anomaly_times, anomaly_scores[anomaly_indices], 
                               color='orange', s=100, label='Anomalies', marker='X', zorder=5)
                    ax3.set_title("Anomaly Detection Scores Over Time")
                    ax3.set_ylabel("Anomaly Score")
                    ax3.set_xlabel("Time")
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                    st.pyplot(fig2)
                    
                    anomaly_data = filtered_data.iloc[anomalies == -1]
                    st.dataframe(anomaly_data[["Timestamp", "Stream", "Flow_m3_h", "Pressure_kPa", "EC_mS_cm"]])
                    
                    st.markdown("""
                    <div class="explanation-box">
                    <div class="explanation-header">Interpretation:</div>
                    Anomalies represent unusual patterns in the water system that deviate from normal operation.
                    These could indicate:
                    <ul>
                    <li><strong>Equipment malfunctions</strong>: Pumps, valves, or sensors not operating correctly</li>
                    <li><strong>Water quality issues</strong>: Contamination or treatment process failures</li>
                    <li><strong>System integrity problems</strong>: Developing leaks or pressure issues</li>
                    <li><strong>Operational changes</strong>: Unplanned changes in water extraction or distribution</li>
                    </ul>
                    Each anomaly should be investigated to determine the root cause and appropriate corrective action.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.subheader("Inspection Recommendations")
                    st.write("""
                    - **Immediate inspection**: High-priority anomalies
                    - **Scheduled maintenance**: Moderate priority issues  
                    - **Preventive measures**: Address root causes
                    - **Document findings**: Update inspection records
                    """)
        
        st.subheader("Water Quality Compliance")
        if "Quality_Classification" in filtered_data.columns:
            compliance_data = filtered_data.groupby("Stream")["Quality_Classification"].apply(
                lambda x: (x.str.contains("Suitable")).mean() * 100
            ).round(2)
            
            st.dataframe(compliance_data.rename("Compliance Rate (%)"))
            
            fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
            compliance_data.plot(kind='bar', ax=ax, color='green')
            ax.set_title("Water Quality Compliance by Stream")
            ax.set_ylabel("Compliance Rate (%)")
            ax.axhline(y=90, color='red', linestyle='--', label='Target Compliance (90%)')
            ax.legend()
            st.pyplot(fig)
            
            st.markdown("""
            <div class="explanation-box">
            <div class="explanation-header">Interpretation:</div>
            Compliance rates measure the percentage of water samples that meet quality standards for agricultural use.
            <ul>
            <li><strong>Above 90%</strong>: Excellent compliance. Water quality is consistently suitable for irrigation.</li>
            <li><strong>75-90%</strong>: Good compliance. Minor improvements may be needed.</li>
            <li><strong>Below 75%</strong>: Concerning compliance. Significant improvements needed in water treatment.</li>
            </ul>
            The red line shows the target compliance rate of 90%. Streams falling below this target
            require immediate attention and potential process improvements.
            </div>
            """, unsafe_allow_html=True)
    
    elif analysis_mode == "Leak Detection":
        st.header("🚨 Leak Detection Module")
        
        if not filtered_data.empty and all(col in filtered_data.columns for col in ["Flow_m3_h", "Pressure_kPa"]):
            leak_results = []
            
            for stream in selected_streams:
                stream_data = filtered_data[filtered_data["Stream"] == stream]
                if len(stream_data) > 10:
                    leaks = detect_leaks_advanced(
                        stream_data["Flow_m3_h"].values,
                        stream_data["Pressure_kPa"].values,
                        stream_data["Timestamp"].values
                    )
                    
                    if not leaks.empty:
                        leaks["Stream"] = stream
                        leak_results.append(leaks)
            
            if leak_results:
                all_leaks = pd.concat(leak_results, ignore_index=True)
                st.error(f"🚨 {len(all_leaks)} potential leaks detected!")
                
                critical_leaks = all_leaks[all_leaks["severity"] == "Critical"]
                warning_leaks = all_leaks[all_leaks["severity"] == "Warning"]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Critical Leaks", len(critical_leaks))
                with col2:
                    st.metric("Warning Leaks", len(warning_leaks))
                
                st.markdown("""
                <div class="explanation-box">
                <div class="explanation-header">Interpretation:</div>
                The leak detection system uses multiple machine learning algorithms to identify potential leaks:
                <ul>
                <li><strong>Critical Leaks</strong>: High confidence detections requiring immediate attention</li>
                <li><strong>Warning Leaks</strong>: Potential leaks that should be monitored and investigated</li>
                </ul>
                Leaks are detected based on abnormal patterns in flow and pressure data that deviate from
                normal system operation. Early detection helps minimize water loss and infrastructure damage.
                </div>
                """, unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
                for stream in selected_streams:
                    stream_data = filtered_data[filtered_data["Stream"] == stream]
                    ax.plot(stream_data["Timestamp"], stream_data["Flow_m3_h"], label=f"{stream} Flow", alpha=0.7)
                
                if not critical_leaks.empty:
                    ax.scatter(critical_leaks["timestamp"], 
                              [max(filtered_data["Flow_m3_h"]) * 0.9] * len(critical_leaks),
                              color='red', s=100, label='Critical Leaks', marker='X')
                
                if not warning_leaks.empty:
                    ax.scatter(warning_leaks["timestamp"], 
                              [max(filtered_data["Flow_m3_h"]) * 0.85] * len(warning_leaks),
                              color='orange', s=80, label='Warning Leaks', marker='^')
                
                ax.set_title("Flow Rate with Detected Leaks")
                ax.set_ylabel("Flow (m³/h)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                st.subheader("Leak Response Protocol")
                if not critical_leaks.empty:
                    st.error("""
                    **CRITICAL LEAKS DETECTED**  
                    → Immediate shutdown of affected sections  
                    → Dispatch emergency repair team  
                    → Notify operations manager  
                    → Implement contingency water supply
                    """)
                
                if not warning_leaks.empty:
                    st.warning("""
                    **WARNING LEVEL LEAKS**  
                    → Schedule inspection within 24 hours  
                    → Increase monitoring frequency  
                    → Prepare repair resources  
                    → Update maintenance schedule
                    """)
            else:
                st.success("✅ No leaks detected in the current data")
                st.markdown("""
                <div class="explanation-box">
                <div class="explanation-header">Interpretation:</div>
                No leaks have been detected in the current dataset. This indicates:
                <ul>
                <li>Pipeline integrity is maintained</li>
                <li>System pressure and flow patterns are normal</li>
                <li>Equipment is functioning properly</li>
                </ul>
                Continue regular monitoring to maintain system integrity and quickly detect any future issues.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Flow and pressure data required for leak detection")
    
    elif analysis_mode == "Predictive Analytics":
        st.header("🤖 Predictive Analytics Module")
        st.subheader("16-Month Water Demand Forecast")
        
        if len(filtered_data) > 100 and all(col in filtered_data.columns for col in ["Flow_m3_h", "Temperature_C"]):
            with st.spinner("Training predictive models for 16-month forecast..."):
                fig_size = LARGE_FIGSIZE
                
                st.subheader("Water Flow Prediction")
                fig_flow, ax_flow = plt.subplots(figsize=fig_size)
                
                for stream in selected_streams:
                    stream_data = filtered_data[filtered_data["Stream"] == stream].copy()
                    stream_data = stream_data.set_index("Timestamp")
                    daily_data = stream_data.resample('D').mean(numeric_only=True)
                    
                    daily_data['day_of_year'] = daily_data.index.dayofyear
                    daily_data['month'] = daily_data.index.month
                    daily_data['quarter'] = daily_data.index.quarter
                    daily_data['year'] = daily_data.index.year
                    
                    last_date = daily_data.index.max()
                    future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                               periods=16*30, freq='D')
                    
                    future_features = pd.DataFrame({
                        'day_of_year': future_dates.dayofyear,
                        'month': future_dates.month,
                        'quarter': future_dates.quarter,
                        'year': future_dates.year,
                        'Temperature_C': daily_data['Temperature_C'].mean() if 'Temperature_C' in daily_data.columns else 0  
                    })
                    
                    X = daily_data[['day_of_year', 'month', 'quarter', 'year', 'Temperature_C']].dropna()
                    y = daily_data['Flow_m3_h'].loc[X.index]
                    
                    if len(X) > 50:
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                        model.fit(X, y)
                        predictions = model.predict(future_features)
                        
                        historical_cutoff = last_date - pd.DateOffset(months=12)
                        historical_data = daily_data[daily_data.index >= historical_cutoff]
                        
                        ax_flow.plot(historical_data.index, historical_data['Flow_m3_h'], 
                                   label=f"{stream} Historical", linewidth=2)
                        ax_flow.plot(future_dates, predictions, 
                                   label=f"{stream} Forecast", linewidth=2, linestyle='--')
                
                ax_flow.set_title("16-Month Water Flow Forecast")
                ax_flow.set_ylabel("Flow (m³/h)")
                ax_flow.set_xlabel("Date")
                ax_flow.legend()
                ax_flow.grid(True, alpha=0.3)
                st.pyplot(fig_flow)
                
                st.markdown(f"""
                <div class="explanation-box">
                <div class="explanation-header">Flow Forecast Interpretation:</div>
                This water flow forecast shows historical patterns and future predictions for the selected streams.
                <ul>
                <li><strong>Historical Trend</strong>: The solid lines show actual water flow measurements over the past year</li>
                <li><strong>Future Prediction</strong>: The dashed lines represent the AI-powered 16-month forecast</li>
                <li><strong>Seasonal Patterns</strong>: Notice the repeating patterns that correspond to seasonal water demand changes</li>
                <li><strong>Peak Periods</strong>: The highest flow rates typically occur between {future_dates[predictions.argmax()].strftime('%B')} and {future_dates[predictions.argmax() + 90].strftime('%B')}</li>
                <li><strong>Low Demand Periods</strong>: Lowest flow is predicted around {future_dates[predictions.argmin()].strftime('%B %Y')}</li>
                """, unsafe_allow_html=True)
                
                parameters_to_predict = ['EC_mS_cm', 'pH', 'Pressure_kPa', 'Temperature_C']
                parameter_descriptions = {
                    'EC_mS_cm': 'Electrical Conductivity',
                    'pH': 'pH Level', 
                    'Pressure_kPa': 'System Pressure',
                    'Temperature_C': 'Temperature'
                }
                
                parameter_interpretations = {
                    'EC_mS_cm': """
                    <ul>
                    <li><strong>Mineral Content Indicator</strong>: Higher values indicate increased dissolved minerals and salts</li>
                    <li><strong>Treatment Requirements</strong>: Rising EC may require additional treatment processes</li>
                    <li><strong>Seasonal Variations</strong>: EC often increases during dry periods due to evaporation</li>
                    <li><strong>Quality Implications</strong>: Values above 0.8 mS/cm may affect agricultural suitability</li>
                    <li><strong>Trend Analysis</strong>: The forecast helps plan for treatment chemical requirements</li>
                    """,
                    'pH': """
                    <ul>
                    <li><strong>Acidity/Alkalinity Balance</strong>: Optimal range for agricultural use is 6.5-8.5</li>
                    <li><strong>Corrosion Potential</strong>: Low pH values (<6.5) may indicate corrosive water</li>
                    <li><strong>Scaling Potential</strong>: High pH values (>8.5) may lead to scaling in pipes</li>
                    <li><strong>Treatment Needs</strong>: pH adjustments may be required for optimal treatment efficiency</li>
                    <li><strong>Biological Impact</strong>: pH affects microbial activity in treatment processes</li>
                    """,
                    'Pressure_kPa': """
                    <ul>
                    <li><strong>System Integrity</strong>: Stable pressure indicates well-maintained infrastructure</li>
                    <li><strong>Energy Consumption</strong>: Pressure variations affect pumping energy requirements</li>
                    <li><strong>Leak Detection</strong>: Unexpected pressure drops may indicate developing leaks</li>
                    <li><strong>Demand Patterns</strong>: Pressure fluctuations often correlate with usage patterns</li>
                    <li><strong>Maintenance Planning</strong>: Forecast helps schedule pump maintenance during low-pressure periods</li>
                    """,
                    'Temperature_C': """
                    <ul>
                    <li><strong>Seasonal Variations</strong>: Temperature follows predictable seasonal patterns</li>
                    <li><strong>Treatment Efficiency</strong>: Biological treatment processes are temperature-sensitive</li>
                    <li><strong>Energy Requirements</strong>: Water heating/cooling needs vary with temperature</li>
                    <li><strong>Usage Patterns</strong>: Higher temperatures often correlate with increased water demand</li>
                    <li><strong>Environmental Impact</strong>: Discharge temperature affects receiving water ecosystems</li>
                    """
                }
                
                for parameter in parameters_to_predict:
                    if parameter in filtered_data.columns:
                        st.subheader(f"{parameter_descriptions[parameter]} Prediction")
                        fig_param, ax_param = plt.subplots(figsize=fig_size)
                        
                        for stream in selected_streams:
                            stream_data = filtered_data[filtered_data["Stream"] == stream].copy()
                            stream_data = stream_data.set_index("Timestamp")
                            daily_data = stream_data.resample('D').mean(numeric_only=True)
                            
                            daily_data['day_of_year'] = daily_data.index.dayofyear
                            daily_data['month'] = daily_data.index.month
                            daily_data['quarter'] = daily_data.index.quarter
                            daily_data['year'] = daily_data.index.year
                            
                            last_date = daily_data.index.max()
                            future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                                       periods=16*30, freq='D')
                            
                            future_features = pd.DataFrame({
                                'day_of_year': future_dates.dayofyear,
                                'month': future_dates.month,
                                'quarter': future_dates.quarter,
                                'year': future_dates.year
                            })
                            
                            X = daily_data[['day_of_year', 'month', 'quarter', 'year']].dropna()
                            y = daily_data[parameter].loc[X.index]
                            
                            if len(X) > 50:
                                model = RandomForestRegressor(n_estimators=100, random_state=42)
                                model.fit(X, y)
                                predictions = model.predict(future_features)
                                
                                historical_cutoff = last_date - pd.DateOffset(months=12)
                                historical_data = daily_data[daily_data.index >= historical_cutoff]
                                
                                ax_param.plot(historical_data.index, historical_data[parameter], 
                                           label=f"{stream} Historical", linewidth=2)
                                ax_param.plot(future_dates, predictions, 
                                           label=f"{stream} Forecast", linewidth=2, linestyle='--')
                        
                        ax_param.set_title(f"16-Month {parameter_descriptions[parameter]} Forecast")
                        ax_param.set_ylabel(parameter)
                        ax_param.set_xlabel("Date")
                        ax_param.legend()
                        ax_param.grid(True, alpha=0.3)
                        st.pyplot(fig_param)
                        
                        st.markdown(f"""
                        <div class="explanation-box">
                        <div class="explanation-header">{parameter_descriptions[parameter]} Forecast Interpretation:</div>
                        This {parameter_descriptions[parameter].lower()} forecast shows both historical measurements and future predictions.
                        {parameter_interpretations[parameter]}
                        </div>
                        """, unsafe_allow_html=True)
                
                st.subheader("Operational Recommendations")
                recommendations = []
                
                if 'future_dates' in locals() and 'predictions' in locals():
                    peak_month = future_dates[predictions.argmax()].strftime('%B')
                    low_month = future_dates[predictions.argmin()].strftime('%B')
                    recommendations.append(f"→ Prepare for peak demand in {peak_month} with additional storage capacity")
                    recommendations.append(f"→ Schedule maintenance during low-demand period in {low_month}")
                
                concerning_trends = []
                if 'EC_mS_cm' in parameters_to_predict and 'EC_mS_cm' in filtered_data.columns:
                    avg_ec = filtered_data['EC_mS_cm'].mean()
                    if avg_ec > 0.7:
                        concerning_trends.append("Elevated Electrical Conductivity suggests increasing mineral content")
                
                if 'pH' in parameters_to_predict and 'pH' in filtered_data.columns:
                    avg_ph = filtered_data['pH'].mean()
                    if avg_ph < 6.5 or avg_ph > 8.5:
                        concerning_trends.append(f"pH level ({avg_ph:.1f}) outside optimal range for agricultural use") 
                
                if concerning_trends:
                    recommendations.append("→ Monitor water quality parameters closely and adjust treatment processes")
                    for trend in concerning_trends:
                        recommendations.append(f"   - {trend}")   
                
                recommendations.extend([
                    "→ Optimize treatment plant operations based on predicted demand patterns",
                    "→ Coordinate with agricultural users for water conservation during peak periods",
                    "→ Ensure flexibility in operational strategies to adapt to forecasted changes"
                ])   
                
                st.info("**Based on 16-month forecast:**  \n" + "  \n".join(recommendations))      
        else:
            st.warning("Need at least 100 records with flow and temperature data for forecasting")
    
    elif analysis_mode == "Carbon Footprint":
        st.header("🌍 Carbon Footprint Module")

        if not filtered_data.empty:
            total_volume = filtered_data["Flow_m3_h"].sum()

            st.subheader("Emission Factors Configuration")
            col1, col2, col3 = st.columns(3)
            with col1:
                treatment_energy = st.slider("Treatment Energy (kWh/m³)", 0.1, 2.0, 0.8, 0.1)
            with col2:
                transport_dist = st.slider("Transport Distance (km)", 1, 100, 20, 1)
            with col3:
                energy_carbon = st.slider("Grid Carbon Intensity (kg CO₂/kWh)", 0.1, 1.0, 0.5, 0.05)

            emissions = calculate_carbon_footprint(total_volume, treatment_energy, transport_dist, 
                                                 energy_carbon, 0.15)

            st.subheader("Carbon Emission Results (16-Month Baseline)")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Water", f"{total_volume:,.0f} m³")
            with col2:
                st.metric("Total CO₂", f"{emissions['total_emissions']:,.0f} kg")
            with col3:
                st.metric("Treatment CO₂", f"{emissions['treatment_emissions']:,.0f} kg")
            with col4:
                st.metric("Transport CO₂", f"{emissions['transport_emissions']:,.0f} kg")
            
            st.subheader("Emission Calculation Breakdown")
            calculation_data = {
                'Component': [
                    'Water Treatment', 
                    'Water Transport', 
                    'Operations',
                    'Total Emissions'
                ],
                'Calculation': [
                    f"{total_volume:,.0f} m³ × {treatment_energy} kWh/m³ × {energy_carbon} kg CO₂/kWh",
                    f"{total_volume:,.0f} m³ × {transport_dist} km × 0.15 kg CO₂/m³/km",
                    f"{total_volume:,.0f} m³ × 0.12 kg CO₂/m³",
                    "Sum of all components"
                ],
                'Emissions (kg CO₂)': [
                    emissions['treatment_emissions'],
                    emissions['transport_emissions'],
                    emissions['operations_emissions'],
                    emissions['total_emissions']
                ]
            }
            
            calculation_df = pd.DataFrame(calculation_data)
            st.dataframe(calculation_df)
            
            st.subheader("Emission Breakdown")
            fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
            emission_types = ['Treatment', 'Transport', 'Operations']
            emission_values = [
                emissions['treatment_emissions'],
                emissions['transport_emissions'], 
                emissions['operations_emissions']
            ]
            
            colors = [SECONDARY_COLOR, ACCENT_COLOR, '#F25C54']
            wedges, texts, autotexts = ax.pie(emission_values, labels=emission_types, autopct='%1.1f%%', 
                                             startangle=90, colors=colors, shadow=True)
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(11)
            
            for text in texts:
                text.set_fontweight('bold')
                text.set_fontsize(11)
            
            ax.set_title('Carbon Emission Breakdown', fontweight='bold', fontsize=14, pad=20)
            ax.axis('equal')
            
            legend_labels = [f'{label}: {value:,.0f} kg CO₂' for label, value in zip(emission_types, emission_values)]
            ax.legend(wedges, legend_labels, title="Emissions Breakdown", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            st.pyplot(fig)
            
            st.markdown("""
            <div class="explanation-box">
            <div class="explanation-header">Interpretation:</div>
            Carbon footprint calculation based on the ML Emissions Calculator methodology:
            <ul>
            <li><strong>Total Water Volume</strong>: The amount of water processed over the analysis period</li>
            <li><strong>Treatment Emissions</strong>: CO₂ from energy used in water treatment processes</li>
            <li><strong>Transport Emissions</strong>: CO₂ from transporting water through the distribution system</li>
            <li><strong>Operations Emissions</strong>: CO₂ from general operational activities</li>
            </ul>
            These calculations help quantify the environmental impact of water management operations
            and identify opportunities for emissions reduction.
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Daily Carbon Footprint Analysis")
            daily_data = filtered_data.copy()
            daily_data['Date'] = pd.to_datetime(daily_data['Timestamp']).dt.date
            
            daily_totals = daily_data.groupby('Date').agg({
                'Flow_m3_h': 'sum',
                'EC_mS_cm': 'mean',
                'pH': 'mean',
                'Pressure_kPa': 'mean'
            }).reset_index()
            
            daily_totals['Treatment_Emissions_kg'] = daily_totals['Flow_m3_h'] * treatment_energy * energy_carbon
            daily_totals['Transport_Emissions_kg'] = daily_totals['Flow_m3_h'] * transport_dist * 0.15
            daily_totals['Operations_Emissions_kg'] = daily_totals['Flow_m3_h'] * 0.12
            daily_totals['Total_Emissions_kg'] = (daily_totals['Treatment_Emissions_kg'] + 
                                                daily_totals['Transport_Emissions_kg'] + 
                                                daily_totals['Operations_Emissions_kg'])
            daily_totals['Emissions_per_m3_kg'] = daily_totals['Total_Emissions_kg'] / daily_totals['Flow_m3_h']
            
            st.subheader("Daily Emissions Trend")
            fig_trend, ax_trend = plt.subplots(figsize=STANDARD_FIGSIZE)
            ax_trend.plot(daily_totals['Date'], daily_totals['Total_Emissions_kg'], 
                         marker='o', linewidth=2, markersize=4, color=PRIMARY_COLOR)
            ax_trend.fill_between(daily_totals['Date'], daily_totals['Total_Emissions_kg'], 
                                alpha=0.3, color=SECONDARY_COLOR)
            ax_trend.set_title('Daily Carbon Emissions Trend', fontweight='bold')
            ax_trend.set_ylabel('CO₂ Emissions (kg)')
            ax_trend.set_xlabel('Date')
            ax_trend.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_trend)
            
            st.subheader("Daily Emissions Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Daily Emissions", f"{daily_totals['Total_Emissions_kg'].mean():.1f} kg CO₂")
            with col2:
                st.metric("Max Daily Emissions", f"{daily_totals['Total_Emissions_kg'].max():.1f} kg CO₂")
            with col3:
                st.metric("Min Daily Emissions", f"{daily_totals['Total_Emissions_kg'].min():.1f} kg CO₂")
            with col4:
                st.metric("Avg Emissions/m³", f"{daily_totals['Emissions_per_m3_kg'].mean():.3f} kg CO₂/m³")
            
            st.subheader("Download Daily Results")
            daily_results = daily_totals.copy()
            daily_results.rename(columns={
                'Date': 'Date',
                'Flow_m3_h': 'Daily_Water_Volume_m3',
                'EC_mS_cm': 'Avg_EC_mS_cm',
                'pH': 'Avg_pH',
                'Pressure_kPa': 'Avg_Pressure_kPa',
                'Treatment_Emissions_kg': 'Treatment_Emissions_kg_CO2',
                'Transport_Emissions_kg': 'Transport_Emissions_kg_CO2',
                'Operations_Emissions_kg': 'Operations_Emissions_kg_CO2',
                'Total_Emissions_kg': 'Total_Emissions_kg_CO2',
                'Emissions_per_m3_kg': 'Emissions_per_m3_kg_CO2'
            }, inplace=True)
            
            daily_results['Treatment_Calculation'] = f"Volume × {treatment_energy} kWh/m³ × {energy_carbon} kg CO₂/kWh"
            daily_results['Transport_Calculation'] = f"Volume × {transport_dist} km × 0.15 kg CO₂/m³/km"
            daily_results['Operations_Calculation'] = "Volume × 0.12 kg CO₂/m³"
            
            csv_daily = daily_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Daily Carbon Footprint Results",
                data=csv_daily,
                file_name="daily_carbon_footprint_results.csv",
                mime="text/csv",
                help="Download detailed daily carbon footprint calculation results"
            )
            
            st.write("Daily Results Preview (first 10 days):")
            st.dataframe(daily_results.head(10))
            
            st.subheader("Sustainability Action Plan")
            avg_daily_emissions = daily_totals['Total_Emissions_kg'].mean()
            
            if avg_daily_emissions > 100:
                st.error("""
                **HIGH CARBON FOOTPRINT**  
                → Implement advanced water recycling (40% reduction potential)  
                → Optimize transport routes (20% reduction)  
                → Transition to renewable energy (60% reduction)  
                → Total reduction potential: 50-70%
                """)
            elif avg_daily_emissions > 50:
                st.warning("""
                **MODERATE CARBON FOOTPRINT**  
                → Improve treatment efficiency  
                → Consider solar-powered pumps  
                → Implement leak detection system  
                → Reduction potential: 30-50%
                """)
            else:
                st.success("""
                **LOW CARBON FOOTPRINT**  
                → Maintain current practices  
                → Continue monitoring and optimization  
                → Pursue sustainability certification
                """)
        else:
            st.warning("No data available for carbon footprint calculation")
    
    elif analysis_mode == "Geological Analysis":
        st.header("⛰️ Geological Analysis Module")
        
        if not data.empty and all(col in data.columns for col in ['Depth_m', 'Porosity_pct', 'Permeability_mD']):
            geo_data = data
        else:
            geo_data = generate_geological_data()
            st.info("Using synthetic geological data for demonstration purposes")
        
        st.subheader("Karoo Basin Geological Characteristics")
        
        st.markdown("#### 1. Mineral Content vs Depth")
        fig1, ax1 = plt.subplots(figsize=STANDARD_FIGSIZE)
        for mineral in ['Na_mg_L', 'Cl_mg_L', 'Ca_mg_L', 'Mg_mg_L', 'SO4_mg_L', 'TDS_mg_L']:
            ax1.scatter(geo_data[mineral], geo_data['Depth_m'], alpha=0.6, s=50, label=mineral)
        ax1.set_ylabel('Depth (m)')
        ax1.set_xlabel('Mineral Content (mg/L)')
        ax1.set_title('Mineral Content vs Depth')
        ax1.invert_yaxis()  
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        st.pyplot(fig1)
        
        st.markdown("""
        <div class="explanation-box">
        <div class="explanation-header">Interpretation:</div>
        Mineral content generally increases with depth due to higher pressure and 
        longer water-rock interaction times. Shale layers typically show higher mineral content due to their 
        fine-grained nature and higher surface area for mineral dissolution. This trend is important for 
        predicting water treatment requirements at different depths.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 2. Porosity vs Depth")
        fig2, ax2 = plt.subplots(figsize=STANDARD_FIGSIZE)
        z = np.polyfit(geo_data['Porosity_pct'], geo_data['Depth_m'], 1)
        p = np.poly1d(z)
        ax2.scatter(geo_data['Porosity_pct'], geo_data['Depth_m'], alpha=0.6, s=50)
        ax2.plot(geo_data['Porosity_pct'], p(geo_data['Porosity_pct']), "r--", alpha=0.8, label='Trend')

        ax2.set_ylabel('Depth (m)')
        ax2.set_xlabel('Porosity (%)')
        ax2.set_title('Porosity vs Depth')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        st.pyplot(fig2)

        st.markdown("""
        <div class="explanation-box">
        <div class="explanation-header">Interpretation:</div>
        Porosity decreases with depth due to compaction effects. Sandstone layers 
        maintain higher porosity compared to shale. The red trend line shows the general decrease in porosity 
        with increasing depth. This relationship is crucial for estimating water storage capacity and flow 
        characteristics in different geological formations.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 3. Permeability vs Porosity")
        fig3, ax3 = plt.subplots(figsize=STANDARD_FIGSIZE)

        ax3.scatter(geo_data['Porosity_pct'], geo_data['Permeability_mD'], alpha=0.6, s=50)

        x = np.linspace(geo_data['Porosity_pct'].min(), geo_data['Porosity_pct'].max(), 100)
        y_trend = 0.5 * x**2  
        ax3.plot(x, y_trend, 'r--', alpha=0.8, label='General Trend')

        ax3.set_xlabel('Porosity (%)')
        ax3.set_ylabel('Permeability (mD)')
        ax3.set_title('Permeability vs Porosity')
        ax3.set_yscale('log')  
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        st.pyplot(fig3)

        st.markdown("""
        <div class="explanation-box">
        <div class="explanation-header">Interpretation:</div>
        Permeability shows a strong positive correlation with porosity, following 
        a power-law relationship. Sandstone formations typically have both higher porosity and permeability 
        compared to shale. This relationship is critical for predicting fluid flow rates and designing 
        efficient extraction systems. The logarithmic scale highlights the exponential increase in permeability 
        with small increases in porosity.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 4. Porosity vs Lithology")
        fig4, ax4 = plt.subplots(figsize=STANDARD_FIGSIZE)

        if 'Lithology' in geo_data.columns:
            lithology_order = ['Sandstone', 'Siltstone', 'Limestone', 'Shale']
            plot_data = [geo_data[geo_data['Lithology'] == lith]['Porosity_pct'] for lith in lithology_order]

            box = ax4.boxplot(plot_data, labels=lithology_order, patch_artist=True)

            colors = ['goldenrod', 'lightblue', 'darkgray', 'gray']
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)

            ax4.set_xlabel('Lithology')
            ax4.set_ylabel('Porosity (%)')
            ax4.set_title('Porosity Distribution by Lithology')
            ax4.grid(True, alpha=0.3)
        else:
            st.warning("Lithology data not available for this analysis")

        st.pyplot(fig4)

        st.markdown("""
        <div class="explanation-box">
        <div class="explanation-header">Interpretation:</div>
        Sandstone typically shows the highest porosity due to its granular structure, 
        followed by siltstone and limestone. Shale has the lowest porosity due to its fine-grained, compact nature. 
        Understanding these lithological differences is essential for predicting water storage capacity and 
        designing appropriate extraction strategies for different geological formations in the Karoo Basin.
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Geological Summary Statistics")

        if not geo_data.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Depth", f"{geo_data['Depth_m'].mean():.0f} m")
            with col2:
                st.metric("Avg Porosity", f"{geo_data['Porosity_pct'].mean():.1f}%")
            with col3:
                st.metric("Avg Permeability", f"{geo_data['Permeability_mD'].mean():.0f} mD")
            with col4:
                st.metric("Avg Mineral Content", f"{geo_data[['Na_mg_L', 'Cl_mg_L', 'Ca_mg_L', 'Mg_mg_L', 'SO4_mg_L', 'TDS_mg_L']].mean().sum():.0f} mg/L")

            if 'Lithology' in geo_data.columns:
                st.write("**Lithology Distribution:**")
                lith_counts = geo_data['Lithology'].value_counts()
                st.dataframe(lith_counts)

        st.subheader("Implications for Hydraulic Fracturing Operations")

        st.markdown("""
        <div class="explanation-box">
        <div class="explanation-header">Key Geological Considerations:</div>

        1. <strong>Depth Trends</strong>: Deeper formations require higher pressure for fracturing but may yield 
           water with higher mineral content requiring more treatment.

        2. <strong>Porosity-Permeability Relationship</strong>: High porosity zones are preferred targets for 
           water extraction, but permeability determines flow rates and extraction efficiency.

        3. <strong>Lithological Controls</strong>: Sandstone layers are optimal for water storage and extraction, 
           while shale layers may require more intensive fracturing.

        4. <strong>Mineral Content</strong>: Deeper water sources will likely require more extensive treatment 
           for agricultural use, increasing operational costs.

        5. <strong>Formation Stability</strong>: Understanding geological properties helps in designing safe and 
           effective fracturing operations that minimize environmental impact.
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.header("📊 Raw Data")
    st.write(data)    

st.markdown("---")
st.header("🔄 Integrated Water Management Process")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""
    <div style="background-color: #e7f3ff; padding: 15px; border-radius: 10px; border-left: 5px solid #2196F3;">
        <strong>Fracking Operations</strong><br>
        → Oil & Gas Extraction<br>
        → Produced Water Generation<br>
        → Initial Water Treatment
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background-color: #e7f3ff; padding: 15px; border-radius: 10px; border-left: 5px solid #2196F3;">
        <strong>Water Treatment</strong><br>
        → Advanced Filtration<br>
        → Reverse Osmosis<br>
        → Chemical Treatment<br>
        → Quality Monitoring
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background-color: #e7f3ff; padding: 15px; border-radius: 10px; border-left: 5px solid #2196F3;">
        <strong>Treated Water Storage</strong><br>
        → Reservoir Management<br>
        → Quality Assurance<br>
        → Distribution Readiness<br>
        → Emergency Supply
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="background-color: #e7f3ff; padding: 15px; border-radius: 10px; border-left: 5px solid #2196F3;">
        <strong>Water Distribution</strong><br>
        → Pipeline Network<br>
        → Pump Stations<br>
        → Pressure Control<br>
        → Leak Monitoring
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div style="background-color: #e8f5e9; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;">
        <strong>Final Usage</strong><br>
        → Agricultural Irrigation<br>
        → Livestock Watering<br>
        → Environmental Release<br>
        → Community Supply
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="team-attribution">
    <h3 style="color: #ffc857; margin-bottom: 15px;">👩🏽‍🔬👨🏽‍🔬 Advance Chem Assignment Team</h3>
    <div style="display: flex; justify-content: center; gap: 15px; flex-wrap: wrap;">
        <div class="team-member"><strong>Kekeletso Ramahuma</strong><br>2302543</div>
        <div class="team-member"><strong>Gessica Cumbane</strong><br>1853353</div>
        <div class="team-member"><strong>Kefiloe Letsie</strong><br>2320312</div>
        <div class="team-member"><strong>Lebohang Mabe</strong><br>2326751</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
<b>AI-Powered Karoo Water Intelligence System</b><br/>
Real-time monitoring • Predictive analytics • Adaptive decision-making • Carbon accountability<br/>
Built with Streamlit • Integrating ML Emissions Calculator methodology
</div>
""", unsafe_allow_html=True)