# app.py — AI-Powered Karoo Water Intelligence (Fracking, Karoo Basin)
# Author: Advance Chem Assignment Team
# Date: Aug 2025

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

# ------------------------------- Page & Theme --------------------------------
st.set_page_config(
    page_title="AI-Powered Karoo Water Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global brand styling (blue, gold, white) ---
st.markdown("""
<style>
/* App background gradient - works in both light and dark mode */
.stApp { 
    background: linear-gradient(180deg, #f9fbfd 0%, #eef6ff 100%);
}
@media (prefers-color-scheme: dark) {
    .stApp {
        background: linear-gradient(180deg, #0a1a2a 0%, #142b45 100%);
    }
}

/* Primary typography */
h1, h2, h3, h4 { 
    color:#0a3d62; 
    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
}
@media (prefers-color-scheme: dark) {
    h1, h2, h3, h4 {
        color: #ffc857;
    }
}

p, li, span, div { 
    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
}

/* Top banner card */
.karoo-banner {
  background: linear-gradient(135deg, #0a3d62 0%, #1e5aa7 60%);
  color: #fff; padding: 16px 20px; border-radius: 14px; margin-bottom: 16px;
  display: flex; align-items: center; gap: 16px; box-shadow: 0 6px 18px rgba(10,61,98,0.25);
}
.karoo-badge { 
    background:#ffc857; 
    color:#0a3d62; 
    border-radius: 10px; 
    padding: 6px 10px; 
    font-weight: 700; 
}

<<<<<<< HEAD
=======
/* Cover page styling */
.cover-container {
    background: linear-gradient(135deg, #0a3d62 0%, #1e5aa7 100%);
    color: white;
    padding: 40px;
    border-radius: 20px;
    margin: 20px 0;
    text-align: center;
}
.cover-title {
    font-size: 3.5em;
    font-weight: 700;
    margin-bottom: 20px;
    color: #ffc857;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}
.cover-subtitle {
    font-size: 1.8em;
    font-weight: 300;
    margin-bottom: 30px;
    color: #ffffff;
}
.cover-content {
    background: rgba(255, 255, 255, 0.1);
    padding: 30px;
    border-radius: 15px;
    margin: 20px 0;
    text-align: left;
    backdrop-filter: blur(10px);
}
.cover-stats {
    display: flex;
    justify-content: space-around;
    flex-wrap: wrap;
    margin: 30px 0;
}
.stat-card {
    background: rgba(255, 200, 87, 0.2);
    padding: 20px;
    border-radius: 12px;
    margin: 10px;
    min-width: 200px;
    text-align: center;
    border: 2px solid #ffc857;
}
.stat-number {
    font-size: 2.5em;
    font-weight: bold;
    color: #ffc857;
    margin-bottom: 5px;
}
.stat-label {
    font-size: 1.1em;
    color: #ffffff;
}

/* Image styling */
.karoo-image {
    border-radius: 15px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    margin: 20px 0;
    transition: transform 0.3s ease;
}
.karoo-image:hover {
    transform: scale(1.02);
}

>>>>>>> feature/cover-page
/* KPI metric cards */
[data-testid="stMetric"] {
  background: #ffffff; 
  border: 1px solid #d9e6f2; 
  border-radius: 12px; 
  padding: 10px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.04);
}
@media (prefers-color-scheme: dark) {
    [data-testid="stMetric"] {
        background: #1e3b5a;
        border: 1px solid #2a4b6a;
    }
    [data-testid="stMetric"] label, [data-testid="stMetric"] div {
        color: #ffffff !important;
    }
}

/* Tabs */
.stTabs [role="tab"] {
  background-color: #0a3d620f; 
  color:#0a3d62; 
  border-radius: 8px; 
  padding: 8px 14px;
  font-weight: 600; 
  border: 1px solid #d9e6f2; 
  margin-right:8px;
}
.stTabs [role="tab"][aria-selected="true"] {
  background-color: #ffc85722; 
  border-color:#ffc857; 
  color:#0a3d62;
}
@media (prefers-color-scheme: dark) {
    .stTabs [role="tab"] {
        background-color: #1e3b5a;
        color: #ffffff;
        border: 1px solid #2a4b6a;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #ffc85733;
        border-color: #ffc857;
        color: #ffc857;
    }
}

/* Section headers */
.section-header {
  background:#ffffff; 
  border-left:6px solid #ffc857; 
  padding:10px 14px; 
  border-radius:8px;
  margin: 8px 0 12px 0; 
  color:#0a3d62; 
  box-shadow: 0 3px 10px rgba(0,0,0,0.03);
}
@media (prefers-color-scheme: dark) {
    .section-header {
        background: #1e3b5a;
        color: #ffffff;
    }
}

/* Dataframes */
.block-container .stDataFrame { 
    border: 1px solid #dfeaf5; 
    border-radius: 10px; 
}
@media (prefers-color-scheme: dark) {
    .block-container .stDataFrame {
        border: 1px solid #2a4b6a;
    }
}

/* Sidebar - Blue text in white boxes */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0a3d62 0%, #1e5aa7 100%);
}
section[data-testid="stSidebar"] .stSelectbox, 
section[data-testid="stSidebar"] .stMultiSelect, 
section[data-testid="stSidebar"] .stDateInput,
section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] .stFileUploader {
  background:#ffffff; 
  padding:8px; 
  border-radius:10px;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stFileUploader label {
  color: #0a3d62 !important; 
  font-weight: 600;
}
section[data-testid="stSidebar"] * { 
    color:#ffffffdd; 
}
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 { 
    color:#ffc857; 
}

/* Footer */
.footer {
  text-align:center; 
  font-size:0.9em; 
  color:#607d8b; 
  padding:16px 0 6px 0;
}
@media (prefers-color-scheme: dark) {
    .footer {
        color: #a0b4c8;
    }
}

/* Team attribution at bottom */
.team-attribution {
  background: linear-gradient(135deg, #0a3d62 0%, #1e5aa7 100%);
  color: white; padding: 20px; border-radius: 12px; margin-top: 30px;
  text-align: center;
}
.team-member {
  background: rgba(255,200,87,0.2); padding: 10px; border-radius: 8px;
  margin: 5px; display: inline-block; color: #0a3d62 !important;
}

/* Explanation boxes */
.explanation-box {
  background: #f8f9fa; 
  border-left: 4px solid #0a3d62; 
  padding: 15px;
  border-radius: 8px; 
  margin: 10px 0; 
  font-size: 0.95em;
}
@media (prefers-color-scheme: dark) {
    .explanation-box {
        background: #1e3b5a;
        border-left: 4px solid #ffc857;
    }
}
.explanation-header {
  color: #0a3d62; 
  font-weight: bold; 
  margin-bottom: 8px; 
  font-size: 1.1em;
}
@media (prefers-color-scheme: dark) {
    .explanation-header {
        color: #ffc857;
    }
}

/* Matplotlib figure styling for dark mode */
@media (prefers-color-scheme: dark) {
    .stPlot {
        background-color: transparent !important;
    }
}

/* Chart text colors for dark mode */
@media (prefers-color-scheme: dark) {
    .stPlot text {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    .stPlot .xtick text, .stPlot .ytick text {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    .stPlot .title {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
}

/* Alert boxes for dark mode */
@media (prefers-color-scheme: dark) {
    .stAlert {
        background-color: #2a4b6a !important;
        border: 1px solid #3a5b7a !important;
    }
}
</style>
""", unsafe_allow_html=True)

<<<<<<< HEAD
# ------------------------------- Header --------------------------------------
st.markdown(
    """
    <div class="karoo-banner">
      <div>
        <div class="karoo-badge">Karoo Basin • Fracking Water Intelligence</div>
        <h2 style="margin:6px 0 0 0;">AI-Supervised Water Management & Carbon Accountability</h2>
        <div>Real-time monitoring • Predictive analytics • Adaptive decisions • ML Emissions baseline (16 months)</div>
      </div>
=======
# ---------------------------- COVER PAGE -------------------------------------
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
                <div class="stat-number">💧</div>
                <div class="stat-label">Water Scarcity</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">🤖</div>
                <div class="stat-label">AI Supervised</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">16M</div>
                <div class="stat-label">Month Analysis</div>
            </div>
        </div>
        
        <div class="cover-content">
            <h3 style="color: #ffc857; margin-bottom: 20px;">The Karoo Water Challenge</h3>
            
            <p style="font-size: 1.2em; line-height: 1.6; margin-bottom: 20px;">
            Hydraulic fracturing (fracking) is heavily water-dependent with each well requiring between 
            <strong>3–8 million gallons of water</strong> over its lifetime. This immense water usage poses a significant 
            challenge, particularly in the water-scarce Karoo Basin region.
            </p>
            
            <!-- Karoo Landscape Images -->
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap; margin: 30px 0;">
                <div style="text-align: center; margin: 10px;">
                    <img src="https://images.unsplash.com/photo-1570481662006-a3a1374699f8?w=400&h=300&fit=crop" 
                         class="karoo-image" 
                         alt="Karoo Landscape" 
                         style="width: 300px; height: 200px; object-fit: cover;">
                    <p style="color: #ffc857; margin-top: 8px;">Karoo Semi-Desert Landscape</p>
                </div>
                <div style="text-align: center; margin: 10px;">
                    <img src="https://images.unsplash.com/photo-1516426122078-c23e76319801?w=400&h=300&fit=crop" 
                         class="karoo-image" 
                         alt="Water Scarcity" 
                         style="width: 300px; height: 200px; object-fit: cover;">
                    <p style="color: #ffc857; margin-top: 8px;">Water Scarcity Challenges</p>
                </div>
                <div style="text-align: center; margin: 10px;">
                    <img src="https://images.unsplash.com/photo-1587351021759-3e566b3c7a7a?w=400&h=300&fit=crop" 
                         class="karoo-image" 
                         alt="Agricultural Impact" 
                         style="width: 300px; height: 200px; object-fit: cover;">
                    <p style="color: #ffc857; margin-top: 8px;">Agricultural Water Needs</p>
                </div>
            </div>
            
            <div style="background: rgba(255, 200, 87, 0.1); padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h4 style="color: #ffc857; margin-bottom: 15px;">🚨 Critical Challenges:</h4>
                <ul style="line-height: 1.8;">
                    <li><strong>Water Scarcity:</strong> Competing demands between fracking, agriculture, and communities</li>
                    <li><strong>Produced Water Management:</strong> Saline and brackish water requiring advanced treatment</li>
                    <li><strong>Environmental Compliance:</strong> Strict regulations for water disposal and reuse</li>
                    <li><strong>Food Security:</strong> Potential threat to agricultural production from water depletion</li>
                </ul>
            </div>
            
            <p style="font-size: 1.2em; line-height: 1.6; margin-bottom: 20px;">
            Reducing reliance on fresh water for fracturing is essential to minimizing effects on agriculture 
            and local communities. Managing produced and flowback water, which is often saline and brackish, 
            adds complexity as it must be treated or disposed of in compliance with environmental regulations.
            </p>
            
            <div style="background: rgba(76, 175, 80, 0.1); padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h4 style="color: #4CAF50; margin-bottom: 15px;">💡 Our AI-Powered Solution:</h4>
                <ul style="line-height: 1.8;">
                    <li><strong>Real-time Monitoring:</strong> Continuous water quality and quantity tracking</li>
                    <li><strong>Predictive Analytics:</strong> 16-month water demand forecasting</li>
                    <li><strong>Leak Detection:</strong> Advanced ML algorithms for early warning systems</li>
                    <li><strong>Carbon Accountability:</strong> ML Emissions Calculator integration</li>
                    <li><strong>Smart Treatment:</strong> Optimized water recycling and reuse strategies</li>
                </ul>
            </div>
            
            <div style="text-align: center; margin: 30px 0;">
                <div style="display: inline-block; background: #ffc857; color: #0a3d62; 
                          padding: 15px 30px; border-radius: 25px; font-weight: bold; 
                          font-size: 1.2em; cursor: pointer; transition: transform 0.3s ease;"
                     onmouseover="this.style.transform='scale(1.05)'" 
                     onmouseout="this.style.transform='scale(1)'"
                     onclick="window.scrollTo(0, document.body.scrollHeight)">
                    🚀 Explore the Solution
                </div>
            </div>
        </div>
>>>>>>> feature/cover-page
    </div>
    """, unsafe_allow_html=True)
    
    # Process Flow Diagram
    st.markdown("---")
    st.header("🔄 Integrated Water Management Process")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div style="background-color: #e7f3ff; padding: 15px; border-radius: 10px; border-left: 5px solid #2196F3; text-align: center;">
            <strong>🏭 Fracking Operations</strong><br><br>
            → Oil & Gas Extraction<br>
            → Produced Water Generation<br>
            → Initial Water Treatment
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #e7f3ff; padding: 15px; border-radius: 10px; border-left: 5px solid #2196F3; text-align: center;">
            <strong>💧 Water Treatment</strong><br><br>
            → Advanced Filtration<br>
            → Reverse Osmosis<br>
            → Chemical Treatment<br>
            → Quality Monitoring
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color: #e7f3ff; padding: 15px; border-radius: 10px; border-left: 5px solid #2196F3; text-align: center;">
            <strong>📦 Treated Water Storage</strong><br><br>
            → Reservoir Management<br>
            → Quality Assurance<br>
            → Distribution Readiness<br>
            → Emergency Supply
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background-color: #e7f3ff; padding: 15px; border-radius: 10px; border-left: 5px solid #2196F3; text-align: center;">
            <strong>📡 Water Distribution</strong><br><br>
            → Pipeline Network<br>
            → Pump Stations<br>
            → Pressure Control<br>
            → Leak Monitoring
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div style="background-color: #e8f5e9; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50; text-align: center;">
            <strong>🌾 Final Usage</strong><br><br>
            → Agricultural Irrigation<br>
            → Livestock Watering<br>
            → Environmental Release<br>
            → Community Supply
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin: 30px 0;">
        <em>Figure 1: Process flow diagram from production of produced water in fracking to its reuse for irrigation and livestock</em>
    </div>
    """, unsafe_allow_html=True)

<<<<<<< HEAD
=======
# ------------------------------- Header --------------------------------------
def show_main_header():
    st.markdown(
        """
        <div class="karoo-banner">
          <div>
            <div class="karoo-badge">Karoo Basin • Fracking Water Intelligence</div>
            <h2 style="margin:6px 0 0 0;">AI-Supervised Water Management & Carbon Accountability</h2>
            <div>Real-time monitoring • Predictive analytics • Adaptive decisions • ML Emissions baseline (16 months)</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

>>>>>>> feature/cover-page
# ---------------------------- Utility Functions ------------------------------
def ensure_datetime(df: pd.DataFrame, col="Timestamp"):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def safe_col(df: pd.DataFrame, name: str, default=np.nan):
    if name not in df.columns:
        df[name] = default
    return df

def calculate_salinity(ec_mS_cm):
    """Convert EC to approximate salinity (mg/L)"""
    return ec_mS_cm * 640.0

def calculate_sar(na_mgL, ca_mgL, mg_mgL):
    """Calculate Sodium Adsorption Ratio (proper meq/L conversion)"""
    na_meq = na_mgL / 23.0
    ca_meq = ca_mgL / 20.0
    mg_meq = mg_mgL / 12.2
    return na_meq / np.sqrt((ca_meq + mg_meq) / 2 + 1e-6)

def classify_water_quality(ec, sar, ph):
    """USDA water quality classification for irrigation suitability"""
    if pd.isna(ec) or pd.isna(sar) or pd.isna(ph):
        return "Unknown"
    
    # Salinity hazard classification
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
    
    # Sodicity hazard classification
    if sar < 10:
        sodicity_class = "Low"
    elif sar < 18:
        sodicity_class = "Medium"
    elif sar < 26:
        sodicity_class = "High"
    else:
        sodicity_class = "Very High"
    
    # pH suitability
    if 6.0 <= ph <= 8.5:
        ph_class = "Suitable"
    else:
        ph_class = "Unsuitable"
    
    return f"{salinity_class} Salinity, {sodicity_class} Sodicity, {ph_class} pH"

def detect_leaks_advanced(flow_data, pressure_data, timestamps, window_size=6):
    """Advanced leak detection using multiple ML methods"""
    results = []
    
    if len(flow_data) < 10:
        return pd.DataFrame()
    
    # Method 1: Statistical anomaly detection
    flow_zscore = np.abs((flow_data - np.mean(flow_data)) / np.std(flow_data))
    pressure_zscore = np.abs((pressure_data - np.mean(pressure_data)) / np.std(pressure_data))
    
    # Method 2: Isolation Forest for outliers
    X = np.column_stack([flow_data, pressure_data])
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outliers = iso_forest.fit_predict(X)
    
    # Method 3: DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(X)
    
    for i in range(len(flow_data)):
        leak_score = 0
        
        # Statistical outliers
        if flow_zscore[i] > 2.5 or pressure_zscore[i] > 2.5:
            leak_score += 0.3
        
        # Isolation Forest outliers
        if outliers[i] == -1:
            leak_score += 0.4
        
        # DBSCAN outliers
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
    """Calculate carbon footprint using ML Emissions Calculator methodology"""
    treatment_emissions = water_volume * treatment_kwh_m3 * grid_carbon_intensity
    transport_emissions = water_volume * transport_km * transport_intensity
    operations_emissions = water_volume * 0.12  # Base operational emissions
    
    total_emissions = treatment_emissions + transport_emissions + operations_emissions
    
    return {
        'total_emissions': total_emissions,
        'treatment_emissions': treatment_emissions,
        'transport_emissions': transport_emissions,
        'operations_emissions': operations_emissions
    }

def predict_water_demand_16months(data, months=16):
    """Predict water demand for up to 16 months using advanced time series forecasting"""
    if len(data) < 100:
        return None, None, None
    
    # Prepare data for forecasting
    df = data.copy()
    df = df.set_index('Timestamp')
    df = df.resample('D').mean().ffill()  # Resample to daily frequency
    
    # Feature engineering for long-term forecasting
    df['day_of_year'] = df.index.dayofyear
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    
    # Create future dates for prediction
    last_date = df.index.max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                               periods=months*30, freq='D')
    
    # Prepare features for prediction
    X = df[['day_of_year', 'month', 'quarter', 'year', 'Temperature_C']].dropna()
    y = df['Flow_m3_h'].loc[X.index]
    
    if len(X) < 50:
        return None, None, None
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Create future feature matrix
    future_features = pd.DataFrame({
        'day_of_year': future_dates.dayofyear,
        'month': future_dates.month,
        'quarter': future_dates.quarter,
        'year': future_dates.year,
        'Temperature_C': df['Temperature_C'].mean()  # Use average temperature
    })
    
    # Make predictions
    predictions = model.predict(future_features)
    
    return future_dates, predictions, model.score(X, y)

def generate_geological_data():
    """Generate synthetic geological data for demonstration"""
    np.random.seed(42)
    n_samples = 200
    
    # Synthetic depth data (meters)
    depth = np.random.uniform(500, 3500, n_samples)
    
    # Mineral content (mg/L) - increases with depth
    mineral_content = 50 + 0.1 * depth + np.random.normal(0, 15, n_samples)
    
<<<<<<< HEAD
    # Porosity (%) - decreases with depth due to compaction
=======
     # Porosity (%) - decreases with depth due to compaction
>>>>>>> feature/cover-page
    porosity = 25 - 0.004 * depth + np.random.normal(0, 3, n_samples)
    porosity = np.clip(porosity, 2, 40)
    
    # Permeability (mD) - related to porosity
    permeability = 0.5 * porosity**2 + np.random.normal(0, 50, n_samples)
    permeability = np.clip(permeability, 0.1, 2000)
    
    # Lithology types
    lithology_types = ['Sandstone', 'Shale', 'Siltstone', 'Limestone']
    lithology = np.random.choice(lithology_types, n_samples, p=[0.4, 0.3, 0.2, 0.1])
    
    return pd.DataFrame({
        'Depth_m': depth,
        'Mineral_Content_mgL': mineral_content,
        'Porosity_pct': porosity,
        'Permeability_mD': permeability,
        'Lithology': lithology
    })

# ------------------------------ Sidebar: Data --------------------------------
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
    
    prediction_horizon = st.selectbox("Prediction Horizon", 
                                    ["1 month", "3 months", "6 months", "12 months", "16 months"],
                                    index=4)

<<<<<<< HEAD
# ------------------------------- Load Data -----------------------------------
@st.cache_data
def load_and_preprocess_data(file):
    """Load and preprocess the water quality data"""
    if file is None:
        return pd.DataFrame()
    
    df = pd.read_csv(file)
    df = ensure_datetime(df, "Timestamp")
    
    # Required columns
    required_cols = ["Stream", "Timestamp", "Flow_m3_h", "Pressure_kPa", "EC_mS_cm", "pH", "Temperature_C"]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            return pd.DataFrame()
    
    # Optional columns with defaults
    optional_cols = ["Na_mg_L", "Cl_mg_L", "Ca_mg_L", "Mg_mg_L", "SO4_mg_L", "TDS_mg_L"]
    for col in optional_cols:
        df = safe_col(df, col, np.nan)
    
    # Calculate derived metrics
    if all(col in df.columns for col in ["Na_mg_L", "Ca_mg_L", "Mg_mg_L"]):
        df["SAR"] = calculate_sar(df["Na_mg_L"], df["Ca_mg_L"], df["Mg_mg_L"])
    
    if "EC_mS_cm" in df.columns:
        df["Salinity_mg_L"] = calculate_salinity(df["EC_mS_cm"])
    
    # Water quality classification
    df["Quality_Classification"] = df.apply(
        lambda row: classify_water_quality(
            row.get("EC_mS_cm", np.nan),
            row.get("SAR", np.nan),
            row.get("pH", np.nan)
        ), axis=1
=======
# Set default to Cover Page and show it first
if 'first_load' not in st.session_state:
    st.session_state.first_load = True

# Show cover page by default on first load
if st.session_state.first_load or analysis_mode == "Cover Page":
    show_cover_page()
    st.session_state.first_load = False
else:
    show_main_header()
    
    # ------------------------------- Load Data -----------------------------------
    @st.cache_data
    def load_and_preprocess_data(file):
        """Load and preprocess the water quality data"""
        if file is None:
            return pd.DataFrame()
        
        df = pd.read_csv(file)
        df = ensure_datetime(df, "Timestamp")
        
        # Required columns
        required_cols = ["Stream", "Timestamp", "Flow_m3_h", "Pressure_kPa", "EC_mS_cm", "pH", "Temperature_C"]
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Missing required column: {col}")
                return pd.DataFrame()
        
        # Optional columns with defaults
        optional_cols = ["Na_mg_L", "Cl_mg_L", "Ca_mg_L", "Mg_mg_L", "SO4_mg_L", "TDS_mg_L"]
        for col in optional_cols:
            df = safe_col(df, col, np.nan)
        
        # Calculate derived metrics
        if all(col in df.columns for col in ["Na_mg_L", "Ca_mg_L", "Mg_mg_L"]):
            df["SAR"] = calculate_sar(df["Na_mg_L"], df["Ca_mg_L"], df["Mg_mg_L"])
        
        if "EC_mS_cm" in df.columns:
            df["Salinity_mg_L"] = calculate_salinity(df["EC_mS_cm"])
        
        # Water quality classification
        df["Quality_Classification"] = df.apply(
            lambda row: classify_water_quality(
                row.get("EC_mS_cm", np.nan),
                row.get("SAR", np.nan),
                row.get("pH", np.nan)
            ), axis=1
        )
        
        return df

    # Load data
    if uploaded_file is not None:
        data = load_and_preprocess_data(uploaded_file)
    else:
        # Generate synthetic geological data for demonstration
        geological_data = generate_geological_data()
        st.info("👆 Please upload a CSV file to begin analysis. Showing synthetic geological data for demonstration.")
        data = pd.DataFrame()

    # Stream selection
    available_streams = data["Stream"].unique() if not data.empty else ["Synthetic_Data"]
    selected_streams = st.sidebar.multiselect(
        "Select Streams for Analysis",
        options=available_streams,
        default=available_streams[:min(2, len(available_streams))],
        help="Select water streams to analyze"
>>>>>>> feature/cover-page
    )

<<<<<<< HEAD
# Load data
if uploaded_file is not None:
    data = load_and_preprocess_data(uploaded_file)
else:
    # Generate synthetic geological data for demonstration
    geological_data = generate_geological_data()
    st.info("👆 Please upload a CSV file to begin analysis. Showing synthetic geological data for demonstration.")
    data = pd.DataFrame()

# Stream selection
available_streams = data["Stream"].unique() if not data.empty else ["Synthetic_Data"]
selected_streams = st.sidebar.multiselect(
    "Select Streams for Analysis",
    options=available_streams,
    default=available_streams[:min(2, len(available_streams))],
    help="Select water streams to analyze"
)

# Filter data
if not data.empty:
    filtered_data = data[data["Stream"].isin(selected_streams)].copy()
    
    # Apply time filter for analysis period
    if analysis_period:
        latest_time = filtered_data["Timestamp"].max()
        cutoff_time = latest_time - pd.DateOffset(months=analysis_period)
        filtered_data = filtered_data[filtered_data["Timestamp"] >= cutoff_time]
else:
    filtered_data = pd.DataFrame()

# ---------------------------- MODULE 1: WATER MANAGEMENT ---------------------------
if analysis_mode == "Water Management":
    st.header("💧 Water Management Module")
    
    # Real-time KPIs
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
    
    # Water Quality Overview
    st.subheader("Water Quality Assessment")
    if "Quality_Classification" in filtered_data.columns:
        quality_counts = filtered_data["Quality_Classification"].value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
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
    
    # ---------------------------- SALINITY ANALYSIS ----------------------------
    st.subheader("🧂 Salinity Analysis")

    if "EC_mS_cm" in filtered_data.columns and "Salinity_mg_L" in filtered_data.columns:
=======
    # Filter data
    if not data.empty:
        filtered_data = data[data["Stream"].isin(selected_streams)].copy()
        
        # Apply time filter for analysis period
        if analysis_period:
            latest_time = filtered_data["Timestamp"].max()
            cutoff_time = latest_time - pd.DateOffset(months=analysis_period)
            filtered_data = filtered_data[filtered_data["Timestamp"] >= cutoff_time]
    else:
        filtered_data = pd.DataFrame()

    # ---------------------------- MODULE 1: WATER MANAGEMENT ---------------------------
    if analysis_mode == "Water Management":
        st.header("💧 Water Management Module")
        
        # Real-time KPIs
>>>>>>> feature/cover-page
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_salinity = filtered_data["Salinity_mg_L"].mean()
            st.metric("Average Salinity", f"{avg_salinity:.0f} mg/L")
        
        with col2:
            max_salinity = filtered_data["Salinity_mg_L"].max()
            st.metric("Max Salinity", f"{max_salinity:.0f} mg/L")
        
        with col3:
            min_salinity = filtered_data["Salinity_mg_L"].min()
            st.metric("Min Salinity", f"{min_salinity:.0f} mg/L")
        
        with col4:
<<<<<<< HEAD
            # Salinity classification
            if avg_salinity < 500:
                salinity_class = "Fresh"
                color = "green"
            elif avg_salinity < 3000:
                salinity_class = "Brackish"
                color = "orange"
            else:
                salinity_class = "Saline"
                color = "red"
            st.metric("Water Type", salinity_class)
        
        # Salinity time series
        fig_sal, ax_sal = plt.subplots(figsize=(12, 6))
        for stream in selected_streams:
            stream_data = filtered_data[filtered_data["Stream"] == stream]
            ax_sal.plot(stream_data["Timestamp"], stream_data["Salinity_mg_L"], 
                       label=stream, linewidth=2)
        
        # Add salinity thresholds
        ax_sal.axhline(y=500, color='green', linestyle='--', alpha=0.7, label='Fresh Water (<500 mg/L)')
        ax_sal.axhline(y=3000, color='orange', linestyle='--', alpha=0.7, label='Brackish Water (<3000 mg/L)')
        ax_sal.axhline(y=10000, color='red', linestyle='--', alpha=0.7, label='Saline Water (>3000 mg/L)')
        
        ax_sal.set_title("Salinity Trends Over Time")
        ax_sal.set_ylabel("Salinity (mg/L)")
        ax_sal.set_xlabel("Time")
        ax_sal.legend()
        ax_sal.grid(True, alpha=0.3)
        st.pyplot(fig_sal)
        
        # Salinity distribution by stream
        st.markdown("#### Salinity Distribution by Stream")
        fig_sal_dist, ax_sal_dist = plt.subplots(figsize=(10, 6))
        
        salinity_data = []
        for stream in selected_streams:
            stream_data = filtered_data[filtered_data["Stream"] == stream]
            salinity_data.append(stream_data["Salinity_mg_L"].dropna())
        
        ax_sal_dist.boxplot(salinity_data, labels=selected_streams)
        ax_sal_dist.set_title("Salinity Distribution by Water Stream")
        ax_sal_dist.set_ylabel("Salinity (mg/L)")
        ax_sal_dist.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig_sal_dist)
        
        # Salinity Impact Assessment
        st.markdown("#### 🚰 Salinity Impact Assessment")
        
        salinity_impact = ""
        if avg_salinity < 500:
            salinity_impact = """
            <div style="background-color: #e8f5e9; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;">
                <strong>✅ Excellent Water Quality</strong><br>
                Salinity levels are within optimal range for agricultural use. This water is suitable for:
                <ul>
                <li>All crop irrigation without restrictions</li>
                <li>Livestock watering</li>
                <li>Direct reuse with minimal treatment</li>
                </ul>
            </div>
            """
        elif avg_salinity < 1500:
            salinity_impact = """
            <div style="background-color: #fff3e0; padding: 15px; border-radius: 10px; border-left: 5px solid #ff9800;">
                <strong>⚠️ Moderate Salinity - Caution Required</strong><br>
                Water may require some management for sensitive crops:
                <ul>
                <li>Monitor soil salinity regularly</li>
                <li>Implement leaching practices</li>
                <li>Consider blending with fresher water sources</li>
                <li>Avoid use on salt-sensitive crops</li>
                </ul>
            </div>
            """
        else:
            salinity_impact = """
            <div style="background-color: #ffebee; padding: 15px; border-radius: 10px; border-left: 5px solid #f44336;">
                <strong>🚨 High Salinity - Treatment Required</strong><br>
                Water requires treatment before agricultural use:
                <ul>
                <li>Reverse osmosis or desalination needed</li>
                <li>Not suitable for most crops without treatment</li>
                <li>Risk of soil salinization</li>
                <li>Consider alternative water sources</li>
                </ul>
            </div>
            """
        
        st.markdown(salinity_impact, unsafe_allow_html=True)
        
        # Treatment Recommendations
        st.markdown("#### 💡 Treatment Recommendations")
        
        if avg_salinity < 500:
            st.success("""
            **Minimal Treatment Required:**
            - Basic filtration sufficient
            - Chlorination for disinfection
            - pH adjustment if needed
            - Estimated treatment cost: Low
            """)
        elif avg_salinity < 1500:
            st.warning("""
            **Moderate Treatment Recommended:**
            - Enhanced filtration
            - Partial reverse osmosis
            - Blending with fresh water
            - Estimated treatment cost: Medium
            """)
        else:
            st.error("""
            **Advanced Treatment Necessary:**
            - Full reverse osmosis system
            - Electrodialysis or distillation
            - Significant energy requirements
            - Estimated treatment cost: High
            """)
        
        st.markdown("""
        <div class="explanation-box">
        <div class="explanation-header">Interpretation:</div>
        Salinity analysis is critical for agricultural water management in the Karoo Basin:
        
        <ul style="color: #0a3d62;">
        <li><strong>Fresh Water (<500 mg/L)</strong>: Ideal for irrigation with minimal impact on crops</li>
        <li><strong>Brackish Water (500-3000 mg/L)</strong>: Requires careful management and monitoring</li>
        <li><strong>Saline Water (>3000 mg/L)</strong>: Can damage crops and soil without treatment</li>
        </ul>
        
        Monitoring salinity trends helps in planning water treatment requirements and ensuring
        sustainable agricultural practices in water-scarce regions.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("Salinity data not available. Required columns: EC_mS_cm and derived Salinity_mg_L")
        
        # Time Series Monitoring
        st.subheader("Real-time Monitoring")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
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
        
        st.markdown("""
        <div class="explanation-box">
        <div class="explanation-header">Interpretation:</div>
        These time series plots provide a comprehensive view of water system performance over time:
        
        <ul>
        <li><strong>Flow Rate</strong>: Shows water usage patterns. Consistent drops may indicate leaks or system issues.</li>
        <li><strong>Electrical Conductivity</strong>: Measures salinity. Spikes may indicate contamination or changing water sources.</li>
        <li><strong>System Pressure</strong>: Critical for pipeline integrity. Drops may indicate leaks; spikes may indicate blockages.</li>
        <li><strong>pH Level</strong>: Important for water treatment and agricultural suitability. Stable pH is ideal.</li>
        </ul>
        
        Regular monitoring of these parameters helps in early detection of system issues and ensures
        consistent water quality for agricultural use.
        </div>
        """, unsafe_allow_html=True)

# ---------------------------- MODULE 2: INSPECTION ---------------------------
elif analysis_mode == "Inspection":
    st.header("🔍 Inspection Module")
    
    # Anomaly Detection
    st.subheader("Anomaly Detection & Inspection Scheduling")
    
    if not filtered_data.empty and all(col in filtered_data.columns for col in ["Flow_m3_h", "Pressure_kPa"]):
        # Use Isolation Forest for anomaly detection
        X = filtered_data[["Flow_m3_h", "Pressure_kPa", "EC_mS_cm", "pH"]].dropna()
        if len(X) > 10:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(X_scaled)
            
            # Count anomalies
            normal_count = np.sum(anomalies == 1)
            anomaly_count = np.sum(anomalies == -1)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Normal Readings", normal_count)
            with col2:
                st.metric("Anomalies Detected", anomaly_count)
            
            if anomaly_count > 0:
                st.warning(f"🚨 {anomaly_count} anomalies detected. Schedule inspections for these time periods.")
                
                # Show anomaly details
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
                
                # Inspection recommendations
                st.subheader("Inspection Recommendations")
                st.write("""
                - **Immediate inspection**: High-priority anomalies
                - **Scheduled maintenance**: Moderate priority issues  
                - **Preventive measures**: Address root causes
                - **Document findings**: Update inspection records
                """)
    
    # Water Quality Compliance
    st.subheader("Water Quality Compliance")
    if "Quality_Classification" in filtered_data.columns:
        compliance_data = filtered_data.groupby("Stream")["Quality_Classification"].apply(
            lambda x: (x.str.contains("Suitable")).mean() * 100
        ).round(2)
        
        st.dataframe(compliance_data.rename("Compliance Rate (%)"))
        
        # Compliance visualization
        fig, ax = plt.subplots(figsize=(10, 6))
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

# ---------------------------- MODULE 3: LEAK DETECTION ---------------------------
elif analysis_mode == "Leak Detection":
    st.header("🚨 Leak Detection Module")
    
    if not filtered_data.empty and all(col in filtered_data.columns for col in ["Flow_m3_h", "Pressure_kPa"]):
        # Advanced leak detection
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
            
            # Leak statistics
            critical_leaks = all_leaks[all_leaks["severity"] == "Critical"]
            warning_leaks = all_leaks[all_leaks["severity"] == "Warning"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Critical Leaks", len(critical_leaks))
            with col2:
                st.metric("Warning Leaks", len(warning_leaks))
=======
            total_volume = filtered_data["Flow_m3_h"].sum() if not filtered_data.empty else 0
            st.metric("Total Volume", f"{total_volume:,.0f} m³")
        
        # Water Quality Overview
        st.subheader("Water Quality Assessment")
        if "Quality_Classification" in filtered_data.columns:
            quality_counts = filtered_data["Quality_Classification"].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
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
        
        # ---------------------------- SALINITY ANALYSIS ----------------------------
        st.subheader("🧂 Salinity Analysis")

        if "EC_mS_cm" in filtered_data.columns and "Salinity_mg_L" in filtered_data.columns:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_salinity = filtered_data["Salinity_mg_L"].mean()
                st.metric("Average Salinity", f"{avg_salinity:.0f} mg/L")
            
            with col2:
                max_salinity = filtered_data["Salinity_mg_L"].max()
                st.metric("Max Salinity", f"{max_salinity:.0f} mg/L")
            
            with col3:
                min_salinity = filtered_data["Salinity_mg_L"].min()
                st.metric("Min Salinity", f"{min_salinity:.0f} mg/L")
            
            with col4:
                # Salinity classification
                if avg_salinity < 500:
                    salinity_class = "Fresh"
                    color = "green"
                elif avg_salinity < 3000:
                    salinity_class = "Brackish"
                    color = "orange"
                else:
                    salinity_class = "Saline"
                    color = "red"
                st.metric("Water Type", salinity_class)
            
            # Salinity time series
            fig_sal, ax_sal = plt.subplots(figsize=(12, 6))
            for stream in selected_streams:
                stream_data = filtered_data[filtered_data["Stream"] == stream]
                ax_sal.plot(stream_data["Timestamp"], stream_data["Salinity_mg_L"], 
                           label=stream, linewidth=2)
            
            # Add salinity thresholds
            ax_sal.axhline(y=500, color='green', linestyle='--', alpha=0.7, label='Fresh Water (<500 mg/L)')
            ax_sal.axhline(y=3000, color='orange', linestyle='--', alpha=0.7, label='Brackish Water (<3000 mg/L)')
            ax_sal.axhline(y=10000, color='red', linestyle='--', alpha=0.7, label='Saline Water (>3000 mg/L)')
            
            ax_sal.set_title("Salinity Trends Over Time")
            ax_sal.set_ylabel("Salinity (mg/L)")
            ax_sal.set_xlabel("Time")
            ax_sal.legend()
            ax_sal.grid(True, alpha=0.3)
            st.pyplot(fig_sal)
            
            # Salinity distribution by stream
            st.markdown("#### Salinity Distribution by Stream")
            fig_sal_dist, ax_sal_dist = plt.subplots(figsize=(10, 6))
            
            salinity_data = []
            for stream in selected_streams:
                stream_data = filtered_data[filtered_data["Stream"] == stream]
                salinity_data.append(stream_data["Salinity_mg_L"].dropna())
            
            ax_sal_dist.boxplot(salinity_data, labels=selected_streams)
            ax_sal_dist.set_title("Salinity Distribution by Water Stream")
            ax_sal_dist.set_ylabel("Salinity (mg/L)")
            ax_sal_dist.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig_sal_dist)
            
            # Salinity Impact Assessment
            st.markdown("#### 🚰 Salinity Impact Assessment")
            
            salinity_impact = ""
            if avg_salinity < 500:
                salinity_impact = """
                <div style="background-color: #e8f5e9; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;">
                    <strong>✅ Excellent Water Quality</strong><br>
                    Salinity levels are within optimal range for agricultural use. This water is suitable for:
                    <ul>
                    <li>All crop irrigation without restrictions</li>
                    <li>Livestock watering</li>
                    <li>Direct reuse with minimal treatment</li>
                    </ul>
                </div>
                """
            elif avg_salinity < 1500:
                salinity_impact = """
                <div style="background-color: #fff3e0; padding: 15px; border-radius: 10px; border-left: 5px solid #ff9800;">
                    <strong>⚠️ Moderate Salinity - Caution Required</strong><br>
                    Water may require some management for sensitive crops:
                    <ul>
                    <li>Monitor soil salinity regularly</li>
                    <li>Implement leaching practices</li>
                    <li>Consider blending with fresher water sources</li>
                    <li>Avoid use on salt-sensitive crops</li>
                    </ul>
                </div>
                """
            else:
                salinity_impact = """
                <div style="background-color: #ffebee; padding: 15px; border-radius: 10px; border-left: 5px solid #f44336;">
                    <strong>🚨 High Salinity - Treatment Required</strong><br>
                    Water requires treatment before agricultural use:
                    <ul>
                    <li>Reverse osmosis or desalination needed</li>
                    <li>Not suitable for most crops without treatment</li>
                    <li>Risk of soil salinization</li>
                    <li>Consider alternative water sources</li>
                    </ul>
                </div>
                """
            
            st.markdown(salinity_impact, unsafe_allow_html=True)
            
            # Treatment Recommendations
            st.markdown("#### 💡 Treatment Recommendations")
            
            if avg_salinity < 500:
                st.success("""
                **Minimal Treatment Required:**
                - Basic filtration sufficient
                - Chlorination for disinfection
                - pH adjustment if needed
                - Estimated treatment cost: Low
                """)
            elif avg_salinity < 1500:
                st.warning("""
                **Moderate Treatment Recommended:**
                - Enhanced filtration
                - Partial reverse osmosis
                - Blending with fresh water
                - Estimated treatment cost: Medium
                """)
            else:
                st.error("""
                **Advanced Treatment Necessary:**
                - Full reverse osmosis system
                - Electrodialysis or distillation
                - Significant energy requirements
                - Estimated treatment cost: High
                """)
            
            st.markdown("""
            <div class="explanation-box">
            <div class="explanation-header">Interpretation:</div>
            Salinity analysis is critical for agricultural water management in the Karoo Basin:
            
            <ul style="color: #0a3d62;">
            <li><strong>Fresh Water (<500 mg/L)</strong>: Ideal for irrigation with minimal impact on crops</li>
            <li><strong>Brackish Water (500-3000 mg/L)</strong>: Requires careful management and monitoring</li>
            <li><strong>Saline Water (>3000 mg/L)</strong>: Can damage crops and soil without treatment</li>
            </ul>
            
            Monitoring salinity trends helps in planning water treatment requirements and ensuring
            sustainable agricultural practices in water-scarce regions.
            </div>
            """, unsafe_allow_html=True)

        else:
            st.warning("Salinity data not available. Required columns: EC_mS_cm and derived Salinity_mg_L")
            
            # Time Series Monitoring
            st.subheader("Real-time Monitoring")
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
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
            
            st.markdown("""
            <div class="explanation-box">
            <div class="explanation-header">Interpretation:</div>
            These time series plots provide a comprehensive view of water system performance over time:
            
            <ul>
            <li><strong>Flow Rate</strong>: Shows water usage patterns. Consistent drops may indicate leaks or system issues.</li>
            <li><strong>Electrical Conductivity</strong>: Measures salinity. Spikes may indicate contamination or changing water sources.</li>
            <li><strong>System Pressure</strong>: Critical for pipeline integrity. Drops may indicate leaks; spikes may indicate blockages.</li>
            <li><strong>pH Level</strong>: Important for water treatment and agricultural suitability. Stable pH is ideal.</li>
            </ul>
            
            Regular monitoring of these parameters helps in early detection of system issues and ensures
            consistent water quality for agricultural use.
            </div>
            """, unsafe_allow_html=True)

    # ---------------------------- MODULE 2: INSPECTION ---------------------------
    elif analysis_mode == "Inspection":
        st.header("🔍 Inspection Module")
        
        # Anomaly Detection
        st.subheader("Anomaly Detection & Inspection Scheduling")
        
        if not filtered_data.empty and all(col in filtered_data.columns for col in ["Flow_m3_h", "Pressure_kPa"]):
            # Use Isolation Forest for anomaly detection
            X = filtered_data[["Flow_m3_h", "Pressure_kPa", "EC_mS_cm", "pH"]].dropna()
            if len(X) > 10:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                anomalies = iso_forest.fit_predict(X_scaled)
                
                # Count anomalies
                normal_count = np.sum(anomalies == 1)
                anomaly_count = np.sum(anomalies == -1)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Normal Readings", normal_count)
                with col2:
                    st.metric("Anomalies Detected", anomaly_count)
                
                if anomaly_count > 0:
                    st.warning(f"🚨 {anomaly_count} anomalies detected. Schedule inspections for these time periods.")
                    
                    # Show anomaly details
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
                    
                    # Inspection recommendations
                    st.subheader("Inspection Recommendations")
                    st.write("""
                    - **Immediate inspection**: High-priority anomalies
                    - **Scheduled maintenance**: Moderate priority issues  
                    - **Preventive measures**: Address root causes
                    - **Document findings**: Update inspection records
                    """)
        
        # Water Quality Compliance
        st.subheader("Water Quality Compliance")
        if "Quality_Classification" in filtered_data.columns:
            compliance_data = filtered_data.groupby("Stream")["Quality_Classification"].apply(
                lambda x: (x.str.contains("Suitable")).mean() * 100
            ).round(2)
            
            st.dataframe(compliance_data.rename("Compliance Rate (%)"))
            
            # Compliance visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            compliance_data.plot(kind='bar', ax=ax, color='green')
            ax.set_title("Water Quality Compliance by Stream")
            ax.set_ylabel("Compliance Rate (%)")
            ax.axhline(y=90, color='red', linestyle='--', label='Target Compliance (90%)')
            ax.legend()
            st.pyplot(fig)
>>>>>>> feature/cover-page
            
            st.markdown("""
            <div class="explanation-box">
            <div class="explanation-header">Interpretation:</div>
<<<<<<< HEAD
            The leak detection system uses multiple machine learning algorithms to identify potential leaks:
            
            <ul>
            <li><strong>Critical Leaks</strong>: High confidence detections requiring immediate attention</li>
            <li><strong>Warning Leaks</strong>: Potential leaks that should be monitored and investigated</li>
            </ul>
            
            Leaks are detected based on abnormal patterns in flow and pressure data that deviate from
            normal system operation. Early detection helps minimize water loss and infrastructure damage.
            </div>
            """, unsafe_allow_html=True)
            
            # Leak visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            for stream in selected_streams:
                stream_data = filtered_data[filtered_data["Stream"] == stream]
                ax.plot(stream_data["Timestamp"], stream_data["Flow_m3_h"], label=f"{stream} Flow", alpha=0.7)
            
            # Mark leak locations
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
            
            # Leak response recommendations
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
=======
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

    # ---------------------------- MODULE 3: LEAK DETECTION ---------------------------
    elif analysis_mode == "Leak Detection":
        st.header("🚨 Leak Detection Module")
        
        if not filtered_data.empty and all(col in filtered_data.columns for col in ["Flow_m3_h", "Pressure_kPa"]):
            # Advanced leak detection
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
                
                # Leak statistics
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
                
                # Leak visualization
                fig, ax = plt.subplots(figsize=(12, 6))
                for stream in selected_streams:
                    stream_data = filtered_data[filtered_data["Stream"] == stream]
                    ax.plot(stream_data["Timestamp"], stream_data["Flow_m3_h"], label=f"{stream} Flow", alpha=0.7)
                
                # Mark leak locations
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
                
                # Leak response recommendations
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

    # ---------------------------- MODULE 4: PREDICTIVE ANALYTICS ---------------------------
    elif analysis_mode == "Predictive Analytics":
        st.header("🤖 Predictive Analytics Module")
        
        # 16-month water demand prediction
        st.subheader("16-Month Water Demand Forecast")
        
        if len(filtered_data) > 100 and all(col in filtered_data.columns for col in ["Flow_m3_h", "Temperature_C"]):
            with st.spinner("Training predictive models for 16-month forecast..."):
                future_dates, predictions, r2_score = predict_water_demand_16months(filtered_data, months=16)
            
            if predictions is not None:
                st.success(f"✅ 16-month forecast model trained (R² = {r2_score:.3f})")
                
                # Plot predictions
                fig, ax = plt.subplots(figsize=(14, 6))
                
                # Historical data (last 6 months)
                historical_cutoff = filtered_data["Timestamp"].max() - pd.DateOffset(months=6)
                historical_data = filtered_data[filtered_data["Timestamp"] >= historical_cutoff]
                
                for stream in selected_streams:
                    stream_data = historical_data[historical_data["Stream"] == stream]
                    ax.plot(stream_data["Timestamp"], stream_data["Flow_m3_h"], 
                           label=f"{stream} Historical", linewidth=2, alpha=0.7)
                
                # Predictions
                ax.plot(future_dates, predictions, label="16-Month Forecast", 
                       linewidth=3, color='red', linestyle='-')
                
                ax.set_title("16-Month Water Demand Forecast")
                ax.set_ylabel("Flow (m³/h)")
                ax.set_xlabel("Time")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                st.markdown("""
                <div class="explanation-box">
                <div class="explanation-header">Interpretation:</div>
                This forecast predicts water demand for the next 16 months using machine learning algorithms
                that analyze historical patterns, seasonal variations, and temperature correlations.
                
                Key insights from this forecast:
                
                <ul>
                <li><strong>Seasonal Patterns</strong>: Expected variations in water demand based on historical patterns</li>
                <li><strong>Peak Demand Periods</strong>: Times when water requirements will be highest</li>
                <li><strong>Trend Analysis</strong>: Long-term increasing or decreasing demand patterns</li>
                <li><strong>Model Confidence</strong>: R² value indicates how well the model fits historical data</li>
                </ul>
                
                Use this forecast for capacity planning, resource allocation, and operational scheduling.
                </div>
                """, unsafe_allow_html=True)
                
                # Forecast statistics
                st.subheader("Forecast Insights")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Peak Demand", f"{np.max(predictions):.1f} m³/h")
                with col2:
                    st.metric("Average Demand", f"{np.mean(predictions):.1f} m³/h")
                with col3:
                    st.metric("Seasonal Variation", f"{(np.max(predictions) - np.min(predictions)):.1f} m³/h")
                
                # Operational recommendations
                st.subheader("Operational Recommendations")
                st.info("""
                **Based on 16-month forecast:**  
                → Plan water storage capacity for peak demand periods  
                → Schedule maintenance during low-demand periods  
                → Optimize treatment plant operations  
                → Coordinate with agricultural water needs  
                → Prepare for seasonal variations
>>>>>>> feature/cover-page
                """)
                
            else:
                st.warning("Insufficient data for 16-month forecasting")
        else:
<<<<<<< HEAD
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

# ---------------------------- MODULE 4: PREDICTIVE ANALYTICS ---------------------------
elif analysis_mode == "Predictive Analytics":
    st.header("🤖 Predictive Analytics Module")
    
    # 16-month water demand prediction
    st.subheader("16-Month Water Demand Forecast")
    
    if len(filtered_data) > 100 and all(col in filtered_data.columns for col in ["Flow_m3_h", "Temperature_C"]):
        with st.spinner("Training predictive models for 16-month forecast..."):
            future_dates, predictions, r2_score = predict_water_demand_16months(filtered_data, months=16)
        
        if predictions is not None:
            st.success(f"✅ 16-month forecast model trained (R² = {r2_score:.3f})")
            
            # Plot predictions
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Historical data (last 6 months)
            historical_cutoff = filtered_data["Timestamp"].max() - pd.DateOffset(months=6)
            historical_data = filtered_data[filtered_data["Timestamp"] >= historical_cutoff]
            
            for stream in selected_streams:
                stream_data = historical_data[historical_data["Stream"] == stream]
                ax.plot(stream_data["Timestamp"], stream_data["Flow_m3_h"], 
                       label=f"{stream} Historical", linewidth=2, alpha=0.7)
            
            # Predictions
            ax.plot(future_dates, predictions, label="16-Month Forecast", 
                   linewidth=3, color='red', linestyle='-')
            
            ax.set_title("16-Month Water Demand Forecast")
            ax.set_ylabel("Flow (m³/h)")
            ax.set_xlabel("Time")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.markdown("""
            <div class="explanation-box">
            <div class="explanation-header">Interpretation:</div>
            This forecast predicts water demand for the next 16 months using machine learning algorithms
            that analyze historical patterns, seasonal variations, and temperature correlations.
            
            Key insights from this forecast:
            
            <ul>
            <li><strong>Seasonal Patterns</strong>: Expected variations in water demand based on historical patterns</li>
            <li><strong>Peak Demand Periods</strong>: Times when water requirements will be highest</li>
            <li><strong>Trend Analysis</strong>: Long-term increasing or decreasing demand patterns</li>
            <li><strong>Model Confidence</strong>: R² value indicates how well the model fits historical data</li>
            </ul>
            
            Use this forecast for capacity planning, resource allocation, and operational scheduling.
            </div>
            """, unsafe_allow_html=True)
            
            # Forecast statistics
            st.subheader("Forecast Insights")
            col1, col2, col3 = st.columns(3)
=======
            st.warning("Need at least 100 records with flow and temperature data for forecasting")

    # ---------------------------- MODULE 5: CARBON FOOTPRINT ---------------------------
    elif analysis_mode == "Carbon Footprint":
        st.header("🌍 Carbon Footprint Module")
        
        if not filtered_data.empty:
            # Calculate total water volume for 16 months
            total_volume = filtered_data["Flow_m3_h"].sum()
            
            st.subheader("Emission Factors Configuration")
            col1, col2, col3 = st.columns(3)
            with col1:
                treatment_energy = st.slider("Treatment Energy (kWh/m³)", 0.1, 2.0, 0.8, 0.1)
            with col2:
                transport_dist = st.slider("Transport Distance (km)", 1, 100, 20, 1)
            with col3:
                energy_carbon = st.slider("Grid Carbon Intensity (kg CO₂/kWh)", 0.1, 1.0, 0.5, 0.05)
            
            # Calculate emissions
            emissions = calculate_carbon_footprint(total_volume, treatment_energy, transport_dist, 
                                                 energy_carbon, 0.15)
            
            # Display results
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
            
            st.markdown("""
            <div class="explanation-box">
            <div class="explanation-header">Interpretation:</div>
            Carbon footprint calculation based on the ML Emissions Calculator methodology:
            
            <ul>
            <li><strong>Total Water Volume</strong>: The amount of water processed over the 16-month period</li>
            <li><strong>Treatment Emissions</strong>: CO₂ from energy used in water treatment processes</li>
            <li><strong>Transport Emissions</strong>: CO₂ from transporting water through the distribution system</li>
            <li><strong>Operations Emissions</strong>: CO₂ from general operational activities</li>
            </ul>
            
            These calculations help quantify the environmental impact of water management operations
            and identify opportunities for emissions reduction.
            </div>
            """, unsafe_allow_html=True)
            
            # Emission breakdown (smaller pie chart)
            st.subheader("Emission Breakdown")
            fig, ax = plt.subplots(figsize=(6, 6))  # Smaller pie chart
            emission_types = ['Treatment', 'Transport', 'Operations']
            emission_values = [
                emissions['treatment_emissions'],
                emissions['transport_emissions'], 
                emissions['operations_emissions']
            ]
            
            ax.pie(emission_values, labels=emission_types, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            ax.set_title('Carbon Emission Breakdown')
            st.pyplot(fig)
            
            # Sustainability recommendations
            st.subheader("Sustainability Action Plan")
            
            if emissions['total_emissions'] > 100000:
                st.error("""
                **HIGH CARBON FOOTPRINT**  
                → Implement advanced water recycling (40% reduction potential)  
                → Optimize transport routes (20% reduction)  
                → Transition to renewable energy (60% reduction)  
                → Total reduction potential: 50-70%
                """)
            elif emissions['total_emissions'] > 50000:
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

    # ---------------------------- MODULE 6: GEOLOGICAL ANALYSIS ---------------------------
    elif analysis_mode == "Geological Analysis":
        st.header("⛰️ Geological Analysis Module")
        
        # Generate or load geological data
        if not data.empty and all(col in data.columns for col in ['Depth_m', 'Porosity_pct', 'Permeability_mD']):
            geo_data = data
        else:
            geo_data = generate_geological_data()
            st.info("Using synthetic geological data for demonstration purposes")
        
        st.subheader("Karoo Basin Geological Characteristics")
        
        # 1. Mineral Content vs Depth
        st.markdown("#### 1. Mineral Content vs Depth")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        if 'Lithology' in geo_data.columns:
            lithology_colors = {'Sandstone': 'goldenrod', 'Shale': 'gray', 'Siltstone': 'lightblue', 'Limestone': 'darkgray'}
            for lith in geo_data['Lithology'].unique():
                lith_data = geo_data[geo_data['Lithology'] == lith]
                ax1.scatter(lith_data['Mineral_Content_mgL'], lith_data['Depth_m'], 
                           label=lith, alpha=0.7, s=50, c=lithology_colors.get(lith, 'blue'))
        else:
            ax1.scatter(geo_data['Mineral_Content_mgL'], geo_data['Depth_m'], alpha=0.6, s=50)
        
        ax1.set_ylabel('Depth (m)')
        ax1.set_xlabel('Mineral Content (mg/L)')
        ax1.set_title('Mineral Content vs Depth')
        ax1.invert_yaxis()  # Depth increases downward
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
        
        # 2. Porosity vs Depth
        st.markdown("#### 2. Porosity vs Depth")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        if 'Lithology' in geo_data.columns:
            for lith in geo_data['Lithology'].unique():
                lith_data = geo_data[geo_data['Lithology'] == lith]
                ax2.scatter(lith_data['Porosity_pct'], lith_data['Depth_m'], 
                           label=lith, alpha=0.7, s=50, c=lithology_colors.get(lith, 'blue'))
        else:
            ax2.scatter(geo_data['Porosity_pct'], geo_data['Depth_m'], alpha=0.6, s=50)
        
        # Add trend line
        z = np.polyfit(geo_data['Porosity_pct'], geo_data['Depth_m'], 1)
        p = np.poly1d(z)
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
        
        # 3. Permeability vs Porosity
        st.markdown("#### 3. Permeability vs Porosity")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        
        if 'Lithology' in geo_data.columns:
            for lith in geo_data['Lithology'].unique():
                lith_data = geo_data[geo_data['Lithology'] == lith]
                ax3.scatter(lith_data['Porosity_pct'], lith_data['Permeability_mD'], 
                           label=lith, alpha=0.7, s=50, c=lithology_colors.get(lith, 'blue'))
        else:
            ax3.scatter(geo_data['Porosity_pct'], geo_data['Permeability_mD'], alpha=0.6, s=50)
        
        # Add power law trend (common in reservoir engineering)
        x = np.linspace(geo_data['Porosity_pct'].min(), geo_data['Porosity_pct'].max(), 100)
        y_trend = 0.5 * x**2  # Simple power law relationship
        ax3.plot(x, y_trend, 'r--', alpha=0.8, label='General Trend')
        
        ax3.set_xlabel('Porosity (%)')
        ax3.set_ylabel('Permeability (mD)')
        ax3.set_title('Permeability vs Porosity')
        ax3.set_yscale('log')  # Log scale for better visualization
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
        
        # 4. Porosity vs Lithology
        st.markdown("#### 4. Porosity vs Lithology")
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        
        if 'Lithology' in geo_data.columns:
            # Box plot for porosity by lithology
            lithology_order = ['Sandstone', 'Siltstone', 'Limestone', 'Shale']
            plot_data = [geo_data[geo_data['Lithology'] == lith]['Porosity_pct'] for lith in lithology_order]
            
            box = ax4.boxplot(plot_data, labels=lithology_order, patch_artist=True)
            
            # Color the boxes
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
        
        # Geological Summary Statistics
        st.subheader("Geological Summary Statistics")
        
        if not geo_data.empty:
            col1, col2, col3, col4 = st.columns(4)
>>>>>>> feature/cover-page
            with col1:
                st.metric("Peak Demand", f"{np.max(predictions):.1f} m³/h")
            with col2:
                st.metric("Average Demand", f"{np.mean(predictions):.1f} m³/h")
            with col3:
<<<<<<< HEAD
                st.metric("Seasonal Variation", f"{(np.max(predictions) - np.min(predictions)):.1f} m³/h")
            
            # Operational recommendations
            st.subheader("Operational Recommendations")
            st.info("""
            **Based on 16-month forecast:**  
            → Plan water storage capacity for peak demand periods  
            → Schedule maintenance during low-demand periods  
            → Optimize treatment plant operations  
            → Coordinate with agricultural water needs  
            → Prepare for seasonal variations
            """)
            
        else:
            st.warning("Insufficient data for 16-month forecasting")
    else:
        st.warning("Need at least 100 records with flow and temperature data for forecasting")

# ---------------------------- MODULE 5: CARBON FOOTPRINT ---------------------------
elif analysis_mode == "Carbon Footprint":
    st.header("🌍 Carbon Footprint Module")
    
    if not filtered_data.empty:
        # Calculate total water volume for 16 months
        total_volume = filtered_data["Flow_m3_h"].sum()
        
        st.subheader("Emission Factors Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            treatment_energy = st.slider("Treatment Energy (kWh/m³)", 0.1, 2.0, 0.8, 0.1)
        with col2:
            transport_dist = st.slider("Transport Distance (km)", 1, 100, 20, 1)
        with col3:
            energy_carbon = st.slider("Grid Carbon Intensity (kg CO₂/kWh)", 0.1, 1.0, 0.5, 0.05)
        
        # Calculate emissions
        emissions = calculate_carbon_footprint(total_volume, treatment_energy, transport_dist, 
                                             energy_carbon, 0.15)
        
        # Display results
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
        
        st.markdown("""
        <div class="explanation-box">
        <div class="explanation-header">Interpretation:</div>
        Carbon footprint calculation based on the ML Emissions Calculator methodology:
        
        <ul>
        <li><strong>Total Water Volume</strong>: The amount of water processed over the 16-month period</li>
        <li><strong>Treatment Emissions</strong>: CO₂ from energy used in water treatment processes</li>
        <li><strong>Transport Emissions</strong>: CO₂ from transporting water through the distribution system</li>
        <li><strong>Operations Emissions</strong>: CO₂ from general operational activities</li>
        </ul>
        
        These calculations help quantify the environmental impact of water management operations
        and identify opportunities for emissions reduction.
=======
                st.metric("Avg Permeability", f"{geo_data['Permeability_mD'].mean():.0f} mD")
            with col4:
                st.metric("Avg Mineral Content", f"{geo_data['Mineral_Content_mgL'].mean():.0f} mg/L")
            
            # Lithology distribution
            if 'Lithology' in geo_data.columns:
                st.write("**Lithology Distribution:**")
                lith_counts = geo_data['Lithology'].value_counts()
                st.dataframe(lith_counts)
        
        # Implications for Fracking Operations
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
>>>>>>> feature/cover-page
        </div>
        """, unsafe_allow_html=True)
        
        # Emission breakdown (smaller pie chart)
        st.subheader("Emission Breakdown")
        fig, ax = plt.subplots(figsize=(6, 6))  # Smaller pie chart
        emission_types = ['Treatment', 'Transport', 'Operations']
        emission_values = [
            emissions['treatment_emissions'],
            emissions['transport_emissions'], 
            emissions['operations_emissions']
        ]
        
        ax.pie(emission_values, labels=emission_types, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title('Carbon Emission Breakdown')
        st.pyplot(fig)
        
        # Sustainability recommendations
        st.subheader("Sustainability Action Plan")
        
        if emissions['total_emissions'] > 100000:
            st.error("""
            **HIGH CARBON FOOTPRINT**  
            → Implement advanced water recycling (40% reduction potential)  
            → Optimize transport routes (20% reduction)  
            → Transition to renewable energy (60% reduction)  
            → Total reduction potential: 50-70%
            """)
        elif emissions['total_emissions'] > 50000:
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

<<<<<<< HEAD
# ---------------------------- MODULE 6: GEOLOGICAL ANALYSIS ---------------------------
elif analysis_mode == "Geological Analysis":
    st.header("⛰️ Geological Analysis Module")
    
    # Generate or load geological data
    if not data.empty and all(col in data.columns for col in ['Depth_m', 'Porosity_pct', 'Permeability_mD']):
        geo_data = data
    else:
        geo_data = generate_geological_data()
        st.info("Using synthetic geological data for demonstration purposes")
    
    st.subheader("Karoo Basin Geological Characteristics")
    
    # 1. Mineral Content vs Depth
    st.markdown("#### 1. Mineral Content vs Depth")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    if 'Lithology' in geo_data.columns:
        lithology_colors = {'Sandstone': 'goldenrod', 'Shale': 'gray', 'Siltstone': 'lightblue', 'Limestone': 'darkgray'}
        for lith in geo_data['Lithology'].unique():
            lith_data = geo_data[geo_data['Lithology'] == lith]
            ax1.scatter(lith_data['Mineral_Content_mgL'], lith_data['Depth_m'], 
                       label=lith, alpha=0.7, s=50, c=lithology_colors.get(lith, 'blue'))
    else:
        ax1.scatter(geo_data['Mineral_Content_mgL'], geo_data['Depth_m'], alpha=0.6, s=50)
    
    ax1.set_ylabel('Depth (m)')
    ax1.set_xlabel('Mineral Content (mg/L)')
    ax1.set_title('Mineral Content vs Depth')
    ax1.invert_yaxis()  # Depth increases downward
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
    
    # 2. Porosity vs Depth
    st.markdown("#### 2. Porosity vs Depth")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    if 'Lithology' in geo_data.columns:
        for lith in geo_data['Lithology'].unique():
            lith_data = geo_data[geo_data['Lithology'] == lith]
            ax2.scatter(lith_data['Porosity_pct'], lith_data['Depth_m'], 
                       label=lith, alpha=0.7, s=50, c=lithology_colors.get(lith, 'blue'))
    else:
        ax2.scatter(geo_data['Porosity_pct'], geo_data['Depth_m'], alpha=0.6, s=50)
    
    # Add trend line
    z = np.polyfit(geo_data['Porosity_pct'], geo_data['Depth_m'], 1)
    p = np.poly1d(z)
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
    
    # 3. Permeability vs Porosity
    st.markdown("#### 3. Permeability vs Porosity")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    if 'Lithology' in geo_data.columns:
        for lith in geo_data['Lithology'].unique():
            lith_data = geo_data[geo_data['Lithology'] == lith]
            ax3.scatter(lith_data['Porosity_pct'], lith_data['Permeability_mD'], 
                       label=lith, alpha=0.7, s=50, c=lithology_colors.get(lith, 'blue'))
    else:
        ax3.scatter(geo_data['Porosity_pct'], geo_data['Permeability_mD'], alpha=0.6, s=50)
    
    # Add power law trend (common in reservoir engineering)
    x = np.linspace(geo_data['Porosity_pct'].min(), geo_data['Porosity_pct'].max(), 100)
    y_trend = 0.5 * x**2  # Simple power law relationship
    ax3.plot(x, y_trend, 'r--', alpha=0.8, label='General Trend')
    
    ax3.set_xlabel('Porosity (%)')
    ax3.set_ylabel('Permeability (mD)')
    ax3.set_title('Permeability vs Porosity')
    ax3.set_yscale('log')  # Log scale for better visualization
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
    
    # 4. Porosity vs Lithology
    st.markdown("#### 4. Porosity vs Lithology")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    if 'Lithology' in geo_data.columns:
        # Box plot for porosity by lithology
        lithology_order = ['Sandstone', 'Siltstone', 'Limestone', 'Shale']
        plot_data = [geo_data[geo_data['Lithology'] == lith]['Porosity_pct'] for lith in lithology_order]
        
        box = ax4.boxplot(plot_data, labels=lithology_order, patch_artist=True)
        
        # Color the boxes
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
    
    # Geological Summary Statistics
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
            st.metric("Avg Mineral Content", f"{geo_data['Mineral_Content_mgL'].mean():.0f} mg/L")
        
        # Lithology distribution
        if 'Lithology' in geo_data.columns:
            st.write("**Lithology Distribution:**")
            lith_counts = geo_data['Lithology'].value_counts()
            st.dataframe(lith_counts)
    
    # Implications for Fracking Operations
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

# ---------------------------- Process Flow Visualization ---------------------------
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

=======
>>>>>>> feature/cover-page
# ---------------------------- Team Attribution (ALWAYS AT BOTTOM) ---------------------------
st.markdown("""
<div class="team-attribution">
    <h3 style="color: #ffc857; margin-bottom: 15px;">👩🏽‍🔬👨🏽‍🔬 Advance Chem Assignment Team</h3>
    <div style="display: flex; justify-content: center; gap: 15px; flex-wrap: wrap;">
        <div class="team-member"><strong>Kekeletso Ramahuma</strong><br>2302543</div>
        <div class="team-member"><strong>Gessica Cumbane</strong><br>1853353</div>
        <div class="team-member"><strong>Kefiloe Letsie</strong><br>2320312</div>
        <div class="team-member"><strong>Lebogang Mabe</strong><br>2326751</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Final footer
st.markdown("""
<div class="footer">
<b>AI-Powered Karoo Water Intelligence System</b><br/>
Real-time monitoring • Predictive analytics • Adaptive decision-making • Carbon accountability<br/>
Built with Streamlit • Integrating ML Emissions Calculator methodology
</div>
""", unsafe_allow_html=True)