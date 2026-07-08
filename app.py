import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

# Configure matplotlib to run headlessly
import matplotlib
matplotlib.use('Agg')

# Set page settings
st.set_page_config(
    page_title="Disease Risk Predictor AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Glassmorphism + Dark Mode Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Background color */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Glassmorphic Container */
    .glass-card {
        background: rgba(30, 41, 59, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px 0 rgba(0, 0, 0, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    /* Custom Risk Indicators */
    .risk-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 30px;
        font-weight: 800;
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
    }
    .risk-low {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        color: #FFFFFF;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .risk-moderate {
        background: linear-gradient(135deg, #D97706 0%, #F59E0B 100%);
        color: #FFFFFF;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .risk-high {
        background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
        color: #FFFFFF;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    
    /* Custom Metric Display */
    .metric-value {
        font-size: 2.3rem;
        font-weight: 800;
        color: #38BDF8;
        line-height: 1.1;
        margin-top: 8px;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    
    /* Recommendation Item */
    .rec-item {
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
        font-size: 0.95rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# Load Machine Learning components
@st.cache_resource
def load_ml_components():
    try:
        model = joblib.load("disease_prediction_model.pkl")
        scaler = joblib.load("disease_prediction_scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        return None, None

# Load the local environmental dataset
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("environmental_health_data.csv")
    except FileNotFoundError:
        return None

# Generate localized recommendations
def generate_recommendations(inputs, risk_level):
    recs = []
    
    # Base risk guidelines
    if risk_level == "High":
        recs.append("🔴 **High Risk Alert**: Environmental metrics suggest an elevated potential for acute respiratory/cardiovascular triggers. Susceptible populations should limit outdoor time.")
        recs.append("😷 **Wear N95/FFP2 Mask**: Use respirator grade masks to shield from particulate pollution if you must travel outdoors.")
        recs.append("🏠 **HEPA Air Filtration**: Keep windows and doors securely closed. Active air scrubbers are highly recommended.")
    elif risk_level == "Moderate":
        recs.append("🟡 **Moderate Risk Warning**: Air and climate parameters are in a cautionary range. Sensitive groups (asthma, COPD, cardiac issues) should monitor conditions closely.")
        recs.append("🌬️ **Limit Peak-Hour Outings**: Restrict strenuous activities outdoors during morning and evening rush hours.")
        recs.append("🚪 **Ventilation Planning**: Open windows only when outdoor air index improves, and run indoor recirculating filters.")
    else:
        recs.append("🟢 **Low Risk Conditions**: General environment is safe. Air quality meets baseline protective metrics.")
        recs.append("🌳 **Safe Outdoors**: Excellent time for outdoor workouts, ventilation, and general open-air activities.")

    # Variable specific triggers
    pm25 = inputs["PM2.5"][0]
    pm10 = inputs["PM10"][0]
    no2 = inputs["NO2"][0]
    o3 = inputs["O3"][0]
    temp = inputs["Temperature"][0]
    industrial_proximity = inputs["Industrial_Proximity"][0]
    population_density = inputs["Population_Density"][0]
    
    if pm25 > 35.0:
        recs.append("⚠️ **Elevated PM2.5 ({:.1f} μg/m³)**: Fine particulate concentrations can bypass lung filters. Avoid high-exertion sports outside.".format(pm25))
    if pm10 > 100.0:
        recs.append("⚠️ **Elevated PM10 ({:.1f} μg/m³)**: Coarse dust and pollen can trigger throat congestion and allergy issues.".format(pm10))
    if no2 > 50.0:
        recs.append("🚗 **High NO2 ({:.1f} ppb)**: Correlated with automotive emissions. Keep spacing away from high-traffic arteries and expressways.".format(no2))
    if o3 > 70.0:
        recs.append("☀️ **Elevated Ozone ({:.1f} ppb)**: Secondary pollutant peaked by solar heat. Keep cool indoors during sunny afternoons.".format(o3))
    if industrial_proximity == 1:
        recs.append("🏭 **Industrial Buffer Zone**: Operating near industrial sources increases exposure to chemicals. Use active carbon air filters in your ventilation system.")
    if population_density > 10000:
        recs.append("🏢 **Urban Core Density ({:,.0f}/km²)**: Density amplifies vehicle exhausts and heating pollution. Seek green urban parks to break exposures.".format(population_density))
    if temp > 38.0:
        recs.append("🌡️ **Extreme Heat Alert ({:.1f}°C)**: Extreme thermal loads tax the cardiovascular system. Remain hydrated and seek air conditioning.".format(temp))
    elif temp < 5.0:
        recs.append("❄️ **Cold Air Trigger ({:.1f}°C)**: Inhaling freezing air can trigger respiratory spasms. Cover nose and mouth with a scarf when outdoors.".format(temp))

    return recs

def main():
    # Load ML systems
    model, scaler = load_ml_components()
    df = load_dataset()

    # Header section
    st.markdown('<div class="dashboard-title">🌍 Disease Risk Prediction AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Analyzing environmental factors to predict cardiovascular and respiratory health risks</div>', unsafe_allow_html=True)

    if model is None or scaler is None:
        st.error("❌ Machine Learning artifacts not found! Run the model training script `train_model.py` first to generate files.")
        return

    # Define Application tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Interactive Risk Predictor", 
        "📊 Model Performance & Evaluation", 
        "📂 Batch Forecasting",
        "🔍 Training Data Explorer"
    ])

    # ------------------ TAB 1: INTERACTIVE RISK PREDICTOR ------------------
    with tab1:
        st.write("")
        col_inputs, col_results = st.columns([1, 1], gap="large")
        
        with col_inputs:
            st.markdown('<h3 style="margin-top:0;">Environmental & Demographic Parameters</h3>', unsafe_allow_html=True)
            
            st.write("##### Air Quality Indices")
            pm25 = st.slider("PM2.5 (μg/m³)", min_value=0.0, max_value=250.0, value=30.0, step=0.5, help="Particulate matter smaller than 2.5 microns (Fine)")
            pm10 = st.slider("PM10 (μg/m³)", min_value=0.0, max_value=400.0, value=50.0, step=1.0, help="Particulate matter smaller than 10 microns (Coarse)")
            no2 = st.slider("NO2 (ppb)", min_value=0.0, max_value=150.0, value=15.0, step=0.5, help="Nitrogen Dioxide (Vehicle emissions)")
            so2 = st.slider("SO2 (ppb)", min_value=0.0, max_value=100.0, value=10.0, step=0.5, help="Sulfur Dioxide (Industrial combustion)")
            o3 = st.slider("Ozone (O3 - ppb)", min_value=0.0, max_value=150.0, value=40.0, step=0.5, help="Ground-level Ozone (Photochemical smog)")

            st.write("##### Weather & Geography")
            temp = st.slider("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.5)
            humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
            
            st.write("##### Demographics & Location")
            ind_prox_str = st.selectbox("Proximity to Industrial Area", ["Far from industrial zone", "Close to industrial zone"])
            industrial_proximity = 1 if "Close" in ind_prox_str else 0
            pop_density = st.number_input("Population Density (people/sq km)", min_value=100, max_value=50000, value=1500, step=100)

            # Package current inputs into dataframe
            input_df = pd.DataFrame({
                "PM2.5": [pm25],
                "PM10": [pm10],
                "NO2": [no2],
                "SO2": [so2],
                "O3": [o3],
                "Temperature": [temp],
                "Humidity": [humidity],
                "Industrial_Proximity": [industrial_proximity],
                "Population_Density": [pop_density]
            })

        with col_results:
            st.markdown('<h3 style="margin-top:0;">AI Diagnostic Result</h3>', unsafe_allow_html=True)
            
            # Predict and calculate probabilities
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Build probability map
            classes = model.classes_
            proba_dict = dict(zip(classes, prediction_proba))
            risk_prob = proba_dict.get(prediction, 0.0) * 100
            
            # Match badge colors
            if prediction == "High":
                badge_style = "risk-high"
                color_code = "#EF4444"
            elif prediction == "Moderate":
                badge_style = "risk-moderate"
                color_code = "#F59E0B"
            else:
                badge_style = "risk-low"
                color_code = "#10B981"
                
            # HTML card output
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-top: 4px solid {color_code};">
                <div class="metric-label" style="font-size: 0.95rem; margin-bottom: 8px;">Estimated Disease Risk Level</div>
                <div style="margin: 12px 0;">
                    <span class="risk-badge {badge_style}">{prediction} RISK</span>
                </div>
                <div style="font-size: 0.9rem; color: #94A3B8; margin-top: 15px;">Model Diagnostic Confidence:</div>
                <div style="font-size: 2.3rem; font-weight: 800; color: {color_code}; margin-bottom: 10px;">{risk_prob:.1f}%</div>
                <div style="width: 100%; background-color: rgba(255,255,255,0.06); border-radius: 10px; height: 10px; overflow: hidden; margin: 10px 0;">
                    <div style="background-color: {color_code}; width: {risk_prob}%; height: 100%; border-radius: 10px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probabilities distribution chart
            st.markdown("##### Prediction Probability Breakdown")
            proba_df = pd.DataFrame({
                "Risk Category": classes,
                "Probability": prediction_proba * 100
            })
            
            prob_chart = alt.Chart(proba_df).mark_bar(cornerRadiusEnd=5).encode(
                x=alt.X('Probability:Q', title='Probability (%)', scale=alt.Scale(domain=[0, 100])),
                y=alt.Y('Risk Category:N', title='', sort=['Low', 'Moderate', 'High']),
                color=alt.Color('Risk Category:N', scale=alt.Scale(
                    domain=['Low', 'Moderate', 'High'],
                    range=['#10B981', '#F59E0B', '#EF4444']
                ), legend=None),
                tooltip=['Risk Category', 'Probability']
            ).properties(height=160)
            
            st.altair_chart(prob_chart, use_container_width=True)

            # Recommendations Card
            st.markdown("##### Dynamic Actionable Health Recommendations")
            recs = generate_recommendations(input_df, prediction)
            
            recs_html = ""
            for rec in recs:
                # Format bold text markdown inside recommendations
                formatted_rec = rec.replace("**", "<b>").replace("**", "</b>")
                recs_html += f'<div class="rec-item">{formatted_rec}</div>'
                
            st.markdown(f"""
            <div class="glass-card" style="padding: 20px; background: rgba(15, 23, 42, 0.4);">
                {recs_html}
            </div>
            """, unsafe_allow_html=True)

    # ------------------ TAB 2: MODEL PERFORMANCE & EVALUATION ------------------
    with tab2:
        st.write("")
        
        if df is None:
            st.warning("⚠️ Environmental health dataset missing! Cannot run live model evaluation.")
        else:
            # Recreate test split parameters
            X = df.drop(columns=["Disease_Risk", "Risk_Score"])
            y = df["Disease_Risk"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            
            # Compute classification reports
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Stat KPI display cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <div class="metric-label">Model Accuracy</div>
                    <div class="metric-value">{acc * 100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <div class="metric-label">Weighted Precision</div>
                    <div class="metric-value">{prec * 100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <div class="metric-label">Weighted Recall</div>
                    <div class="metric-value">{rec * 100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <div class="metric-label">Weighted F1-Score</div>
                    <div class="metric-value">{f1 * 100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

            col_left, col_right = st.columns([1, 1], gap="large")
            
            with col_left:
                st.markdown("### Confusion Matrix")
                st.write("Visualizes how often the Random Forest model confuses specific risk categories with others.")
                
                # Plot Confusion Matrix using Matplotlib + Seaborn
                cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
                fig, ax = plt.subplots(figsize=(6, 5))
                fig.patch.set_facecolor('#1E293B')
                ax.set_facecolor('#1E293B')
                
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues', 
                    xticklabels=model.classes_, 
                    yticklabels=model.classes_,
                    ax=ax,
                    cbar=False,
                    annot_kws={"size": 13, "weight": "bold"}
                )
                
                # Dark mode styling for plot labels
                ax.set_xlabel('Predicted Label', color='#94A3B8', fontsize=11, fontweight='bold', labelpad=10)
                ax.set_ylabel('True Label', color='#94A3B8', fontsize=11, fontweight='bold', labelpad=10)
                ax.tick_params(colors='#94A3B8', labelsize=10)
                for spine in ax.spines.values():
                    spine.set_color('#334155')
                plt.tight_layout()
                st.pyplot(fig)
                
                st.write("##### Detailed Classification Report")
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose().round(4)
                st.dataframe(report_df)

            with col_right:
                st.markdown("### Global Feature Importance")
                st.write("Relative importance coefficients of the Random Forest model. These weights show how much predictive power is derived from each feature.")
                
                # Importances
                importances = model.feature_importances_
                feat_imp_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=True)
                
                imp_chart = alt.Chart(feat_imp_df).mark_bar(
                    cornerRadiusTopRight=4,
                    cornerRadiusBottomRight=4,
                    color='#38BDF8'
                ).encode(
                    x=alt.X('Importance:Q', title='Relative Importance Value'),
                    y=alt.Y('Feature:N', sort='-x', title='Environmental Metric'),
                    tooltip=['Feature', 'Importance']
                ).properties(height=350)
                
                st.altair_chart(imp_chart, use_container_width=True)
                
                st.markdown("""
                <div class="glass-card" style="font-size: 0.9rem; margin-top: 15px; background: rgba(15, 23, 42, 0.4);">
                    💡 <b>Observation</b>: Air particulate indices (<b>PM2.5, PM10</b>) and 
                    <b>Industrial Proximity</b> carry the largest significance in forecasting disease risks. 
                    This aligns with epidemiological studies showing strong associations between particulate load 
                    and cardiovascular/respiratory admissions.
                </div>
                """, unsafe_allow_html=True)

    # ------------------ TAB 3: BATCH FORECASTING ------------------
    with tab3:
        st.write("")
        st.markdown("### Batch Forecasting Engine")
        st.write("Upload a CSV file containing multiple environmental recordings to predict disease risk levels in bulk.")
        
        # Define columns we want in inputs
        expected_cols = ["PM2.5", "PM10", "NO2", "SO2", "O3", "Temperature", "Humidity", "Industrial_Proximity", "Population_Density"]
        
        # Layout template download & uploader
        col_actions, col_bulk_run = st.columns([1, 2], gap="large")
        
        with col_actions:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.write("##### Step 1: Download Template")
            st.write("Ensure your batch files contain correct column names.")
            
            # Generate sample CSV Template
            sample_df = pd.DataFrame({
                "PM2.5": [12.4, 45.8, 110.2],
                "PM10": [22.1, 75.3, 190.5],
                "NO2": [10.2, 42.1, 88.0],
                "SO2": [4.5, 12.0, 32.1],
                "O3": [32.0, 56.4, 95.0],
                "Temperature": [21.5, 34.0, 2.5],
                "Humidity": [55.0, 72.0, 40.0],
                "Industrial_Proximity": [0, 1, 1],
                "Population_Density": [800, 3500, 18000]
            })
            csv_template = sample_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Template CSV",
                data=csv_template,
                file_name="disease_prediction_template.csv",
                mime="text/csv"
            )
            
            st.write("---")
            st.write("##### Step 2: Upload CSV")
            uploaded_file = st.file_uploader("Select input CSV", type=["csv"])
            st.markdown('</div>', unsafe_allow_html=True)

        with col_bulk_run:
            if uploaded_file is not None:
                try:
                    df_upload = pd.read_csv(uploaded_file)
                    st.success("✅ File successfully loaded!")
                    
                    # Columns check
                    missing_cols = [col for col in expected_cols if col not in df_upload.columns]
                    
                    if missing_cols:
                        st.error(f"❌ Verification failed. The CSV is missing the following required columns: {', '.join(missing_cols)}")
                    else:
                        st.markdown("### Run Inference")
                        # Run predict
                        scaled_upload = scaler.transform(df_upload[expected_cols])
                        bulk_preds = model.predict(scaled_upload)
                        bulk_probs = model.predict_proba(scaled_upload)
                        bulk_max_probs = np.max(bulk_probs, axis=1) * 100
                        
                        # Add results
                        df_result = df_upload.copy()
                        df_result["Predicted_Risk_Level"] = bulk_preds
                        df_result["Diagnostic_Confidence_Pct"] = np.round(bulk_max_probs, 2)
                        
                        # Display summary stats
                        st.write("##### Predictions Summary")
                        summary_counts = df_result["Predicted_Risk_Level"].value_counts()
                        
                        cols_summary = st.columns(3)
                        for idx, category in enumerate(["Low", "Moderate", "High"]):
                            count = summary_counts.get(category, 0)
                            pct = (count / len(df_result)) * 100
                            with cols_summary[idx]:
                                st.markdown(f"""
                                <div class="glass-card" style="text-align: center;">
                                    <div class="metric-label">{category} Risk Records</div>
                                    <div class="metric-value" style="color: {'#10B981' if category == 'Low' else '#F59E0B' if category == 'Moderate' else '#EF4444'}">{count} ({pct:.1f}%)</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                        # Distribution chart
                        summary_df = pd.DataFrame({
                            "Risk Level": summary_counts.index,
                            "Count": summary_counts.values
                        })
                        
                        donut_chart = alt.Chart(summary_df).mark_arc(innerRadius=50).encode(
                            theta=alt.Theta(field="Count", type="quantitative"),
                            color=alt.Color(field="Risk Level", type="nominal", scale=alt.Scale(
                                domain=['Low', 'Moderate', 'High'],
                                range=['#10B981', '#F59E0B', '#EF4444']
                            )),
                            tooltip=['Risk Level', 'Count']
                        ).properties(height=200)
                        
                        st.altair_chart(donut_chart, use_container_width=True)
                        
                        st.write("##### Prediction Results Preview")
                        st.dataframe(df_result)
                        
                        # Download result
                        csv_output = df_result.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Predictions CSV",
                            data=csv_output,
                            file_name="environmental_prediction_results.csv",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"Error parsing file: {e}")
            else:
                st.info("💡 Upload environmental measurements via the sidebar uploader to process batch estimations.")

    # ------------------ TAB 4: TRAINING DATA EXPLORER ------------------
    with tab4:
        st.write("")
        st.markdown("### Training Dataset Explorer")
        
        if df is None:
            st.warning("⚠️ No dataset file found. Please ensure environmental_health_data.csv exists.")
        else:
            st.write(f"The model was trained on **{len(df):,}** records with **{df.shape[1]}** features.")
            
            col_left, col_right = st.columns([1, 1], gap="large")
            
            with col_left:
                st.write("##### Numerical Distributions Grouped by Risk Level")
                st.write("Select a feature variable below to visualize its distribution across Low, Moderate, and High risk targets.")
                
                expected_cols = ["PM2.5", "PM10", "NO2", "SO2", "O3", "Temperature", "Humidity", "Population_Density"]
                selected_feat = st.selectbox("Feature to Analyze", expected_cols)
                
                # Distribution Area Plot
                dist_chart = alt.Chart(df).mark_area(
                    opacity=0.45,
                    interpolate='monotone'
                ).encode(
                    x=alt.X(f"{selected_feat}:Q", bin=alt.Bin(maxbins=40), title=selected_feat),
                    y=alt.Y('count()', stack=None, title='Record Count'),
                    color=alt.Color('Disease_Risk:N', scale=alt.Scale(
                        domain=['Low', 'Moderate', 'High'],
                        range=['#10B981', '#F59E0B', '#EF4444']
                    ), title='Ground-Truth Risk')
                ).properties(height=300)
                
                st.altair_chart(dist_chart, use_container_width=True)
                
                st.write("##### Descriptors Summary")
                st.dataframe(df[expected_cols].describe().round(2))

            with col_right:
                st.write("##### Feature Correlation Heatmap")
                st.write("Shows linear relationship coefficients between numeric inputs and Risk Score (continuous proxy).")
                
                num_features = expected_cols + ["Risk_Score"]
                corr_matrix = df[num_features].corr()
                
                fig, ax = plt.subplots(figsize=(7, 6))
                fig.patch.set_facecolor('#1E293B')
                ax.set_facecolor('#1E293B')
                
                sns.heatmap(
                    corr_matrix, 
                    annot=True, 
                    cmap='coolwarm', 
                    fmt='.2f', 
                    ax=ax, 
                    vmin=-1, 
                    vmax=1,
                    cbar=True,
                    annot_kws={"size": 8}
                )
                
                ax.tick_params(colors='#94A3B8', labelsize=9)
                for spine in ax.spines.values():
                    spine.set_color('#334155')
                plt.tight_layout()
                st.pyplot(fig)


if __name__ == "__main__":
    main()
