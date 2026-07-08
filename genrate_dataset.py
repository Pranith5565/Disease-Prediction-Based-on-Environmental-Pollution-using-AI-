import pandas as pd
import numpy as np

def generate_synthetic_data(num_samples=5000):
    np.random.seed(42)
    
    # 1. Generate Input Variables (Features)
    # PM2.5 (Particulate Matter, usually 0 to 250+) 
    # Let's assume a normal distribution centered around 30, with some high pollution days
    pm25 = np.random.gamma(shape=2.5, scale=15, size=num_samples)
    
    # PM10 (Usually higher than PM2.5)
    pm10 = pm25 * np.random.uniform(1.2, 2.5, size=num_samples)
    
    # NO2 (Nitrogen Dioxide)
    no2 = np.random.gamma(shape=2.0, scale=15, size=num_samples)
    
    # SO2 (Sulfur Dioxide)
    so2 = np.random.gamma(shape=1.5, scale=10, size=num_samples)
    
    # Ozone (O3)
    o3 = np.random.normal(loc=40, scale=20, size=num_samples)
    o3 = np.clip(o3, 5, 150)
    
    # Temperature (Celsius, say 5 to 45 degrees)
    temperature = np.random.normal(loc=25, scale=10, size=num_samples)
    
    # Humidity (%)
    humidity = np.random.normal(loc=60, scale=15, size=num_samples)
    humidity = np.clip(humidity, 10, 100)
    
    # Proximity to Industrial Area (0: Far, 1: Close)
    industrial_proximity = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])

    # Population Density (People per sq km)
    population_density = np.random.uniform(500, 25000, size=num_samples)
    
    # Determine the "True Risk Score" based on combinations
    # Higher pollution -> Higher Risk Score
    # We will formulate it so Asthma / Cardiovascular disease has a strong correlation with PM2.5 and NO2
    risk_score = (pm25 * 0.4) + (pm10 * 0.1) + (no2 * 0.3) + (so2 * 0.1) + (o3 * 0.1)
    
    # Add modifiers for proximity and temp
    risk_score += industrial_proximity * 20
    # Extreme temperatures increase risk slightly
    risk_score += np.abs(temperature - 25) * 0.5 

    # Normalize risk score conceptually and map to categories
    # 0 -> 40: Low Risk
    # 40 -> 80: Moderate Risk
    # 80+: High Risk
    
    disease_risk = []
    for score in risk_score:
        if score < 50:
            disease_risk.append("Low")
        elif score < 90:
            disease_risk.append("Moderate")
        else:
            disease_risk.append("High")

    # Create DataFrame
    df = pd.DataFrame({
        "PM2.5": np.round(pm25, 2),
        "PM10": np.round(pm10, 2),
        "NO2": np.round(no2, 2),
        "SO2": np.round(so2, 2),
        "O3": np.round(o3, 2),
        "Temperature": np.round(temperature, 2),
        "Humidity": np.round(humidity, 2),
        "Industrial_Proximity": industrial_proximity,
        "Population_Density": np.round(population_density, 2),
        "Risk_Score": np.round(risk_score, 2),
        "Disease_Risk": disease_risk
    })
    
    return df

if __name__ == "__main__":
    print("Generating synthetic dataset...")
    df = generate_synthetic_data(num_samples=5000)
    df.to_csv("environmental_health_data.csv", index=False)
    print(f"Dataset generated with {len(df)} records. Saved to 'environmental_health_data.csv'.")
    print(df["Disease_Risk"].value_counts())
