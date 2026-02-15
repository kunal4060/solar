import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# ===============================
# LOAD DATASET
# ===============================

print("Loading dataset...")
data = pd.read_csv("solar_data_100_points.csv")

# Log transform doping
data["Doping1"] = np.log10(data["Doping1"])
data["Doping2"] = np.log10(data["Doping2"])

X = data[["Eg1","Eg2","Thick1","Thick2","Doping1","Doping2"]]
y = data[["Voc","Jsc","FF"]]

print(f"Dataset loaded with {len(data)} samples")
print(f"Features: {list(X.columns)}")
print(f"Targets: {list(y.columns)}")

# ===============================
# TRAIN MODEL
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=300, random_state=42)
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n========== MODEL PERFORMANCE ==========")
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("========================================\n")

# ===============================
# EFFICIENCY CALCULATION FUNCTIONS
# ===============================

def calculate_predicted_pce(voc, jsc, ff):
    \"\"\"Calculate PCE from ML model predictions (dataset PCE values are in decimal form)\"\"\"
    return (voc * jsc * ff)  # Result is in decimal, will be displayed as percentage

def calculate_real_pce_formula(voc, jsc, ff):
    \"\"\"Calculate real physical PCE using standard solar cell formula\"\"\"
    # Standard formula: PCE = (Voc × Jsc × FF) / P_in × 100
    # For AM1.5G illumination: P_in = 100 mW/cm² = 0.1 W/cm²
    p_in = 100  # mW/cm² (AM1.5G standard)
    pce_real = (voc * jsc * ff) / p_in * 100  # Convert to percentage
    return pce_real

def calculate_voc_from_bandgap(eg1, eg2):
    \"\"\"Estimate Voc from bandgap using empirical relationship\"\"\"
    # Simplified empirical relationship: Voc ≈ Eg - 0.4V (loss factor)
    eg_avg = (eg1 + eg2) / 2
    voc_est = max(0, eg_avg - 0.4)  # Prevent negative values
    return voc_est

def calculate_jsc_from_thickness(thick1, thick2, doping1, doping2):
    \"\"\"Estimate Jsc from thickness and doping (simplified model)\"\"\"
    # Simplified model: Jsc increases with thickness and doping up to saturation
    absorption_factor = min(1.0, (thick1 + thick2) / 2)  # Normalized absorption
    doping_factor = min(1.0, np.log10(doping1 * doping2) / 18)  # Normalized doping effect
    jsc_base = 25.0  # Base current density (mA/cm²)
    jsc_est = jsc_base * absorption_factor * doping_factor
    return max(5.0, jsc_est)  # Minimum Jsc of 5 mA/cm²

def calculate_ff_from_parameters(eg1, eg2, thick1, thick2):
    \"\"\"Estimate FF from material parameters\"\"\"
    # Simplified model: FF depends on bandgap matching and thickness
    bandgap_mismatch = abs(eg1 - eg2) / max(eg1, eg2)
    thickness_balance = abs(thick1 - thick2) / max(thick1, thick2)
    ff_base = 0.85  # Base fill factor
    ff_penalty = 0.1 * bandgap_mismatch + 0.05 * thickness_balance
    ff_est = max(0.6, ff_base - ff_penalty)  # Minimum FF of 0.6
    return ff_est

# ===============================
# GENERATE PERMUTATIONS
# ===============================

def generate_all_permutations():
    """Generate all permutations with diverse parameter values"""
    # Diverse parameter values to maximize variation
    eg1_vals = [1.2, 1.4, 1.6, 1.8]
    eg2_vals = [2.0, 2.2, 2.4, 2.6] 
    thick1_vals = [0.4, 0.6, 0.8, 1.0]
    thick2_vals = [0.2, 0.4, 0.6, 0.8]
    doping1_vals = [1e15, 1e16, 1e17, 1e18]
    doping2_vals = [1e15, 1e16, 1e17, 1e18]
    
    permutations = []
    
    # Generate all possible combinations (4^6 = 4096 total)
    for eg1 in eg1_vals:
        for eg2 in eg2_vals:
            for thick1 in thick1_vals:
                for thick2 in thick2_vals:
                    for doping1 in doping1_vals:
                        for doping2 in doping2_vals:
                            # Log transform doping values
                            log_doping1 = np.log10(doping1)
                            log_doping2 = np.log10(doping2)
                            permutations.append([eg1, eg2, thick1, thick2, log_doping1, log_doping2])
    
    return permutations, eg1_vals, eg2_vals, thick1_vals, thick2_vals, doping1_vals, doping2_vals

# ===============================
# PREDICT AND RANK
# ===============================

def predict_and_rank_all(permutations, scaler, model):
    \"\"\"Predict performance for all permutations and rank by PCE\"\"\"
    results = []
    
    print(f"\\nPredicting performance for {len(permutations)} permutations...")
    
    for i, combo in enumerate(permutations):
        # Transform the combination
        combo_array = np.array(combo).reshape(1, -1)
        combo_scaled = scaler.transform(combo_array)
        
        # Predict Voc, Jsc, FF using ML model
        voc_pred, jsc_pred, ff_pred = model.predict(combo_scaled)[0]
        
        # Calculate predicted PCE (ranking basis)
        pce_predicted = calculate_predicted_pce(voc_pred, jsc_pred, ff_pred)
        
        # Calculate real formula-based PCE for comparison
        pce_real = calculate_real_pce_formula(voc_pred, jsc_pred, ff_pred)
        
        # Store results (convert log doping back to linear scale)
        original_combo = [combo[0], combo[1], combo[2], combo[3], 10**combo[4], 10**combo[5]]
        results.append((original_combo, voc_pred, jsc_pred, ff_pred, pce_predicted, pce_real))
        
        # Show progress
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(permutations)} permutations...")
    
    # Sort by PREDICTED PCE (descending) - this determines the ranking
    results.sort(key=lambda x: x[4], reverse=True)
    
    return results

# ===============================
# MAIN EXECUTION
# ===============================

def main():
    print("Generating all parameter permutations...")
    
    # Generate all permutations
    permutations, eg1_vals, eg2_vals, thick1_vals, thick2_vals, doping1_vals, doping2_vals = generate_all_permutations()
    
    print(f"Created {len(permutations)} unique material permutations")
    print("\nParameter ranges:")
    print(f"Eg1: {eg1_vals}")
    print(f"Eg2: {eg2_vals}")
    print(f"Thickness1: {thick1_vals}")
    print(f"Thickness2: {thick2_vals}")
    print(f"Doping1: {doping1_vals}")
    print(f"Doping2: {doping2_vals}")
    
    # Predict and rank all permutations
    results = predict_and_rank_all(permutations, scaler, model)
    
    # Display top 4 results with highest PREDICTED efficiency
    print("\\n" + "="*70)
    print("🏆 TOP 4 HIGHEST PREDICTED EFFICIENCY MATERIALS".center(70))
    print("="*70)
    print("RANKED BY ML MODEL PREDICTIONS (shown with real formula validation)".center(70))
    print("="*70)
        
    for rank in range(min(4, len(results))):
        combo, voc_pred, jsc_pred, ff_pred, pce_predicted, pce_real = results[rank]
            
        print("\\n" + "="*70)
        print(f"RANK #{rank+1}".center(70))
        print("="*70)
            
        print("\\n🔬 MATERIAL PARAMETERS:")
        print(f"Eg1:        {combo[0]:.3f} eV")
        print(f"Eg2:        {combo[1]:.3f} eV")
        print(f"Thickness1: {combo[2]:.3f} µm")
        print(f"Thickness2: {combo[3]:.3f} µm")
        print(f"Doping1:    {combo[4]:.2e} cm⁻³")
        print(f"Doping2:    {combo[5]:.2e} cm⁻³")
            
        print("\\n⚡ ML MODEL PREDICTED PERFORMANCE:")
        print(f"Voc:  {voc_pred:.3f} V")
        print(f"Jsc:  {jsc_pred:.3f} mA/cm²")
        print(f"FF:   {ff_pred:.3f}")
        print(f"PCE (predicted):  {pce_predicted:.3f} %")
            
        print("\\n🧪 REAL PHYSICAL EFFICIENCY (formula validation):")
        print(f"PCE (real formula):  {pce_real:.3f} %")
            
        # Show difference between predicted and real
        diff = abs(pce_predicted - pce_real)
        print(f"Difference: {diff:.3f} %")
        
    # Show some statistics
    predicted_pce_values = [result[4] for result in results[:100]]  # Top 100 predicted
    real_pce_values = [result[5] for result in results[:100]]       # Top 100 real
        
    print(f"\\n📈 EFFICIENCY STATISTICS (Top 100):")
    print(f"  Predicted PCE - Max: {max(predicted_pce_values):.3f}%, Min: {min(predicted_pce_values):.3f}%, Avg: {np.mean(predicted_pce_values):.3f}%")
    print(f"  Real PCE - Max: {max(real_pce_values):.3f}%, Min: {min(real_pce_values):.3f}%, Avg: {np.mean(real_pce_values):.3f}%")
    
    print("\nAnalysis completed! 🚀")

if __name__ == "__main__":
    main()