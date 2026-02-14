# ============================================
# Solar Cell Optimization using
# Regression + Genetic Algorithm
# ============================================

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ============================================
# 1. LOAD DATASET
# ============================================

print("Loading dataset...")
data = pd.read_csv("solar_data.csv")

# Input features (same for all models)
X = data[["Thickness", "Doping", "Bandgap", "Defect"]]

# Check what target variables are available
print("Available columns in dataset:", list(data.columns))

# Primary target - Overall Efficiency
y_efficiency = data["Efficiency"]

# If additional parameters exist in dataset, use them
# Otherwise, we'll calculate them empirically
available_targets = [col for col in data.columns if col not in ["Thickness", "Doping", "Bandgap", "Defect"]]
print(f"Available target parameters: {available_targets}")

# ============================================
# 2. TRAIN MULTIPLE REGRESSION MODELS
# ============================================

# Split data for training (same split for all models)
X_train, X_test, y_train_eff, y_test_eff = train_test_split(
    X, y_efficiency, test_size=0.2, random_state=42
)

# Model for Overall Efficiency
model_efficiency = RandomForestRegressor(n_estimators=300, random_state=42)
model_efficiency.fit(X_train, y_train_eff)

# Dictionary to store all models
models = {
    'Efficiency': model_efficiency
}

# If dataset contains other parameters, train models for them too
other_parameters = ['PCE', 'VOC', 'Jsc', 'FF']
for param in other_parameters:
    if param in available_targets:
        y_param = data[param]
        _, _, y_train_param, y_test_param = train_test_split(
            X, y_param, test_size=0.2, random_state=42
        )
        model_param = RandomForestRegressor(n_estimators=300, random_state=42)
        model_param.fit(X_train, y_train_param)
        models[param] = model_param
        print(f"Trained model for {param}")
    else:
        print(f"Parameter {param} not found in dataset. Will calculate empirically.")

# ============================================
# 3. MODEL PERFORMANCE
# ============================================

print("\n========== MODEL PERFORMANCE ==========")

# Performance for Efficiency model
y_pred_eff = model_efficiency.predict(X_test)
r2_eff = r2_score(y_test_eff, y_pred_eff)
rmse_eff = np.sqrt(mean_squared_error(y_test_eff, y_pred_eff))
cv_score_eff = np.mean(cross_val_score(model_efficiency, X, y_efficiency, cv=5, scoring='r2'))

print(f"Efficiency Model:")
print(f"  R² Score: {r2_eff:.4f}")
print(f"  RMSE: {rmse_eff:.4f}")
print(f"  Cross-Validation R²: {cv_score_eff:.4f}")

# Performance for other parameters if available
for param_name, model_obj in models.items():
    if param_name != 'Efficiency':
        y_true = data[param_name]
        _, _, _, y_test_param = train_test_split(X, y_true, test_size=0.2, random_state=42)
        y_pred_param = model_obj.predict(X_test)
        r2_param = r2_score(y_test_param, y_pred_param)
        rmse_param = np.sqrt(mean_squared_error(y_test_param, y_pred_param))
        print(f"{param_name} Model:")
        print(f"  R² Score: {r2_param:.4f}")
        print(f"  RMSE: {rmse_param:.4f}")

print("========================================")

# ============================================
# 4. GET 4 VALUES FOR EACH PARAMETER FROM USER
# ============================================

import sys

def parse_values_from_args(param_name, default_values):
    """Parse values from command line arguments or use defaults"""
    # Look for command line argument like --thickness=0.5,0.6,0.7,0.8
    arg_prefix = f"--{param_name.lower()}="
    for arg in sys.argv[1:]:
        if arg.startswith(arg_prefix):
            try:
                values_str = arg[len(arg_prefix):]
                values = [float(x.strip()) for x in values_str.split(',')]
                if len(values) == 4:
                    return values
                else:
                    print(f"Warning: Expected 4 values for {param_name}, got {len(values)}. Using defaults.")
            except ValueError:
                print(f"Warning: Invalid format for {param_name}. Using defaults.")
    return default_values

print("\n===== ENTER YOUR 4 VALUES FOR EACH PARAMETER =====")
print("You can provide values via command line arguments or use defaults.")
print("Command line format: python script.py --thickness=0.5,0.6,0.7,0.8 --doping=1e16,2e16,3e16,4e16")
print("Or just press Enter to use default values.")

# Default values (same as before)
default_thickness = [0.6, 0.7, 0.8, 0.9]
default_doping = [5e16, 1e17, 3e16, 8e16]
default_bandgap = [1.45, 1.1, 1.55, 1.35]
default_defect = [1e14, 5e13, 2e14, 8e13]

# Parse from command line or use defaults
thickness_values = parse_values_from_args("thickness", default_thickness)
doping_values = parse_values_from_args("doping", default_doping)
bandgap_values = parse_values_from_args("bandgap", default_bandgap)
defect_values = parse_values_from_args("defect", default_defect)

print("\n===== CURRENT VALUES =====")
print(f"Thickness values (µm): {thickness_values}")
print(f"Doping values (cm^-3): {doping_values}")
print(f"Bandgap values (eV): {bandgap_values}")
print(f"Defect values (cm^-3): {defect_values}")
print("\nTo change values, run with: python solar_regression_model.py --thickness=a,b,c,d --doping=e,f,g,h etc.")

# (Values displayed above)

# Generate ALL possible combinations (4×4×4×4 = 256 combinations)
print("\n===== TESTING ALL PARAMETER COMBINATIONS =====")
print("Generating all possible combinations from the 4 values...")

all_combinations = []
for t in thickness_values:
    for d in doping_values:
        for b in bandgap_values:
            for df in defect_values:
                combination = [t, d, b, df]
                all_combinations.append(combination)

print(f"Total combinations to test: {len(all_combinations)}")

def predict_solar_parameters(material_combination):
    """Predict all solar cell parameters for a given material combination"""
    predictions = {}
    
    # Predict with trained models
    for param_name, model_obj in models.items():
        pred_value = model_obj.predict([material_combination])[0]
        predictions[param_name] = pred_value
    
    # For parameters not in dataset, calculate empirically
    if 'PCE' not in models:
        # PCE is the same as overall efficiency
        predictions['PCE'] = predictions.get('Efficiency', 0)
    
    if 'VOC' not in models:
        # Empirical calculation: VOC ≈ Bandgap - 0.3 (simplified)
        predictions['VOC'] = max(0, material_combination[2] - 0.3)
    
    if 'Jsc' not in models:
        # Empirical calculation: proportional to doping and thickness
        thickness_factor = material_combination[0] / 0.5  # normalized
        doping_factor = np.log10(material_combination[1]) / 16  # normalized
        predictions['Jsc'] = 25 * thickness_factor * doping_factor  # mA/cm²
    
    if 'FF' not in models:
        # Empirical calculation: FF typically 0.7-0.85
        defect_impact = 1 - (material_combination[3] / 1e15) * 0.2  # defect penalty
        predictions['FF'] = max(0.65, 0.82 * defect_impact)
    
    return predictions

# Evaluate all combinations
print("\nEvaluating all combinations for all solar cell parameters...")
results = []
for i, combo in enumerate(all_combinations):
    # Get all parameter predictions
    predictions = predict_solar_parameters(combo)
    efficiency = predictions['Efficiency']
    results.append((i+1, combo, predictions, efficiency))
    
    # Show progress every 50 combinations
    if (i+1) % 50 == 0 or i == len(all_combinations) - 1:
        print(f"Processed {i+1}/{len(all_combinations)} combinations...")

# Sort by efficiency (descending)
results.sort(key=lambda x: x[3], reverse=True)

print(f"\n{"="*60}")
print("🏆 TOP 4 SOLAR CELL COMBINATIONS - DETAILED ANALYSIS".center(60))
print(f"{"="*60}")
print(f"Evaluated {len(results)} total combinations")
print(f"Showing top 4 with complete performance parameters\n")
print("Parameters predicted:")
print("  • PCE  - Power Conversion Efficiency (%)")
print("  • VOC  - Open Circuit Voltage (V)")
print("  • Jsc  - Short Circuit Current (mA/cm²)")
print("  • FF   - Fill Factor")
print("  • η    - Overall Efficiency (%)")
print(f"{"="*60}\n")

# Display top 4 with all parameters in detailed format
for rank in range(4):
    idx, params, predictions, eff = results[rank]
    t, d, b, df = params
    
    print(f"\n{'='*60}")
    print(f"_RANK #{rank+1} - COMBINATION #{idx}_".center(60))
    print(f"{'='*60}")
    
    # Material Parameters
    print(f"\n🔬 MATERIAL PARAMETERS:")
    print(f"   Thickness:    {t:6.2f} µm")
    print(f"   Doping:       {d:6.2e} cm⁻³")
    print(f"   Bandgap:      {b:6.3f} eV")
    print(f"   Defects:      {df:6.2e} cm⁻³")
    
    # Solar Cell Performance Parameters
    print(f"\n⚡ SOLAR CELL PERFORMANCE:")
    print(f"   PCE  (Power Conversion Efficiency): {predictions['PCE']:6.3f} %")
    print(f"   VOC  (Open Circuit Voltage):        {predictions['VOC']:6.3f} V")
    print(f"   Jsc  (Short Circuit Current):       {predictions['Jsc']:6.3f} mA/cm²")
    print(f"   FF   (Fill Factor):                 {predictions['FF']:6.3f}")
    print(f"   η    (Overall Efficiency):          {predictions['Efficiency']:6.3f} %")
    
    # Derived metrics
    print(f"\n📊 DERIVED METRICS:")
    power_density = predictions['VOC'] * predictions['Jsc'] * predictions['FF'] / 1000  # W/cm²
    print(f"   Max Power Density:                  {power_density:6.4f} W/cm²")
    
    # Quality assessment
    print(f"\n⭐ QUALITY ASSESSMENT:")
    if predictions['Efficiency'] >= 18:
        quality = "EXCELLENT ⭐⭐⭐"
        color = "🟢"
    elif predictions['Efficiency'] >= 15:
        quality = "GOOD ⭐⭐"
        color = "🟡"
    else:
        quality = "FAIR ⭐"
        color = "🔴"
    print(f"   Rating: {color} {quality}")
    
    print(f"{'='*60}\n")

# Best overall summary
best_combo = results[0]
best_idx, best_params, best_predictions, best_eff = best_combo

print(f"\n{"="*60}")
print("🏆 CHAMPION COMBINATION SUMMARY".center(60))
print(f"{"="*60}")
print(f"Rank: #1 (Combination #{best_idx})")
print(f"Overall Efficiency: {best_eff:.3f}%")
print()
print("Key Performance Indicators:")
print(f"  PCE:  {best_predictions['PCE']:6.3f}%")
print(f"  VOC:  {best_predictions['VOC']:6.3f} V")
print(f"  Jsc:  {best_predictions['Jsc']:6.3f} mA/cm²")
print(f"  FF:   {best_predictions['FF']:6.3f}")
print(f"\nRecommended for manufacturing prototype!")
print(f"{"="*60}")

# Show which input values were used
print("\n===== INPUT VALUES USED =====")
print("From the 4 specified values for each parameter:")
print(f"Thickness: {thickness_values}")
print(f"Doping: {doping_values}")
print(f"Bandgap: {bandgap_values}")
print(f"Defect: {defect_values}")

print("\nThe top 4 combinations above are the best selections from these specific input values.")

# NOTE: Direct combination testing replaces GA for this specific task
# We test all 256 combinations from the 4 input values for each parameter
print("\nDirect combination testing completed. See results above.")
