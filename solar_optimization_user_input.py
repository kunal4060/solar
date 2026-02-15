import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, cross_val_score
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
# PCE FORMULA
# ===============================

def calculate_pce(voc, jsc, ff):
    # PCE = Voc × Jsc × FF (dataset PCE values are in decimal form, so multiply by 100 for percentage)
    return (voc * jsc * ff)  # Result is in decimal, will be displayed as percentage

# ===============================
# USER INPUT FUNCTION
# ===============================

def get_user_inputs():
    """Get 4 values for each parameter from user"""
    print("\n=== ENTER 4 VALUES FOR EACH PARAMETER ===")
    print("Format: Enter 4 numbers separated by commas")
    
    # Get inputs for each parameter
    eg1_vals = list(map(float, input("Eg1 (eV): ").split(',')))
    eg2_vals = list(map(float, input("Eg2 (eV): ").split(',')))
    thick1_vals = list(map(float, input("Thickness1 (µm): ").split(',')))
    thick2_vals = list(map(float, input("Thickness2 (µm): ").split(',')))
    doping1_vals = list(map(float, input("Doping1 (cm⁻³): ").split(',')))
    doping2_vals = list(map(float, input("Doping2 (cm⁻³): ").split(',')))
    
    return eg1_vals, eg2_vals, thick1_vals, thick2_vals, doping1_vals, doping2_vals

# ===============================
# GENERATE COMBINATIONS
# ===============================

def generate_permutations(eg1_vals, eg2_vals, thick1_vals, thick2_vals, doping1_vals, doping2_vals):
    \"\"\"Generate all permutations of the input values (each parameter gets all 4 values in different arrangements)\"\"\"
    import itertools
    
    # Create all possible permutations where each parameter can take any of the 4 values
    permutations = []
    
    # Generate cartesian product (all possible combinations of parameters)
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
    
    return permutations

# ===============================
# PREDICT AND RANK COMBINATIONS
# ===============================

def predict_and_rank(combinations, scaler, model):
    """Predict performance for all combinations and rank by PCE"""
    results = []
    
    print(f"\nPredicting performance for {len(combinations)} combinations...")
    
    for i, combo in enumerate(combinations):
        # Transform the combination
        combo_array = np.array(combo).reshape(1, -1)
        combo_scaled = scaler.transform(combo_array)
        
        # Predict Voc, Jsc, FF
        voc, jsc, ff = model.predict(combo_scaled)[0]
        
        # Calculate PCE
        pce = calculate_pce(voc, jsc, ff)
        
        # Store results
        original_combo = [combo[0], combo[1], combo[2], combo[3], 10**combo[4], 10**combo[5]]
        results.append((original_combo, voc, jsc, ff, pce))
        
        # Show progress
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(combinations)} combinations...")
    
    # Sort by PCE (descending)
    results.sort(key=lambda x: x[4], reverse=True)
    
    return results

# ===============================
# MAIN EXECUTION
# ===============================

def main():
    # # Ask user for input method
    # choice = input("Use default values (d) or enter custom values (c)? [d/c]: ").lower().strip()
    # 
    # if choice == 'c':
    #     eg1_vals, eg2_vals, thick1_vals, thick2_vals, doping1_vals, doping2_vals = get_user_inputs()
    # else:
    #     # Default values based on dataset range
    #     eg1_vals = [1.2, 1.4, 1.6, 1.8]
    #     eg2_vals = [2.0, 2.2, 2.4, 2.6]
    #     thick1_vals = [0.4, 0.6, 0.8, 1.0]
    #     thick2_vals = [0.2, 0.4, 0.6, 0.8]
    #     doping1_vals = [1e15, 1e16, 1e17, 1e18]
    #     doping2_vals = [1e15, 1e16, 1e17, 1e18]
    
    # New diverse default values to create variation
    eg1_vals = [1.3, 1.5, 1.7, 1.9]
    eg2_vals = [2.1, 2.3, 2.5, 2.7]
    thick1_vals = [0.5, 0.7, 0.9, 1.1]
    thick2_vals = [0.3, 0.5, 0.7, 0.9]
    doping1_vals = [1e15, 1e16, 1e17, 1e18]  # Wider range
    doping2_vals = [1e15, 1e16, 1e17, 1e18]  # Wider range
    
    print("\\nUsing logical default values:")
    print(f"Eg1: {eg1_vals}")
    print(f"Eg2: {eg2_vals}")
    print(f"Thickness1: {thick1_vals}")
    print(f"Thickness2: {thick2_vals}")
    print(f"Doping1: {doping1_vals}")
    print(f"Doping2: {doping2_vals}")
    
    # Generate all combinations
    combinations = generate_combinations(eg1_vals, eg2_vals, thick1_vals, thick2_vals, doping1_vals, doping2_vals)
    total_combinations = len(combinations)
    print(f"\nGenerated {total_combinations} combinations from your inputs")
    
    # Predict and rank
    results = predict_and_rank(combinations, scaler, model)
    
    # Display top 4 results
    print("\n" + "="*60)
    print("🏆 TOP 4 OPTIMIZED COMBINATIONS".center(60))
    print("="*60)
    
    for rank in range(min(4, len(results))):
        combo, voc, jsc, ff, pce = results[rank]
        
        print("\n" + "="*60)
        print(f"RANK #{rank+1}".center(60))
        print("="*60)
        
        print("\n🔬 MATERIAL PARAMETERS:")
        print(f"Eg1:        {combo[0]:.3f} eV")
        print(f"Eg2:        {combo[1]:.3f} eV")
        print(f"Thickness1: {combo[2]:.3f} µm")
        print(f"Thickness2: {combo[3]:.3f} µm")
        print(f"Doping1:    {combo[4]:.2e} cm⁻³")
        print(f"Doping2:    {combo[5]:.2e} cm⁻³")
        
        print("\n⚡ PERFORMANCE:")
        print(f"Voc:  {voc:.3f} V")
        print(f"Jsc:  {jsc:.3f} mA/cm²")
        print(f"FF:   {ff:.3f}")
        print(f"PCE:  {pce:.3f} %")
    
    print("\nAnalysis completed! 🚀")

if __name__ == "__main__":
    main()