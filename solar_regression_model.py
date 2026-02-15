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
    Pin = 100  # mW/cm²
    return (voc * jsc * ff) / 1000 * 100  # Convert to percentage

# ===============================
# FITNESS FUNCTION
# ===============================

def fitness(individual):
    sample = np.array(individual).reshape(1,-1)
    sample = scaler.transform(sample)
    
    voc, jsc, ff = model.predict(sample)[0]
    pce = calculate_pce(voc, jsc, ff)
    
    return pce, voc, jsc, ff

# ===============================
# GENETIC ALGORITHM
# ===============================

POP_SIZE = 40
GENERATIONS = 40
MUTATION_RATE = 0.2

bounds = [
    (1.2, 1.8),   # Eg1
    (2.0, 2.8),   # Eg2
    (0.3, 1.0),   # Thick1
    (0.2, 0.8),   # Thick2
    (14, 18),     # log Doping1
    (14, 18)      # log Doping2
]

def create_individual():
    return [random.uniform(low, high) for low, high in bounds]

def crossover(p1, p2):
    point = random.randint(1, len(p1)-1)
    return p1[:point] + p2[point:]

def mutate(ind):
    for i in range(len(ind)):
        if random.random() < MUTATION_RATE:
            low, high = bounds[i]
            ind[i] = random.uniform(low, high)
    return ind

# Initial population
population = [create_individual() for _ in range(POP_SIZE)]

# Evolution
for gen in range(GENERATIONS):
    
    scored = []
    for ind in population:
        pce, voc, jsc, ff = fitness(ind)
        scored.append((ind, pce, voc, jsc, ff))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    population = [x[0] for x in scored]
    
    next_gen = population[:5]  # elitism
    
    while len(next_gen) < POP_SIZE:
        p1 = random.choice(population[:15])
        p2 = random.choice(population[:15])
        child = crossover(p1,p2)
        child = mutate(child)
        next_gen.append(child)
    
    population = next_gen
    
    print(f"Generation {gen+1} Best PCE: {scored[0][1]:.3f}%")

# ===============================
# FINAL TOP 4 RESULTS
# ===============================

final_results = []

for ind in population:
    pce, voc, jsc, ff = fitness(ind)
    final_results.append((ind, pce, voc, jsc, ff))

final_results.sort(key=lambda x: x[1], reverse=True)

print("\n" + "="*60)
print("🏆 TOP 4 OPTIMIZED TANDEM SOLAR CELLS".center(60))
print("="*60)

for rank in range(4):
    ind, pce, voc, jsc, ff = final_results[rank]
    
    print("\n" + "="*60)
    print(f"RANK #{rank+1}".center(60))
    print("="*60)
    
    print("\n🔬 MATERIAL PARAMETERS:")
    print(f"Eg1:        {ind[0]:.3f} eV")
    print(f"Eg2:        {ind[1]:.3f} eV")
    print(f"Thickness1: {ind[2]:.3f} µm")
    print(f"Thickness2: {ind[3]:.3f} µm")
    print(f"Doping1:    10^{ind[4]:.2f} cm⁻³")
    print(f"Doping2:    10^{ind[5]:.2f} cm⁻³")
    
    print("\n⚡ PERFORMANCE:")
    print(f"Voc:  {voc:.3f} V")
    print(f"Jsc:  {jsc:.3f} mA/cm²")
    print(f"FF:   {ff:.3f}")
    print(f"PCE:  {pce:.3f} %")

print("\nOptimization completed 🚀")
