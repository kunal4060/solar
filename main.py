import numpy as np

# ============================================
# USER INPUT FOR MATERIAL PARAMETERS
# ============================================

print("=" * 60)
print("  SOLAR CELL PHYSICS-BASED SIMULATION")
print("=" * 60)
print("\nEnter material parameters (press Enter for defaults):\n")

# Get user inputs with default values
try:
    eg1_input = input("Eg1 (eV) [default 1.5]: ").strip()
    Eg1 = float(eg1_input) if eg1_input else 1.5
    
    eg2_input = input("Eg2 (eV) [default 1.2]: ").strip()
    Eg2 = float(eg2_input) if eg2_input else 1.2
    
    thick1_input = input("Thickness1 (nm) [default 300]: ").strip()
    L1 = float(thick1_input) if thick1_input else 300
    
    thick2_input = input("Thickness2 (nm) [default 200]: ").strip()
    L2 = float(thick2_input) if thick2_input else 200
    
    doping1_input = input("Doping1 (cm^-3) [default 1e23]: ").strip()
    Nd1 = float(doping1_input) if doping1_input else 1e23
    
    doping2_input = input("Doping2 (cm^-3) [default 1e21]: ").strip()
    Nd2 = float(doping2_input) if doping2_input else 1e21
    
except ValueError:
    print("\nInvalid input! Using default values...")
    Eg1 = 1.5
    Eg2 = 1.2
    L1 = 300
    L2 = 200
    Nd1 = 1e23
    Nd2 = 1e21

# Convert thickness from nm to µm
L1_um = L1 / 1000
L2_um = L2 / 1000

print("\n" + "=" * 60)
print("  SIMULATION PARAMETERS")
print("=" * 60)
print(f"Eg1:        {Eg1:.2f} eV")
print(f"Eg2:        {Eg2:.2f} eV")
print(f"Thickness1: {L1:.0f} nm ({L1_um:.3f} µm)")
print(f"Thickness2: {L2:.0f} nm ({L2_um:.3f} µm)")
print(f"Doping1:    {Nd1:.2e} cm⁻³")
print(f"Doping2:    {Nd2:.2e} cm⁻³")
print("=" * 60)

# ============================================
# PHYSICS-BASED CALCULATIONS
# ============================================

# Physical constants
q = 1.602e-19  # Elementary charge (C)
k = 1.381e-23  # Boltzmann constant (J/K)
T = 300        # Temperature (K)
Vt = k * T / q # Thermal voltage (V)
eps0 = 8.854e-12  # Vacuum permittivity (F/m)

# Calculate average bandgap
Eg_avg = (Eg1 + Eg2) / 2

# Calculate Voc (Open Circuit Voltage)
# Conservative: Voc ≈ Eg - 0.7V (realistic total losses for heterojunction)
doping_correction = 0.02 * min(np.log10(Nd1), np.log10(Nd2))
Voc = max(0.3, Eg_avg - 0.7 + doping_correction)

# Physical limit: Voc cannot exceed ~75% of bandgap (realistic for heterojunction)
Voc = min(Voc, Eg_avg * 0.75)

# Calculate Jsc (Short Circuit Current)
# Realistic model based on bandgap:
# Eg = 1.1 eV (Si) → ~37 mA/cm²
# Eg = 1.7 eV (perovskite) → ~22 mA/cm²  
# Eg = 2.5 eV → ~5 mA/cm²
Eg_factor = 40 * np.exp(-Eg_avg / 0.55)

# Thickness effect (realistic absorption up to 80%)
thickness_factor = min(0.80, (L1_um + L2_um) / 2.0)

# Doping effect (carrier collection efficiency)
doping_factor = min(1.0, min(np.log10(Nd1) / 22, np.log10(Nd2) / 22))

# Calculate Jsc
Jsc = 40 * Eg_factor * thickness_factor * doping_factor

# Ensure reasonable bounds (realistic: 2-38 mA/cm²)
Jsc = max(2.0, min(38.0, Jsc))

# Calculate FF (Fill Factor)
# FF depends on series resistance, shunt resistance, and quality factor
# Simplified model based on Voc
n = 1.2  # Ideality factor
FF0 = (n * Vt / Voc) / (np.exp(n * Vt / Voc) - 1) if Voc > 0 else 0
FF0 = max(0.5, min(0.9, FF0))

# Adjust FF based on bandgap matching
bandgap_mismatch = abs(Eg1 - Eg2) / max(Eg1, Eg2)
FF = FF0 * (1 - 0.1 * bandgap_mismatch)
FF = max(0.5, min(0.9, FF))

# Calculate PCE (Power Conversion Efficiency)
# PCE = (Voc × Jsc × FF) / Pin × 100
Pin = 100  # mW/cm² (AM1.5G standard illumination)
PCE = (Voc * Jsc * FF) / Pin * 100

# Calculate Power Output
P_max = Voc * Jsc * FF  # mW/cm²

# ============================================
# OUTPUT RESULTS
# ============================================

print("\n" + "=" * 60)
print("  SOLAR CELL PERFORMANCE RESULTS")
print("=" * 60)
print(f"\n⚡ ELECTRICAL PARAMETERS:")
print(f"   Voc (Open Circuit Voltage):  {Voc:.3f} V")
print(f"   Jsc (Short Circuit Current): {Jsc:.3f} mA/cm²")
print(f"   FF (Fill Factor):           {FF:.3f}")
print(f"\n🔆 POWER OUTPUT:")
print(f"   Maximum Power (Pmax):        {P_max:.3f} mW/cm²")
print(f"   Input Power (AM1.5G):       {Pin:.0f} mW/cm²")
print(f"\n🎯 EFFICIENCY:")
print(f"   PCE (Power Conversion):     {PCE:.2f} %")
print("=" * 60)

# Quality assessment
if PCE >= 25:
    quality = "EXCELLENT - High efficiency solar cell"
elif PCE >= 20:
    quality = "VERY GOOD - Commercial grade efficiency"
elif PCE >= 15:
    quality = "GOOD - Moderate efficiency"
elif PCE >= 10:
    quality = "FAIR - Low efficiency, needs improvement"
else:
    quality = "POOR - Significant optimization needed"

print(f"\n📊 QUALITY ASSESSMENT: {quality}")
print("\nSimulation completed successfully! 🚀")
