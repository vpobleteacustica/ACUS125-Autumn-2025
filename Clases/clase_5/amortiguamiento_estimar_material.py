import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# -------------------------
# Parámetros del sistema
# -------------------------
m = 1.0  # masa (kg)
k = 1000.0  # rigidez (N/m)
zeta_real = 0.03  # factor de amortiguamiento real (entre 0 y 1)

x0 = 1.0  # desplazamiento inicial (m)
v0 = 0.0  # velocidad inicial (m/s)

t = np.linspace(0, 10, 1000)

# -------------------------
# Simulación vibración amortiguada
# -------------------------
w0 = np.sqrt(k/m)
wd = w0 * np.sqrt(1 - zeta_real**2)
c = np.sqrt(x0**2 + ((v0 + zeta_real*w0*x0) / wd)**2)
phi = np.arctan((x0 * wd) / (v0 + zeta_real * w0 * x0))

x = c * np.exp(-zeta_real * w0 * t) * np.cos(wd * t - phi)

# -------------------------
# Picos y decremento logarítmico
# -------------------------
peaks, _ = find_peaks(x)
peak_times = t[peaks]
peak_values = x[peaks]

if len(peak_values) < 2:
    raise ValueError("No se detectaron suficientes picos para calcular el decremento logarítmico.")

log_decrements = np.log(peak_values[:-1] / peak_values[1:])
log_mean = np.mean(log_decrements)
zeta_est = log_mean / np.sqrt(4 * np.pi**2 + log_mean**2)

# -------------------------
# Mostrar resultados
# -------------------------
print(f"Decremento logarítmico promedio (δ): {log_mean:.4f}")
print(f"Amortiguamiento estimado (ζ): {zeta_est:.4f}")

# -------------------------
# Gráfico señal y picos
# -------------------------
plt.figure(figsize=(10, 4))
plt.plot(t, x, label='x(t)')
plt.plot(peak_times, peak_values, 'ro', label='Picos')
plt.title('Vibración amortiguada y picos detectados')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# Diccionario de materiales
# -------------------------
materiales = {
    "Acero": (0.001, 0.01),
    "Aluminio": (0.002, 0.02),
    "Plástico": (0.02, 0.08),
    "Madera": (0.03, 0.10),
    "Goma (caucho)": (0.10, 0.30),
    "Espuma": (0.30, 0.50),
}

material_sugerido = "Desconocido"
for mat, (zmin, zmax) in materiales.items():
    if zmin <= zeta_est <= zmax:
        material_sugerido = mat
        break

print(f"Material sugerido: {material_sugerido}")

# -------------------------
# Gráfico ζ estimado vs materiales
# -------------------------
fig, ax = plt.subplots(figsize=(10, 5))

# Dibujar barras horizontales para cada material
for i, (mat, (zmin, zmax)) in enumerate(materiales.items()):
    ax.hlines(y=i, xmin=zmin, xmax=zmax, linewidth=10, label=mat)
    ax.text(zmax + 0.005, i, mat, va='center', fontsize=10)

# Línea vertical con ζ estimado
ax.axvline(zeta_est, color="black", linestyle="--", linewidth=2)
ax.text(zeta_est + 0.005, len(materiales), f"ζ estimado: {zeta_est:.3f}", rotation=45, va='bottom', ha='left')

# Ajustes del eje
ax.set_xlim(0, 0.55)
ax.set_yticks([])
ax.set_xlabel("Amortiguamiento adimensional (ζ)", fontsize=12)
ax.set_title("Comparación de ζ estimado con materiales típicos", fontsize=13)

plt.tight_layout()
plt.show()