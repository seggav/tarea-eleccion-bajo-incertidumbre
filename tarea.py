import numpy as np
import matplotlib.pyplot as plt

# 1. Definir estados del mercado y decisiones
estados = ["Baja fuerte", "Baja leve", "Sube leve", "Sube fuerte"]
decisiones = ["No invertir", "Invertir"]

# 2. Matriz de utilidad (fila=estado, columna=decisión)
U = np.array([
    [0, -10],  # omega_0: Baja fuerte
    [0,  -2],  # omega_1: Baja leve
    [0,   5],  # omega_2: Sube leve
    [0,  15]   # omega_3: Sube fuerte
])

# 3. Probabilidades a priori de cada estado
p_omega = np.array([0.3, 0.35, 0.25, 0.1])  # deben sumar 1

# 4. Valor esperado de cada decisión sin información adicional
VE_no_invertir = np.sum(p_omega * U[:,0])
VE_invertir = np.sum(p_omega * U[:,1])

print("Valor esperado DECISION SIN OBSERVACIONES:")
print(f" - No invertir: {VE_no_invertir:.2f}")
print(f" - Invertir   : {VE_invertir:.2f}")

best_decision = "No invertir" if VE_no_invertir > VE_invertir else "Invertir"
print(f"Mejor decisión sin información adicional: {best_decision}")

# 5. Matriz de probabilidad condicional p(noticia positiva | estado del mercado)
p_z_given_omega = np.array([0.10, 0.30, 0.60, 0.90])

# 6. Probabilidad total de noticia positiva
p_z1 = np.sum(p_z_given_omega * p_omega)

# 7. Probabilidad posterior p(estado | noticia positiva)
post_omega_z1 = (p_z_given_omega * p_omega) / p_z1

print("\nProbabilidad a posteriori dada noticia positiva:")
for i, st in enumerate(estados):
    print(f"  {st}: {post_omega_z1[i]:.2f}")

# 8. Valor esperado de cada decisión con información adicional
VE_no_inv_z1 = np.sum(post_omega_z1 * U[:,0])
VE_inv_z1    = np.sum(post_omega_z1 * U[:,1])

print("\nValor esperado DECISION DADO noticia positiva:")
print(f" - No invertir: {VE_no_inv_z1:.2f}")
print(f" - Invertir   : {VE_inv_z1:.2f}")

best_dec_z1 = "No invertir" if VE_no_inv_z1 > VE_inv_z1 else "Invertir"
print(f"Mejor decisión si hay noticia positiva: {best_dec_z1}")

# 9. Gráfica de comparación de decisiones variando la probabilidad de mercado alcista
p_alcista_range = np.linspace(0, 1, 50)
VE_no_inv = []
VE_inv = []

for p_alcista in p_alcista_range:
    p_baja_fuerte = 0.3 * (1 - p_alcista)
    p_baja_leve = 0.3 * (1 - p_alcista)
    p_sube_leve = 0.4 * p_alcista
    p_sube_fuerte = 0.6 * p_alcista
    p_curr = np.array([p_baja_fuerte, p_baja_leve, p_sube_leve, p_sube_fuerte])
    
    ve0 = np.sum(p_curr * U[:,0])
    ve1 = np.sum(p_curr * U[:,1])
    VE_no_inv.append(ve0)
    VE_inv.append(ve1)

plt.plot(p_alcista_range, VE_no_inv, label="No invertir")
plt.plot(p_alcista_range, VE_inv, label="Invertir")
plt.xlabel("Probabilidad de mercado alcista")
plt.ylabel("Valor esperado de la decisión")
plt.legend()
plt.title("Comparación de decisiones vs. probabilidad de mercado alcista")
plt.show()
