"""Genera las figuras del informe (en ../informe/figuras) y un resumen numérico."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from modelo import *  # noqa

OUT = os.path.join(os.path.dirname(__file__), "..", "informe", "figuras")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
    "mathtext.fontset": "cm", "font.family": "serif",
    "figure.dpi": 110, "axes.axisbelow": True,
})
A0 = 0.1                       # m^-1  (parábola z = 0.1 x²: a 10 m del eje sube 10 m)
OM = omega_circular(A0)        # rad/s
res = []                       # líneas del resumen numérico


def guardar(fig, nombre, png=False):
    ext = "png" if png else "pdf"
    fig.savefig(os.path.join(OUT, f"{nombre}.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def curva_parabola(ax, a, xm):
    xs = np.linspace(-xm, xm, 300)
    ax.plot(xs, a * xs**2, color="C0", lw=2)


# ------------------------------------------------------------------ b) -----
# b1: trayectorias x(t) y retrato de fases
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for x0 in [2, 4, 6, 8, 10]:
    t, x, v = integrar_b(x0, 0, A0, 14, n=1500)
    ax[0].plot(t, x, label=f"$x_0={x0}$ m")
    ax[1].plot(x, v, label=f"$x_0={x0}$ m")
ax[0].set(xlabel="$t$ (s)", ylabel="$x$ (m)", title="Posición")
ax[1].set(xlabel="$x$ (m)", ylabel=r"$\dot{x}$ (m/s)", title="Retrato de fases")
ax[0].legend(ncol=2, fontsize=8)
guardar(fig, "b_trayectorias")

# b2: periodo vs amplitud y vs a
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
A = np.linspace(0.1, 25, 200)
ax[0].plot(A, periodo_exacto_b(A, A0), label="exacto (integral elíptica)")
Ap = np.array([1, 2, 4, 6, 8, 10, 14, 18, 22])
ax[0].plot(Ap, [periodo_numerico_b(x, A0) for x in Ap], "o", mfc="none", label="integración numérica")
ax[0].axhline(periodo_pequenas_oscilaciones(A0), color="k", ls="--", label="$T_0=2\\pi/\\sqrt{2ga}$")
ax[0].plot(A, 4 * np.sqrt(2 * A0 / G) * A, ":", color="C3", label="asíntota $4\\sqrt{2a/g}\\,A$")
ax[0].set(xlabel="amplitud $A$ (m)", ylabel="periodo $T$ (s)", ylim=(0, 25),
          title=f"Periodo vs amplitud ($a={A0}$ m$^{{-1}}$)")
ax[0].legend(fontsize=8)
av = np.logspace(-2, 0, 40)
for xa, c in zip([0.5, 2, 8], ["C0", "C1", "C2"]):
    ax[1].loglog(av, [periodo_exacto_b(xa, a) for a in av], color=c, label=f"$A={xa}$ m")
ax[1].loglog(av, periodo_pequenas_oscilaciones(av), "k--", label="$T_0\\propto a^{-1/2}$")
ax[1].set(xlabel="$a$ (m$^{-1}$)", ylabel="$T$ (s)", title="Periodo vs pendiente de la parábola")
ax[1].legend(fontsize=8)
guardar(fig, "b_periodo")

# b3: conservación de la energía y dependencia con g y con la masa
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
t, x, v = integrar_b(8, 0, A0, 200, n=20001)
E = energia_b(x, v, A0)
ax[0].semilogy(t, np.abs(E / E[0] - 1) + 1e-18)
ax[0].set(xlabel="$t$ (s)", ylabel=r"$|E/E_0-1|$", title="Conservación de la energía")
res.append(f"b) max|E/E0-1| en 200 s = {np.abs(E/E[0]-1).max():.2e}")
for g_, nom in [(9.81, "Tierra"), (1.62, "Luna"), (24.8, "Júpiter")]:
    t2, x2, _ = integrar_b(8, 0, A0, 15, g=g_, n=1500)
    ax[1].plot(t2, x2, label=f"{nom} ($g={g_}$)")
    res.append(f"b) T(g={g_}) = {periodo_exacto_b(8, A0, g_):.3f} s")
ax[1].set(xlabel="$t$ (s)", ylabel="$x$ (m)", title="Efecto de $g$ (la masa no interviene)")
ax[1].legend(fontsize=8)
guardar(fig, "b_energia_g")

# b4: caso con velocidad inicial desde el fondo
fig, ax = plt.subplots(figsize=(5, 3.6))
for v0 in [2, 4, 6, 8]:
    t, x, v = integrar_b(0, v0, A0, 20, n=2000)
    ax.plot(t, x, label=f"$v_0={v0}$ m/s")
    res.append(f"b) x_max(v0={v0}) = {amplitud_b(0, v0, A0):.3f} m (numérico {x.max():.3f})")
ax.set(xlabel="$t$ (s)", ylabel="$x$ (m)", title="Lanzamientos desde el fondo ($x_0=0$)")
ax.legend(fontsize=8)
guardar(fig, "b_desde_fondo")

res.append(f"b) T0 = {periodo_pequenas_oscilaciones(A0):.4f} s ; T(A=8) exacto = {periodo_exacto_b(8, A0):.4f} s ; "
           f"T(8)/T0 = {periodo_exacto_b(8, A0)/periodo_pequenas_oscilaciones(A0):.4f}")

# ------------------------------------------------------------------ c) -----
# c1: vistas superiores para distintos phi'_0
fig, axs = plt.subplots(1, 4, figsize=(12, 3.5))
x0 = 8.0
for ax, f in zip(axs, [0.4, 0.8, 1.0, 1.4]):
    r = integrar_c(x0, 0, f * OM, A0, 40, n=6000)
    X, Y, Z = posicion_3d(r["x"], r["phi"], A0)
    ax.plot(X, Y, lw=0.9)
    ax.plot(X[0], Y[0], "go")
    ax.plot(0, 0, "k+")
    ax.set_aspect("equal")
    ax.set_title(f"$\\dot\\varphi_0={f}\\,\\Omega_c$\n$\\rho\\in[{r['x'].min():.1f},{r['x'].max():.1f}]$ m", fontsize=9)
    ax.set_xlabel("$x$ (m)")
    E = energia_c(r["x"], r["v"], r["phid"], A0)
    res.append(f"c) phid0={f}Oc: rho in [{r['x'].min():.3f},{r['x'].max():.3f}], "
               f"dE/E={np.abs(E/E[0]-1).max():.1e}, dL/L={np.abs((r['x']**2*r['phid'])/r['L']-1).max():.1e}")
axs[0].set_ylabel("$y$ (m)")
guardar(fig, "c_vista_superior")

# c2: potencial efectivo
fig, ax = plt.subplots(figsize=(5.5, 3.8))
xs = np.linspace(0.8, 12, 400)
for f, c in zip([0.4, 0.8, 1.0, 1.4], ["C0", "C1", "C2", "C3"]):
    L = x0**2 * f * OM
    ax.plot(xs, u_eff(xs, L, A0), color=c, label=f"$\\dot\\varphi_0={f}\\,\\Omega_c$")
    ax.plot(x0, u_eff(x0, L, A0), "o", color=c)
ax.set(xlabel=r"$\rho$ (m)", ylabel=r"$U_{\rm ef}/m$ (J/kg)", ylim=(0, 25),
       title=r"Potencial efectivo ($\rho_0=8$ m, $\dot\rho_0=0$)")
ax.legend(fontsize=8)
guardar(fig, "c_potencial_efectivo")

# c3: caso límite l -> 0 (se recupera el apartado b)
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, f in zip(axs[:2], [0.02, 0.1]):
    r = integrar_c(x0, 0, f * OM, A0, 25, n=20000)
    X, Y, _ = posicion_3d(r["x"], r["phi"], A0)
    ax.plot(X, Y, lw=0.7)
    ax.set_aspect("equal")
    ax.set_title(f"$\\dot\\varphi_0={f}\\,\\Omega_c$", fontsize=9)
    ax.set_xlabel("$x$ (m)")
    res.append(f"c) l->0: phid0={f}Oc -> rho_min = {r['x'].min():.3f} m")
axs[0].set_ylabel("$y$ (m)")
t, xb, _ = integrar_b(x0, 0, A0, 25, n=2000)
r0 = integrar_c(x0, 0, 0.0, A0, 25, n=2000)
axs[2].plot(t, xb, label="apartado b)")
axs[2].plot(r0["t"], r0["x"], "--", label="c) con $l=0$")
axs[2].set(xlabel="$t$ (s)", ylabel="$x$ (m)", title="$l=0$: movimiento plano")
axs[2].legend(fontsize=8)
guardar(fig, "c_limite_l0")

# c4: ángulo apsidal y frecuencias
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
xc = np.linspace(0.5, 25, 100)
for a_, c in zip([0.05, 0.1, 0.2], ["C0", "C1", "C2"]):
    ax[0].plot(xc, angulo_apsidal_teorico(xc, a_) / np.pi, color=c, label=f"$a={a_}$ m$^{{-1}}$")
    xs_ = np.array([1, 4, 8, 14, 22])
    ax[0].plot(xs_, [angulo_apsidal_numerico(x, 1e-3, a_) / np.pi for x in xs_], "o", color=c, mfc="none")
    res.append(f"c) apsidal a={a_}: " + ", ".join(
        f"xc={x}: num {angulo_apsidal_numerico(x,1e-3,a_)/np.pi:.4f}pi teor {angulo_apsidal_teorico(x,a_)/np.pi:.4f}pi"
        for x in xs_))
ax[0].axhline(0.5, color="k", ls="--")
ax[0].set(xlabel="radio de la órbita circular $\\rho_c$ (m)", ylabel=r"$\Delta\varphi_{\rm apsidal}/\pi$",
          title="Precesión: teoría (líneas) y simulación (círculos)")
ax[0].legend(fontsize=8)
# Efecto de kappa = I/m
for kap, c in zip([0, 10, 40, 160], ["C0", "C1", "C2", "C3"]):
    r = integrar_c(8, 0, 0.8 * OM, A0, 30, kappa=kap, n=3000)
    ax[1].plot(r["t"], r["x"], color=c, label=f"$\\kappa=I/m={kap}$ m$^2$")
    res.append(f"c) kappa={kap}: rho in [{r['x'].min():.3f},{r['x'].max():.3f}]")
ax[1].set(xlabel="$t$ (s)", ylabel="$x$ (m)", title="Efecto de la inercia de los rails")
ax[1].legend(fontsize=8)
guardar(fig, "c_apsidal_kappa")

# c5: rails con Omega impuesta
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for f, c in zip([0, 0.5, 0.9], ["C0", "C1", "C2"]):
    t, x, v = integrar_omega(2.0, 0, A0, f * OM, 25, n=2500)
    ax[0].plot(t, x, color=c, label=f"$\\Omega={f}\\,\\Omega_c$")
ax[0].set(xlabel="$t$ (s)", ylabel="$x$ (m)", title="Subcrítico: oscila",
          ylim=(-2.6, 2.6))
ax[0].legend(fontsize=8)
for f, c in zip([1.0, 1.05, 1.3], ["C3", "C4", "C5"]):
    t, x, v = integrar_omega(0.05, 0, A0, f * OM, 25, n=2500, x_max=20)
    ax[1].plot(t, x, color=c, label=f"$\\Omega={f}\\,\\Omega_c$")
ax[1].set(xlabel="$t$ (s)", ylabel="$x$ (m)", title="Crítico y supercrítico: sale despedido",
          ylim=(-0.5, 21))
ax[1].legend(fontsize=8)
guardar(fig, "c_omega_impuesta")
res.append(f"c) Omega_c = sqrt(2ga) = {OM:.4f} rad/s ({OM*60/(2*np.pi):.2f} rpm)")

# c6: trayectoria 3D
r = integrar_c(8, 0, 0.8 * OM, A0, 25, n=4000)
X, Y, Z = posicion_3d(r["x"], r["phi"], A0)
fig = plt.figure(figsize=(6.2, 5))
ax = fig.add_subplot(111, projection="3d")
rr, pp = np.meshgrid(np.linspace(0, 9.5, 25), np.linspace(0, 2 * np.pi, 60))
ax.plot_surface(rr * np.cos(pp), rr * np.sin(pp), A0 * rr**2, alpha=0.12, color="C0", linewidth=0)
ax.plot(X, Y, Z, color="C3", lw=0.9)
ax.set(xlabel="$x$ (m)", ylabel="$y$ (m)", zlabel="$z$ (m)")
ax.view_init(25, -60)
guardar(fig, "c_trayectoria_3d", png=True)

# ------------------------------------------------------------------ d) -----
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for x0_, c in zip([2, 4, 8, 12], ["C0", "C1", "C2", "C3"]):
    t, x, v = integrar_b(x0_, 0, A0, 14, n=2000)
    N = normal_b(x, v, A0) / G
    ax[0].plot(t, N, color=c, label=f"$x_0={x0_}$ m")
    ax[0].axhline(1 + 4 * A0**2 * x0_**2, color=c, ls=":", lw=0.8)
    res.append(f"d) x0={x0_}: N/mg max numerico = {N.max():.4f}, teor 1+4a²x0² = {1+4*A0**2*x0_**2:.4f}; "
               f"N/mg en el extremo = {N[0]:.4f}, teor 1/sqrt(1+4a²x0²) = {1/np.sqrt(1+4*A0**2*x0_**2):.4f}")
ax[0].axhline(1, color="k", lw=0.6)
ax[0].set(xlabel="$t$ (s)", ylabel="$N/(mg)$", title="Reacción normal (punteado: máximo teórico)")
ax[0].legend(fontsize=8)

# verificación independiente: N = m (r'' - g) con derivadas numéricas de la trayectoria
t, x, v = integrar_b(8, 0, A0, 14, n=140001)
z = A0 * x**2
dt = t[1] - t[0]
xdd = np.gradient(np.gradient(x, dt), dt)
zdd = np.gradient(np.gradient(z, dt), dt)
N_fd = np.hypot(xdd, zdd + G)
N_an = normal_b(x, v, A0)
sl = slice(10, -10)
err = np.max(np.abs(N_fd[sl] - N_an[sl]) / N_an[sl])
res.append(f"d) verificación N analítica vs derivadas numéricas: error relativo máx = {err:.2e}")
ax[1].plot(t, N_an / G, label="fórmula (multiplicador)")
ax[1].plot(t[sl][::800], N_fd[sl][::800] / G, "o", mfc="none", label="$m|\\ddot{\\vec r}-\\vec g|$ numérico")
ax[1].set(xlabel="$t$ (s)", ylabel="$N/(mg)$", title="Comprobación independiente", xlim=(0, 7))
ax[1].legend(fontsize=8)

# N a lo largo de la trayectoria y descomposición vectorial
t, x, v = integrar_b(8, 0, A0, 6.7, n=3000)
curva_parabola(ax[2], A0, 9.5)
idx = np.linspace(0, len(t) - 1, 14).astype(int)
for i in idx:
    lam = lambda_b(x[i], v[i], A0)
    Nx, Nz = -2 * A0 * x[i] * lam, lam
    esc = 1.2 / G
    ax[2].annotate("", xy=(x[i] + Nx * esc, A0 * x[i]**2 + Nz * esc), xytext=(x[i], A0 * x[i]**2),
                   arrowprops=dict(arrowstyle="->", color="C3"))
ax[2].plot(x[idx], A0 * x[idx]**2, "k.", ms=4)
ax[2].set(xlabel="$x$ (m)", ylabel="$z$ (m)", title="Vector $\\vec N$ (escala: $mg$ = 1.2 m)")
ax[2].set_aspect("equal")
guardar(fig, "d_normal")

with open(os.path.join(os.path.dirname(__file__), "resumen_numerico.txt"), "w") as f:
    f.write("\n".join(res) + "\n")
print("\n".join(res))
