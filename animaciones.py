"""Genera las animaciones GIF en ../animaciones.

Uso:  python animaciones.py [b|c|d|todas]
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon

sys.path.insert(0, os.path.dirname(__file__))
from modelo import *  # noqa

OUT = os.path.join(os.path.dirname(__file__), "..", "animaciones")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif", "font.size": 9})

FPS = 20
A0 = 0.1


def coche(x, a, w=1.8, h=0.7):
    """Polígono (carrocería) orientado según la tangente a la parábola en x."""
    th = np.arctan(2 * a * x)
    c, s = np.cos(th), np.sin(th)
    pts = np.array([[-w / 2, 0], [w / 2, 0], [w / 2, h], [-w / 2, h]])
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T + np.array([x, a * x**2])
    return pts


# -----------------------------------------------------------------------------
def anim_b(x0=8.0, a=A0, nper=2.0):
    T = periodo_exacto_b(x0, a)
    t_f = nper * T
    n = int(t_f * FPS)
    t, x, v = integrar_b(x0, 0, a, t_f, n=n)
    Ec = 0.5 * (1 + 4 * a**2 * x**2) * v**2
    Ep = G * a * x**2
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.3, 1]})
    xs = np.linspace(-x0 - 1.5, x0 + 1.5, 300)
    ax.plot(xs, a * xs**2, color="C0", lw=2.5)
    ax.set(xlim=(-x0 - 1.5, x0 + 1.5), ylim=(-1, a * (x0 + 1.5)**2 + 1.5), xlabel="$x$ (m)", ylabel="$z$ (m)",
           title=f"El Valle Parabólico: $a={a}$ m$^{{-1}}$, $x_0={x0}$ m")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    car = Polygon(coche(x[0], a), closed=True, fc="C3", ec="k")
    ax.add_patch(car)
    trail, = ax.plot([], [], "C3", lw=0.8, alpha=0.5)
    ax2.plot(t, Ec, "C1", lw=0.8, label="$E_c/m$")
    ax2.plot(t, Ep, "C2", lw=0.8, label="$E_p/m$")
    ax2.plot(t, Ec + Ep, "k", lw=1.2, label="$E/m$")
    cur = ax2.axvline(0, color="gray")
    ax2.set(xlabel="$t$ (s)", ylabel="J/kg", title="Energías por unidad de masa")
    ax2.legend(loc="center right", fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()

    def upd(i):
        car.set_xy(coche(x[i], a))
        j = max(0, i - 30)
        trail.set_data(x[j:i + 1], a * x[j:i + 1]**2 + 0.35)
        cur.set_xdata([t[i], t[i]])
        return car, trail, cur

    FuncAnimation(fig, upd, frames=n, blit=False).save(
        os.path.join(OUT, "b_valle_parabolico.gif"), writer=PillowWriter(fps=FPS), dpi=70)
    plt.close(fig)


# -----------------------------------------------------------------------------
def anim_c(x0=8.0, a=A0, f=0.8, t_f=20.0):
    n = int(t_f * 12)  # 12 fps para que el GIF pese menos
    r = integrar_c(x0, 0, f * omega_circular(a), a, t_f, n=n)
    X, Y, Z = posicion_3d(r["x"], r["phi"], a)
    fig = plt.figure(figsize=(10, 4.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    S = x0 + 1.5
    rr, pp = np.meshgrid(np.linspace(0, S, 14), np.linspace(0, 2 * np.pi, 36))
    ax.plot_wireframe(rr * np.cos(pp), rr * np.sin(pp), a * rr**2, color="C0", alpha=0.12, lw=0.4)
    ss = np.linspace(-S, S, 120)
    rail, = ax.plot([], [], [], "C0", lw=2.5)
    trail, = ax.plot([], [], [], "C3", lw=0.8)
    pt, = ax.plot([], [], [], "o", color="C3", ms=8, mec="k")
    ax.set(xlim=(-S, S), ylim=(-S, S), zlim=(0, a * S**2 + 1), xlabel="$x$", ylabel="$y$", zlabel="$z$")
    ax.set_box_aspect((1, 1, 0.7))
    ax.set_title("El Valle Giratorio (rails libres, $\\kappa=0$)")
    ax2.set_aspect("equal")
    ax2.set(xlim=(-S, S), ylim=(-S, S), xlabel="$x$ (m)", ylabel="$y$ (m)", title="Vista desde arriba")
    ax2.grid(alpha=0.3)
    rline, = ax2.plot([], [], "C0", lw=2)
    tr2, = ax2.plot([], [], "C3", lw=0.8)
    pt2, = ax2.plot([], [], "o", color="C3", mec="k")
    ax2.plot(0, 0, "k+")
    txt = ax2.text(0.02, 0.96, "", transform=ax2.transAxes, va="top")
    fig.tight_layout()

    def upd(i):
        ph = r["phi"][i]
        rail.set_data(ss * np.cos(ph), ss * np.sin(ph))
        rail.set_3d_properties(a * ss**2)
        trail.set_data(X[:i + 1], Y[:i + 1])
        trail.set_3d_properties(Z[:i + 1])
        pt.set_data([X[i]], [Y[i]])
        pt.set_3d_properties([Z[i]])
        ax.view_init(24, -60 + 0.4 * i)
        rline.set_data(ss * np.cos(ph), ss * np.sin(ph))
        tr2.set_data(X[:i + 1], Y[:i + 1])
        pt2.set_data([X[i]], [Y[i]])
        txt.set_text(f"$t={r['t'][i]:.1f}$ s   $\\rho={r['x'][i]:.2f}$ m")
        return rail, trail, pt

    FuncAnimation(fig, upd, frames=n, blit=False).save(
        os.path.join(OUT, "c_valle_giratorio.gif"), writer=PillowWriter(fps=12), dpi=50)
    plt.close(fig)


# -----------------------------------------------------------------------------
def anim_d(x0=8.0, a=A0, nper=2.0):
    T = periodo_exacto_b(x0, a)
    t_f = nper * T
    n = int(t_f * FPS)
    t, x, v = integrar_b(x0, 0, a, t_f, n=n)
    lam = lambda_b(x, v, a)
    Nm = normal_b(x, v, a) / G
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.3, 1]})
    xs = np.linspace(-x0 - 1.5, x0 + 1.5, 300)
    ax.plot(xs, a * xs**2, color="C0", lw=2.5)
    ax.set(xlim=(-x0 - 3, x0 + 3), ylim=(-3, a * (x0 + 1.5)**2 + 3), xlabel="$x$ (m)", ylabel="$z$ (m)",
           title="Reacción normal $\\vec N$ (rojo) y peso $m\\vec g$ (verde)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    car = Polygon(coche(x[0], a), closed=True, fc="0.8", ec="k")
    ax.add_patch(car)
    esc = 1.6 / G
    aN = ax.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="C3", lw=2))
    aW = ax.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="C2", lw=2))
    ax2.plot(t, Nm, "C3")
    ax2.axhline(1 + 4 * a**2 * x0**2, color="k", ls=":", label="$1+4a^2x_0^2$")
    ax2.axhline(1, color="C2", lw=0.8)
    cur, = ax2.plot([], [], "ko")
    ax2.set(xlabel="$t$ (s)", ylabel="$N/(mg)$", title="Reacción normal")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()

    def upd(i):
        car.set_xy(coche(x[i], a))
        px, pz = x[i], a * x[i]**2 + 0.35
        Nx, Nz = -2 * a * x[i] * lam[i], lam[i]
        aN.set_position((px, pz))
        aN.xy = (px + Nx * esc, pz + Nz * esc)
        aW.set_position((px, pz))
        aW.xy = (px, pz - G * esc)
        cur.set_data([t[i]], [Nm[i]])
        return car, cur

    FuncAnimation(fig, upd, frames=n, blit=False).save(
        os.path.join(OUT, "d_reaccion_normal.gif"), writer=PillowWriter(fps=FPS), dpi=70)
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "todas"
    if which in ("b", "todas"):
        anim_b()
    if which in ("c", "todas"):
        anim_c()
    if which in ("d", "todas"):
        anim_d()
    print("Animaciones guardadas en", os.path.abspath(OUT))
