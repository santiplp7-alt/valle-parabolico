"""
Modelo físico del ejercicio 1 (Mecánica y Ondas I, UCO).

Vehículo (masa puntual) sobre unos rails con forma de parábola vertical z = a·x².
Todas las magnitudes en unidades del SI. Las energías y las fuerzas se dan por
unidad de masa (la masa m se cancela en todas las ecuaciones del movimiento).

  b) atracción fija            -> 1 grado de libertad  (x)
  c) rails que giran           -> 2 grados de libertad (x, phi)
  d) reacción normal           -> Lagrangiana auxiliar (multiplicador lambda)
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import ellipe

G = 9.81  # m/s^2
RTOL, ATOL = 1e-11, 1e-12


# ----------------------------------------------------------------------------
# b) Atracción fija:  L = m/2 (1+4a²x²) x'² - m g a x²
# ----------------------------------------------------------------------------
def rhs_b(t, y, a, g=G):
    """(x, x') -> (x', x'') con (1+4a²x²)x'' + 4a²x x'² + 2ga x = 0."""
    x, v = y
    return [v, -(4 * a**2 * x * v**2 + 2 * g * a * x) / (1 + 4 * a**2 * x**2)]


def energia_b(x, v, a, g=G):
    """Energía mecánica por unidad de masa."""
    return 0.5 * (1 + 4 * a**2 * x**2) * v**2 + g * a * x**2


def integrar_b(x0, v0, a, t_f, g=G, n=2000):
    t = np.linspace(0.0, t_f, n)
    s = solve_ivp(rhs_b, (0, t_f), [x0, v0], args=(a, g), t_eval=t,
                  method="DOP853", rtol=RTOL, atol=ATOL)
    return s.t, s.y[0], s.y[1]


def amplitud_b(x0, v0, a, g=G):
    """Punto de retorno x_max a partir de la energía: g a A² = E."""
    return np.sqrt(energia_b(x0, v0, a, g) / (g * a))


def periodo_pequenas_oscilaciones(a, g=G):
    return 2 * np.pi / np.sqrt(2 * g * a)


def periodo_exacto_b(A, a, g=G):
    """T = 4/sqrt(2ga) · E(m=-4a²A²)  (integral elíptica completa de 2ª especie)."""
    return 4.0 / np.sqrt(2 * g * a) * ellipe(-4 * a**2 * A**2)


def periodo_numerico_b(x0, a, g=G):
    """Periodo integrando la ecuación del movimiento (soltando desde el reposo en x0>0)."""
    def cruce(t, y, a, g):
        return y[0]
    cruce.terminal, cruce.direction = True, -1
    s = solve_ivp(rhs_b, (0, 1e4), [x0, 0.0], args=(a, g), events=cruce,
                  method="DOP853", rtol=RTOL, atol=ATOL)
    return 4.0 * s.t_events[0][0]


# ----------------------------------------------------------------------------
# d) Reacción normal con la lagrangiana auxiliar
#    L' = m/2 (x'² + z'²) - m g z + lambda (z - a x²)
#    m x'' = -2 a lambda x ;  m z'' = -m g + lambda ;  z = a x²
# ----------------------------------------------------------------------------
def lambda_b(x, v, a, g=G, m=1.0):
    """Multiplicador de Lagrange = componente vertical de la normal."""
    return m * (g + 2 * a * v**2) / (1 + 4 * a**2 * x**2)


def normal_b(x, v, a, g=G, m=1.0):
    """Módulo de la reacción normal N = lambda · sqrt(1+4a²x²)."""
    return lambda_b(x, v, a, g, m) * np.sqrt(1 + 4 * a**2 * x**2)


# ----------------------------------------------------------------------------
# c) Rails que giran (modelo libre): coordenadas (x, phi),  kappa = I_rails / m [m²]
#    L = m/2 [(1+4a²x²) x'² + x² phi'²] + I/2 phi'² - m g a x²
#    p_phi = (m x² + I) phi' = const
# ----------------------------------------------------------------------------
def rhs_c(t, y, a, L, kappa=0.0, g=G):
    """y = (x, x', phi). L = p_phi/m (constante)."""
    x, v, phi = y
    ph = L / (x * x + kappa)
    xpp = (x * ph**2 - 4 * a**2 * x * v**2 - 2 * g * a * x) / (1 + 4 * a**2 * x**2)
    return [v, xpp, ph]


def momento_c(x0, phidot0, kappa=0.0):
    return (x0**2 + kappa) * phidot0


def energia_c(x, v, phidot, a, kappa=0.0, g=G):
    return 0.5 * ((1 + 4 * a**2 * x**2) * v**2 + (x**2 + kappa) * phidot**2) + g * a * x**2


def integrar_c(x0, v0, phidot0, a, t_f, kappa=0.0, g=G, n=4000, phi0=0.0):
    L = momento_c(x0, phidot0, kappa)
    t = np.linspace(0.0, t_f, n)
    s = solve_ivp(rhs_c, (0, t_f), [x0, v0, phi0], args=(a, L, kappa, g), t_eval=t,
                  method="DOP853", rtol=RTOL, atol=ATOL)
    x, v, phi = s.y
    phid = L / (x**2 + kappa)
    return dict(t=s.t, x=x, v=v, phi=phi, phid=phid, L=L)


def omega_circular(a, g=G):
    """Velocidad angular de la órbita circular (independiente del radio y de kappa)."""
    return np.sqrt(2 * g * a)


def puntos_retorno_c(E, l_m, a, g=G):
    """Radios de retorno (kappa=0). E, l_m = l/m por unidad de masa."""
    disc = E**2 - 2 * g * a * l_m**2
    if disc < 0:
        return None
    return (np.sqrt((E - np.sqrt(disc)) / (2 * g * a)),
            np.sqrt((E + np.sqrt(disc)) / (2 * g * a)))


def u_eff(x, L, a, kappa=0.0, g=G):
    return 0.5 * L**2 / (x**2 + kappa) + g * a * x**2


def cond_cuasicircular(xc, eps, a, kappa=0.0, g=G):
    """Órbita casi circular: L de la circular de radio xc, pero se parte de xc(1+eps)."""
    L = (xc**2 + kappa) * omega_circular(a, g)
    x0 = xc * (1 + eps)
    return x0, 0.0, L / (x0**2 + kappa)


def omega_radial(xc, a, kappa=0.0, g=G):
    """Frecuencia de pequeñas oscilaciones radiales alrededor de la órbita circular."""
    return np.sqrt(8 * g * a * xc**2 / ((xc**2 + kappa) * (1 + 4 * a**2 * xc**2)))


def angulo_apsidal_teorico(xc, a):
    """Ángulo entre dos retornos consecutivos (kappa=0): (π/2)·sqrt(1+4a²xc²)."""
    return 0.5 * np.pi * np.sqrt(1 + 4 * a**2 * xc**2)


def angulo_apsidal_numerico(xc, eps, a, kappa=0.0, g=G):
    x0, v0, pd0 = cond_cuasicircular(xc, eps, a, kappa, g)
    L = momento_c(x0, pd0, kappa)

    def retorno(t, y, a, L, kappa, g):
        return y[1]
    retorno.terminal = False

    t_f = 30 * 2 * np.pi / omega_radial(xc, a, kappa, g)
    s = solve_ivp(rhs_c, (0, t_f), [x0, v0, 0.0], args=(a, L, kappa, g),
                  events=retorno, method="DOP853", rtol=RTOL, atol=ATOL,
                  dense_output=True)
    te = s.t_events[0]
    te = te[te > 1e-6]
    ph = s.sol(te)[2]
    return np.mean(np.diff(ph))


# ----------------------------------------------------------------------------
# Variante: rails con velocidad angular IMPUESTA Omega (ligadura reónoma, 1 g.l.)
#    L = m/2 [(1+4a²x²) x'² + Omega² x²] - m g a x²
# ----------------------------------------------------------------------------
def rhs_omega(t, y, a, Om, g=G):
    x, v = y
    return [v, ((Om**2 - 2 * g * a) * x - 4 * a**2 * x * v**2) / (1 + 4 * a**2 * x**2)]


def integrar_omega(x0, v0, a, Om, t_f, g=G, n=3000, x_max=None):
    ev = None
    if x_max is not None:
        def ev(t, y, a, Om, g):
            return abs(y[0]) - x_max
        ev.terminal = True
    t = np.linspace(0.0, t_f, n)
    s = solve_ivp(rhs_omega, (0, t_f), [x0, v0], args=(a, Om, g), t_eval=t, events=ev,
                  method="DOP853", rtol=RTOL, atol=ATOL)
    return s.t, s.y[0], s.y[1]


# ----------------------------------------------------------------------------
# Geometría auxiliar
# ----------------------------------------------------------------------------
def posicion_3d(x, phi, a):
    return x * np.cos(phi), x * np.sin(phi), a * x**2
