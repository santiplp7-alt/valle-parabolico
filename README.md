# El Valle Parabólico — Mecánica y Ondas I (UCO)

Trabajo de la asignatura **Mecánica y Ondas I** (Grado en Física, Universidad de Córdoba).
Ejercicio 1 del examen extraordinario 2025-26: vehículo sobre rails parabólicos `z = a·x²`
(formulación lagrangiana, rails fijos y giratorios, reacción normal con multiplicadores de Lagrange).

## Contenido

| Carpeta | Qué hay |
|---|---|
| `informe/` | `informe.tex` (LaTeX) y `informe.pdf`; figuras en `informe/figuras/` |
| `simulacion/modelo.py` | Ecuaciones del movimiento, energías, reacción normal, periodo exacto, ángulo apsidal… |
| `simulacion/figuras.py` | Genera todas las figuras del informe y `resumen_numerico.txt` |
| `simulacion/animaciones.py` | Genera las animaciones GIF |
| `animaciones/` | GIFs ya generados |

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python simulacion/figuras.py          # figuras del informe
python simulacion/animaciones.py      # b, c, d  (o:  python simulacion/animaciones.py c)
cd informe && pdflatex informe.tex && pdflatex informe.tex
```

## Animaciones

![b](animaciones/b_valle_parabolico.gif)
![c](animaciones/c_valle_giratorio.gif)
![d](animaciones/d_reaccion_normal.gif)

## Resultados principales

* Rails fijos: `(1+4a²x²) x'' + 4a²x x'² + 2ga x = 0`, periodo exacto `T = 4/√(2ga) · E(-4a²A²)`.
* Rails giratorios: `l = mρ²φ'` y `E` conservadas; órbita circular con `φ' = √(2ga)`;
  ángulo apsidal `(π/2)·√(1+4a²ρc²)`.
* Reacción normal: `N = m(g+2a x'²)/√(1+4a²x²)`, máxima en el fondo: `N = mg(1+4a²x0²)`.

Parámetros por defecto: `a = 0.1 m⁻¹`, `g = 9.81 m/s²`.
