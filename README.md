# AFP Switcher (Chile) — Recomendador de cambios de fondo A–E

Este proyecto genera **recomendaciones de cambio de fondo (A–E)** a partir de históricos (valores cuota o retornos), incluyendo **backtest** para evaluar la estrategia.

> Importante: esto es una herramienta cuantitativa/educativa. No es asesoría financiera. Tú eres responsable de las decisiones y de validar fuentes de datos.

## Requisitos

- Python 3.10+ (recomendado 3.11)

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Formato de datos (CSV)

Opción A: **valores cuota** por fondo (recomendado). Ejemplo: `data/example_prices.csv`

- `date`: fecha (YYYY-MM-DD)
- `A`,`B`,`C`,`D`,`E`: valor cuota (o índice) por fondo

## Uso (CLI)

### 1) Recomendar fondo hoy (según última fecha del CSV)

```bash
python -m afp_switcher recommend --prices data/example_prices.csv
```

### 2) Backtest

```bash
python -m afp_switcher backtest --prices data/example_prices.csv --start 2020-01-01
```

## Qué hace la estrategia (resumen)

- Calcula retornos diarios desde valores cuota.
- Estima “score” por fondo usando:
  - **Momentum** (retorno acumulado a 63 y 252 días),
  - penalización por **volatilidad** (EWMA),
  - penalización por **cambios** (coste por switch y mínimo de días entre switches).
- Selecciona el fondo con mejor score y aplica reglas de “cooldown”.

## Próximos pasos (si quieres hacerlo “real” con Chile)

Si me dices **de dónde sacarás los datos** (por ejemplo, CSV de la Superintendencia, o export de tu AFP), lo adapto al formato exacto y agrego:

- normalización por feriados,
- manejo de fondos no disponibles,
- límites de cambios mensuales,
- reporte por horizonte (1–3–5 años) y perfil de riesgo.

