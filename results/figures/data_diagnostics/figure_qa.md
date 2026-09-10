# Stage 1B figure QA

- Backend: Python / matplotlib only.
- Archetype: quantitative evidence figures.
- Source data: canonical processed CSV files and generated diagnostic tables.
- Exclusions: none. Daylight-only panels use the declared physical predicate `PV > 0.0` at both members of each lag pair.
- Representative day: 2025-03-19, selected as the day whose PV energy is closest to the annual daily median.
- Hankel contract: history=1008, rows=144, spectrum stride=1008, forecastability stride=144.
- Forecastability target: next-day Yesterday baseline error, with no future data in the predictor.
- Exports: SVG and PDF with editable text; JPG at 450 dpi.
- Typography: Songti SC first; minus-sign rendering enabled.
