# Stage 2A figure QA

## Figure contract

- `fig_p2_generation_model_comparison`: show whether the February--December daily generation RMSE differs materially across phase, fusion and low-rank analytical experts. The left panel preserves the useful-method scale; the right panel separately shows the much larger low-rank errors rather than compressing the informative comparisons.
- `fig_p2_joint_residual_scenarios`: show that paired whole-day residual resampling produces temporally coherent joint load/generation scenario envelopes around the fixed point forecasts.
- Archetype: quantitative grid. Backend: Python/matplotlib only.

## Data and statistics

- Source tables: `results/tables/forecasting/q2_daily_metrics.csv`, `q2_forecast_predictions.csv` and `q2_scenarios.npz`.
- Model-comparison replicate unit: one complete operating day; all 334 formal evaluation days are included for every model.
- Center: mean of the 334 daily RMSE values.
- Interval: nonparametric 95% bootstrap confidence interval of the mean, 2000 resamples, seed 2026.
- No hypothesis test or multiple-comparison correction is claimed in the figure.
- Scenario figure date: the evaluation day whose observed daily generation is closest to the evaluation-period median; this rule is fixed and does not select by forecast error.
- Scenario count: 50. Every scenario uses one historical day's complete paired 144-point load/generation residual vectors.
- No observations or model results are excluded. No simulated performance values are used.

## Export and visual checks

- Shared style: Songti SC first, Chinese labels, editable SVG/PDF text, restrained project palette.
- Outputs: SVG, PDF, JPG at 450 dpi, and PNG at 600 dpi.
- The source preflight reports no failures for the shared export/style helper. The plotting module's static-only failures arise because font and export logic are imported from that helper; rendered outputs were inspected at final size.
- Visual inspection: labels, intervals, lines and scenario bands are legible; no clipping, overlap or misleading common-axis compression remains.
