# Regime conditioning for the U-Net FiLM model

> **Status** : design draft, 2026-06-18. Tracks issue D (post-#5 loss redesign).

## Motivation

The current U-Net FiLM (`downscaling.deep_learning.model.DownscalingUNet`) conditions its feature modulation **only** on the DEM. The same modulation function is applied regardless of the synoptic situation. After the regime-stratified evaluation (see `lot-b-calibration-report.md` annex 9 + #38), we know that frost events fall into mechanistically distinct families :

- **R1 Radiative** : calm + clear, dry radiative cooling
- **R2 Advective windy** : forced mixing, cold-air advection
- **R3 Cloudy windy** : perturbed, frontal
- **R4a Cold pool under anticyclone** : calm + cloudy + MSLP ≥ 1020 hPa, valley cold pooling
- **R4b Mild cloudy** : calm + cloudy + no anticyclone

A single global FiLM module struggles to model the regime-specific corrections that physics suggests:

- R1 wants strong DEM × shelter index conditioning (sky view factor, slope orientation)
- R4a wants strong DEM × valley depth conditioning (TPI, drainage path)
- R3 wants weakened DEM influence (mixing destroys cold-pool patterns)

**Regime conditioning** = inject a regime label as additional input to the FiLM modulator. The model learns to weight the DEM features differently per regime.

## What FiLM regime conditioning can and cannot do

### Can do

- **Regime × DEM interactions**: "in R4a, weight TPI strongly; in R3, ignore it"
- **Regime-specific bias correction**: "in R4a expect colder predictions than R1 for similar inputs"
- **Adaptive uncertainty**: combined with pinball loss heads (#5), the model can widen prediction intervals on regimes where it knows it's bad (R3)

### Cannot do

- **Resolve sub-grid orographic features absent from inputs**: if a 5.5 km CERRA cell averages a valley and a plateau, no FiLM modulation will resurrect the cold pool signal. The plateau is needed at finer resolution (AROME 1.3 km via Météo-France partnership, or SURFEX standalone with proper energy balance).
- **Compensate for missing physics**: regime label is a *proxy* for "what type of night is this", not for the underlying energy budget. SURFEX is the proper answer for the latter.

These limits are crucial in the partnership pitch: regime conditioning brings clear gains within the input-resolution envelope, but reaching production-grade R4a detection requires the partnership leverage.

## Architectural choice

### Embedding vs one-hot

We propose **learned embeddings** rather than one-hot encoding:

```python
self.regime_emb = nn.Embedding(n_regimes, regime_embed_dim)
```

Rationale:
- 5–6 regimes only, but the embedding lets the model place "similar" regimes (R4a / R4b) near each other in latent space if data warrants
- Easy to share across all FiLM layers (compute once per sample, reuse)
- Easy to add new regimes later without retraining from scratch (extend the embedding table)

Default `regime_embed_dim = 8` (overkill for 5 regimes but cheap).

### Where to inject the regime context

The regime vector is **concatenated to the pooled DEM feature** before the MLP that generates γ, β :

```
DEM feature ─[AdaptiveAvgPool2d + Flatten]─→ (B, dem_ch)
                                                  │
regime_idx ─[Embedding]─→ context (B, emb_dim)    │
                              │                   │
                              └──[concat]──→ (B, dem_ch + emb_dim)
                                                  │
                                              [MLP]→ γ, β
```

This keeps the modulation fundamentally driven by DEM (the spatial backbone), with the regime acting as a global modifier of the modulation function. Backward compatible: if `regime` is `None` or `n_regimes == 0`, the architecture reverts exactly to current behavior.

### Same embedding across all levels

The regime context is broadcast to **every** FiLM layer (one per encoder level). This lets the model produce regime-specific modulations at every scale (deep semantic + shallow spatial). Alternative would be one embedding per level, but redundant for our scale (5 regimes, 4 levels).

## Training strategy

### Supervision

Regime labels are produced by `flag_regimes.py` (issue #37) from ERA5 synoptic features. The label is a function of the **night**, not of the station — so all samples for a given date share the same regime.

CSV format expected:

```
date,regime,...
2022-02-01,R1,...
2022-02-02,R3,...
...
```

Regime → integer index mapping (suggested):

| Label | Index |
|---|---|
| R0 (catch-all) | 0 |
| R1 Radiative | 1 |
| R2 Advective windy | 2 |
| R3 Cloudy windy | 3 |
| R4a Cold pool anticyclonic | 4 |
| R4b Mild cloudy | 5 |

### Data augmentation

The regime distribution is uneven (~25 % R1, ~10 % R2, ~22 % R3, ~10 % R4a, ~33 % R4b on 2022-2024). To avoid the model collapsing to the majority regime, we suggest:

- **Class-balanced sampling** in the train DataLoader (sampler with regime weights inversely proportional to frequency)
- **Stratified train/val split**: ensure each fold contains the full regime distribution

### Cold start

The embedding is initialized to small values (PyTorch default). For the first few epochs we suggest **freezing the embedding** (only DEM-driven FiLM is updated), then unfreezing for the rest of training. This avoids the regime embedding overfitting before the rest of the model has learned anything useful.

(Not required for the V0 patch — can be a hyperparameter for tuning later.)

## Expected gains

Based on the regime-stratified Lot B baseline (Lot B is regime-agnostic but uses the same DEM info), the expected improvement is :

| Metric | Lot C #5 + FiLM regime | Lot C + AROME (V2) | Lot C + SURFEX (V3) |
|---|---|---|---|
| CSI R1 Radiative | 0.43 → 0.45–0.50 | 0.50 → 0.55 | 0.55+ |
| CSI R4a Cold pool | 0.43 → 0.43–0.48 | 0.48 → 0.55 | 0.55+ |
| FN local R4a (valley) | 46 % → 35–40 % | 30–25 % | < 20 % |

Reaching parity with Lot B on R1 + R4a is realistic. Significantly **exceeding** Lot B on R4a requires the AROME / SURFEX inputs upgrade.

## Backward compatibility

The patch is strictly additive:

- `FiLMLayer(dem_ch, met_ch)` still works (no `context_dim` argument → default 0 → behavior unchanged)
- `DownscalingUNet(...)` still works without `n_regimes` argument → no embedding, no regime input
- `UNetSparseCalibrationModule.forward(x_met, x_dem)` still works without `regime`
- Pre-trained checkpoints from before this patch can still be loaded (the embedding is the only new parameter; it's only created when `n_regimes > 0`)

## Code touchpoints (for review)

| File | Change | Risk |
|---|---|---|
| `downscaling/deep_learning/model.py` | `FiLMLayer.__init__` accepts `context_dim`, `forward` accepts optional `context`. `DownscalingUNet.__init__` accepts `n_regimes` + `regime_embed_dim`, `forward` accepts optional `regime`. `build_model` exposes the new args. | Low: additive |
| `downscaling/deep_learning/sparse_calibration.py` | `UNetSparseCalibrationModule.forward` + `_predict_target` accept optional `regime` and pass through. | Low: additive |
| `downscaling/scripts/recalibrate_dl_film.py` | `BulkSencropDataset` loads regime CSV and emits `regime_idx`. CLI accepts `--regimes-csv`. | Medium: changes data flow; depends on user decision re: train/val splits and regime data source |
| `tests/` | New unit tests for FiLMLayer with/without context, model forward with/without regime. | None: pure addition |

The first two are in the present draft PR. The dataset change is deferred to a follow-up PR after the design is reviewed.

## Open questions for review

1. **Regime embed dim** — 8 OK, or push to 16/32 for room ?
2. **Frozen embedding warm-up** — worth the complexity, or just train end-to-end ?
3. **Class-balanced sampler** vs natural distribution — which gives best R4a POD ?
4. **Should regime feed only the bottleneck FiLM (deepest level)** to keep the modulation low-rank, instead of all levels ?

These are answered empirically with W&B sweeps once the patch is wired in.
