# Experiment runner

The main local runner is

```bash
python experiments/run_sweeps.py --quick --mode all
```

It writes outputs under

```text
experiments/results/quick/
```

including CSVs, a config file, a summary file, and optional plots.

## Recommended runs

### 1. Quick smoke test

```bash
python experiments/run_sweeps.py --quick --mode all
```

This should finish quickly and produce:

```text
experiments/results/quick/filter_sweep.csv
experiments/results/quick/alignment_sweep.csv
experiments/results/quick/config.json
experiments/results/quick/summary.json
experiments/results/quick/plots/
```

### 2. Main deterministic exponent sweep

```bash
python experiments/run_sweeps.py \
  --mode filters \
  --run-name filter_main \
  --dimension 300000 \
  --a-values 1.25,1.5,2.0 \
  --b-values 1.2,1.4,1.8 \
  --rho-values 1e-16,1e-12,1e-8 \
  --wd-values 0,1e-4,1e-3,1e-2 \
  --n-min 10 \
  --n-max 10000 \
  --num-n 50
```

This tests the predicted slopes for:

- SGD learned-mode count: approximately `1/a`;
- Adam/RMSProp learned-mode count before damping: approximately `2/a`;
- SGD bias: approximately `-(b-1)/a`;
- Adam/RMSProp bias before damping: approximately `-2(b-1)/a` in the clean source regime, saturating at `-1` in the smooth-source regime;
- AdamW saturation after the weight-decay horizon.

### 3. Coordinate-alignment sweep

```bash
python experiments/run_sweeps.py \
  --mode alignment \
  --run-name alignment_main \
  --alignment-dimensions 256,512,1024 \
  --a-values 1.25,1.5,2.0 \
  --rho-values 1e-12,1e-8 \
  --include-random
```

This tests the coordinate-alignment theorem:

- aligned coordinates should show slope approximately `-a/2` for the Adam/RMSProp effective spectrum;
- flat/Hadamard coordinates should show slope approximately `-a`, i.e. no exponent improvement;
- random orthogonal coordinates should usually be closer to the flat/SGD-like behavior than the aligned behavior.

### 4. Band-limited versus global Gaussian features

```bash
python experiments/bandlimited_feature_alignment.py \
  --dimension 2048 \
  --a 1.5 \
  --rho 1e-10 \
  --row-fraction 0.5
```

This tests the feature-map distinction from `notes/14_bandlimited_gaussian_features.md`:

- aligned/eigenbasis features should show effective slope approximately `-a/2`;
- band-limited Gaussian features should also show approximately `-a/2`;
- global Gaussian features should be closer to `-a` when coordinate variances are sufficiently flat.

## What to send back

After running, please send back:

```text
experiments/results/<run_name>/filter_sweep.csv
experiments/results/<run_name>/alignment_sweep.csv
experiments/results/<run_name>/summary.json
```

and any plots under

```text
experiments/results/<run_name>/plots/
```

For the band-limited feature experiment, paste the terminal output or redirect it to a file:

```bash
python experiments/bandlimited_feature_alignment.py --dimension 2048 > experiments/results/bandlimited_feature_alignment.txt
```

The most useful columns are the observed/predicted slope pairs, for example:

```text
adam_count_slope_obs, adam_count_slope_pred
adam_bias_slope_obs, adam_bias_slope_pred
adamw_count_slope_obs, adamw_count_slope_pred
aligned_slope_obs, aligned_slope_pred
flat_slope_obs, flat_slope_pred
random_slope_obs, random_slope_pred
```
