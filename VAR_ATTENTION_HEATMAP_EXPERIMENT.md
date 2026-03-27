# VAR Attention Heatmap Experiment

## Goal

We run attention heatmap probes on Infinity to understand how each VAR scale, layer, and head contributes to image `contents` and `structure`.

The working hypothesis is:

- some heads mainly preserve semantic or contextual information (`content heads` / `contextual heads`)
- some heads mainly express geometric or repetitive spatial structure (`structure heads`)
- this behavior changes across VAR scales, because Infinity generates images in a coarse-to-fine autoregressive schedule

This experiment is designed to make those differences visible.

## Why this matters

We observed that when spherical RoPE is applied more aggressively:

- vertical or multi-diagonal attention patterns become stronger
- image structure can remain visible
- semantic content can collapse or drift

This suggests that different heads and different scales play different roles. To verify that, we probe attention maps directly.

## Core idea

For a chosen inference run, we extract self-attention tensors from Infinity and save heatmaps organized by:

- `scale`
- `layer`
- `head`

This lets us compare:

- coarse scale vs fine scale behavior
- early layer vs late layer behavior
- head-specific structural patterns vs content-related patterns

## Model setting

The current probe script is built around `infinity_2b`.

- model depth: `32` layers
- heads per layer: `16`
- total self-attention heads: `32 x 16 = 512`

So one scale corresponds to passing the current stage tokens through all `32` transformer layers.

## What heatmap is being saved

The probe script captures `q` and `k` after RoPE is applied, then reconstructs a head-wise attention map:

1. compute scores with `QK^T`
2. apply `softmax`
3. average over all query tokens
4. reshape the key dimension back to `(height, width)`

So each saved image is:

- one `layer`
- one `head`
- one `scale`
- showing which spatial key locations are most attended on average

Interpretation:

- brighter area -> that spatial location is referenced more strongly by that head
- vertical stripe pattern -> likely column-wise structural bias
- diagonal or multi-diagonal pattern -> likely structural / periodic spatial bias
- local compact blob -> likely local or content-sensitive attention
- diffuse map -> broader/global attention preference

## Important limitation

This is not a per-query attention visualization.

It is a query-averaged head summary. That means:

- it is good for seeing global head behavior
- it is not ideal for analyzing one specific token's attention target

Still, it is useful as a first-pass probe for contextual-vs-structural head behavior.

## Current probing script

We use:

- `tools/probe_attention_heatmap.py`

This script supports:

- selecting a condition: `baseline`, `spherical_all`, `spherical_split`
- selecting one or multiple layers
- selecting one scale, a scale range, or all scales
- saving only heatmaps (no generated image output)

## RoPE conditions used in analysis

### baseline

- standard Infinity RoPE
- no spherical patch applied

### spherical_all

- all heads use spherical-aware RoPE
- `band_ratio` controls how much of the width frequency band is spherical

### spherical_split

- only part of the heads use spherical-aware RoPE
- `head_split_ratio` controls how many heads are assigned to the spherical group
- `band_ratio` controls how much of the width frequency band inside those heads is spherical

In the current implementation:

- `head_split_ratio` works on the `head axis`
- `band_ratio` works on the `frequency-band axis` inside the selected spherical heads

So these two controls are not redundant.

## Why scale-wise probing is necessary in VAR

Infinity is a VAR model with a coarse-to-fine autoregressive schedule.

For each scale:

1. the current stage tokens are prepared
2. they pass through all `32` layers
3. outputs are decoded into codes for that scale
4. the next finer scale is generated using accumulated information

Therefore:

- the same layer can behave differently at different scales
- a head that looks contextual at coarse scale may look structural at fine scale
- content collapse may happen because the wrong heads are modified at the wrong scales

This is why scale-aware heatmap extraction is essential.

## Example commands

### 1. Last 4 layers at the finest scale

```bash
python tools/probe_attention_heatmap.py \
  --condition spherical_split \
  --layers -4 -3 -2 -1 \
  --scale finest \
  --head_split_ratio 0.25 \
  --band_ratio 0.5 \
  --output_dir probe_outputs/attn_heatmaps
```

### 2. First layer across all scales

```bash
python tools/probe_attention_heatmap.py \
  --condition spherical_split \
  --layers 0 \
  --scales all \
  --head_split_ratio 0.25 \
  --band_ratio 0.5 \
  --output_dir probe_outputs/attn_heatmaps
```

### 3. All layers at one coarse scale

```bash
python tools/probe_attention_heatmap.py \
  --condition baseline \
  --layers all \
  --scale 0 \
  --output_dir probe_outputs/attn_heatmaps
```

### 4. Selected layers across a range of scales

```bash
python tools/probe_attention_heatmap.py \
  --condition spherical_split \
  --layers 0 15 31 \
  --scale_range 0 3 \
  --head_split_ratio 0.25 \
  --band_ratio 0.5 \
  --output_dir probe_outputs/attn_heatmaps
```

## Output structure

Saved heatmaps are organized as:

```text
probe_outputs/attn_heatmaps/
  spherical_split/
    L00/
      scale00/
        head00.png
        head01.png
      scale01/
        head00.png
    L31/
      scale12/
        head15.png
```

Meaning:

- `L00` -> layer 0
- `scale00` -> first VAR scale
- `head15.png` -> head index 15 of that layer and scale

## What we are trying to discover

This experiment is meant to answer the following questions:

1. Which heads show stable local/content-preserving patterns?
2. Which heads show strong structural patterns such as vertical stripes or multi-diagonals?
3. At which scales do structural patterns become dominant?
4. Which layers become more sensitive to spherical RoPE modifications?
5. Does content collapse come from modifying the wrong heads, the wrong scales, or both?

## Working interpretation framework

### likely contextual/content-related heads

Expected signs:

- more localized or semantically selective attention
- less rigid geometric repetition
- more stable support for object or scene consistency

### likely structural heads

Expected signs:

- strong vertical stripe patterns
- strong diagonal / multi-diagonal patterns
- repetitive or globally geometric attention layouts

These are only working labels for analysis. The codebase does not currently contain a ground-truth head-role annotation.

## Current conclusion

At this stage, the experiment gives us a direct visual probe of how:

- `scale`
- `layer`
- `head`
- and RoPE condition

jointly affect Infinity's self-attention behavior.

This is the main tool we are using to diagnose why aggressive spherical RoPE settings can preserve spatial structure while harming semantic content generation.

## Next recommended extensions

1. Add query-specific heatmaps for center / seam-left / seam-right queries
2. Save raw attention arrays (`.npy`) together with PNGs
3. Compute per-head variance across prompts for contextual-vs-structural classification
4. Compare baseline vs spherical conditions for the exact same `scale-layer-head`
5. Use the resulting classification to replace the current ratio-based head split with a measured head split
