# DANDI:000017 export (Steinmetz et al. 2019)

Plain-table (CSV / Parquet) export of
[DANDI:000017 v0.240329.1926](https://dandiarchive.org/dandiset/000017/0.240329.1926),
"Distributed coding of choice, action and engagement across the mouse brain"
(Steinmetz, Zatka-Haas, Carandini & Harris, *Nature* 2019). 39 sessions,
10 mice, Neuropixels recordings from ~70 brain regions during a visual
two-alternative contrast discrimination task. The regions include **ZI** (4 sessions) and
**LH** (2 sessions), so the spontaneous firing rates can be compared with
the values extracted from the literature in `otto_reextraction/`.

## Reproduce

```bash
pip install "dandi>=0.74.0" pynwb pandas pyarrow
dandi download DANDI:000017/0.240329.1926        # ~14.7 GB -> ./000017/
python dandi_export/export_000017.py 000017 dandi_export/output
```

`output/tables/*.csv` is committed. The large Parquet files (`spikes/`,
`behavior/`, `trial_unit_counts.parquet`, several GB) are gitignored, so
re-run the script to regenerate them.

## Files

### `tables/sessions.csv`: one row per session
`session_id` (`<mouse>_<date>`, the key used everywhere), subject metadata
(`sex`, `age`, `genotype`, `strain`), `duration_s`, `n_trials`, `n_units`,
`n_good_units`, `regions` (`;`-separated), `spontaneous_duration_s`.

### `tables/units.csv`: one row per sorted unit
| column | meaning |
|---|---|
| `session_id`, `unit_id` | key (join to spikes / trial counts) |
| `region` | Allen CCF acronym at the unit's peak channel (`root` = outside brain/unassigned) |
| `probe`, `ccf_ap`, `ccf_dv`, `ccf_lr` | probe and Allen CCF coordinates (µm) of the peak channel |
| `peak_channel`, `depth_um` | peak channel (1-based) and depth along probe (0 = tip) |
| `phy_annotation`, `good` | 1 = multi-unit, 2 = good, 3 = unsorted; `good` = annotation ≥ 2 (the units the paper analysed) |
| `waveform_duration_ms` | trough-to-peak width of the mean extracellular waveform, which separates narrow-spiking (putative fast-spiking interneuron) from wide-spiking units. This is **not** the same measurement as intracellular AP half-width. |
| `waveform_duration_source` | `nwb` = the published value; `recomputed_from_waveform_mean` = recomputed (trough-to-peak on the highest-amplitude channel of `waveform_mean`) because in 11 of the 39 sessions the published `waveform_duration` column is actually a copy of `cluster_depths` |
| `n_spikes`, `mean_rate_hz` | total spikes and rate over the whole recording |
| `task_rate_hz` | rate during the task block (first trial start to last trial end) |
| `spontaneous_rate_hz` | rate during the `spontaneous` intervals (no task or stimulus running) |

### `tables/trials.csv`: one row per trial
`session_id`, `trial_id`, `start_time`, `stop_time`, `included` (paper's
inclusion flag), `visual_stimulus_time`, `visual_stimulus_left_contrast`,
`visual_stimulus_right_contrast` (0–1), `go_cue`, `response_time`,
`response_choice` (+1 = left choice, i.e. correct for a left stimulus;
−1 = right choice; 0 = no-go),
`feedback_time`, `feedback_type` (1 = reward, −1 = white noise), `rep_num`.
All times are seconds from session start, on the same clock as the spikes.

### `tables/region_summary.csv`: per region (good units, `root` excluded)
`n_units`, `n_sessions`, `spont_rate_mean_hz`, `spont_rate_sd_hz`,
`spont_rate_median_hz`, `mean_rate_mean_hz`, `waveform_duration_ms_median`.

### `trial_unit_counts.parquet`: spike counts per trial × unit
`session_id`, `trial_id`, `unit_id`, and spike counts in four windows:

| column | window |
|---|---|
| `count_prestim` | −0.5 to 0 s before visual stimulus onset |
| `count_stim` | 0 to 0.25 s after stimulus onset |
| `count_premove` | −0.25 to 0 s before `response_time` |
| `count_feedback` | 0 to 0.5 s after feedback |

Divide by window length to get a rate in Hz. Join with `units.csv` for region
and `trials.csv` for stimulus/choice.

### `spikes/<session_id>.parquet`: every spike
`unit_id`, `time` (s), `amp` (template amplitude), `depth` (µm along probe).

### `behavior/<session_id>__<name>.parquet`
`wheel_position` (downsampled 2500 → 100 Hz), `wheel_moves` (start/stop,
`type` 0 = flinch/unclassified, 1 = left, 2 = right), `lick_times`,
`eye_area`, `eye_xy_positions`, `face_motion_energy`. The raw lick-piezo
voltage is omitted because `lick_times` is the extracted signal.

## Quick start

```python
import pandas as pd
units  = pd.read_csv("dandi_export/output/tables/units.csv")
trials = pd.read_csv("dandi_export/output/tables/trials.csv")
zi = units[(units.region == "ZI") & units.good]
print(zi.spontaneous_rate_hz.describe())

counts = pd.read_parquet("dandi_export/output/trial_unit_counts.parquet")
stim_rate = counts.merge(units, on=["session_id", "unit_id"]) \
                  .assign(rate=lambda d: d.count_stim / 0.25)
```
