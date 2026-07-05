I just make a readme so what you think:# Representative Driving Cycle Construction from VED Data

This repository contains a Python pipeline for constructing a representative driving cycle from real-world vehicle speed-time data. The pipeline follows a microtrip-based workflow with 1 Hz resampling, acceleration cleaning, idle/moving-state detection, microtrip segmentation, feature extraction, PCA, K-means clustering, and optimized microtrip subset selection.

The purpose of the project is to convert raw vehicle trajectory data into a compact speed-time driving cycle that preserves the main driving characteristics of the original trip, including mean speed, moving mean speed, idle time, acceleration/deceleration behavior, and speed-bin distribution.

## Project summary

Driving cycles are used in vehicle energy analysis, emissions estimation, powertrain evaluation, and traffic-behavior studies. A useful driving cycle should be shorter than the original raw driving record, but still representative of the original driving behavior.

This repository implements a full end-to-end construction pipeline:

```text
Raw VED-style CSV
→ column standardization
→ 1 Hz resampling
→ acceleration cleaning
→ idle and moving-state detection
→ microtrip segmentation
→ feature extraction
→ PCA
→ K-means clustering
→ optimized microtrip subset selection
→ final representative driving cycle
```

The current demonstration constructs an optimized 1198-second representative driving cycle while targeting a 1200-second final cycle.

---

## Dataset and attribution

This project uses data in the format of the Vehicle Energy Dataset (VED).

The Vehicle Energy Dataset was introduced by G. S. Oh, D. J. LeBlanc, and H. Peng as a large-scale real-world vehicle energy dataset. It contains GPS trajectories and time-series vehicle signals such as speed, fuel/energy use, and auxiliary power usage collected from personal vehicles in Ann Arbor, Michigan.

Original VED dataset repository:

https://github.com/gsoh/VED

Original VED dataset paper:

G. S. Oh, D. J. LeBlanc, and H. Peng,  
**Vehicle Energy Dataset (VED), A Large-scale Dataset for Vehicle Energy Consumption Research**  
arXiv:1905.02081  
https://arxiv.org/abs/1905.02081

Important dataset note:

This repository does not claim ownership of the VED dataset. Raw VED data is not included in this repository. Users should download the dataset from the official VED source and cite the original VED paper when using the data.

---

## Input data format

The pipeline expects a VED-style CSV file with the following columns:

```text
VehId
Trip
Timestamp(s)
Latitude[deg]
Longitude[deg]
Vehicle Speed[km/h]
elevation[m]
```

Example local input path:

```text
data/cycle_v1.csv
```

Raw data files are intentionally not tracked in this repository. The `.gitignore` file excludes local CSV data and generated output folders.

---

## Methodology

### 1. Column standardization

The input CSV is first loaded and the VED column names are mapped into simpler internal names.

```text
VehId                 → veh_id
Trip                  → trip_id
Timestamp(s)          → time_s
Latitude[deg]         → lat
Longitude[deg]        → lon
Vehicle Speed[km/h]   → speed_kmh
elevation[m]          → elevation_m
```

Rows with missing or non-numeric required values are removed. The data is then sorted by vehicle ID, trip ID, and timestamp.

### 2. Resampling to 1 Hz

Raw vehicle data can have irregular time intervals. For drive-cycle construction, a uniform time base is easier to work with and gives cleaner acceleration estimates.

Each trip is resampled to 1 Hz using linear interpolation. After this step, each row represents one second of driving.

### 3. Acceleration cleaning

Acceleration is calculated from the resampled speed profile. Basic physical limits are then applied to reduce unrealistic spikes.

The default limits are:

```text
maximum acceleration:  +4.0 m/s²
maximum deceleration:  -8.0 m/s²
```

This step is not meant to over-smooth the trip. It only corrects points that exceed the defined acceleration bounds.

### 4. Idle and moving-state detection

A point is treated as idle when:

```text
speed <= 1 km/h
```

This is more robust than checking for exact zero speed, because real vehicle speed signals may contain small noise around zero.

The full trip is then split into continuous idle and moving runs.

### 5. Microtrip segmentation

A microtrip is defined as:

```text
one moving run + the following idle run
```

This keeps each driving fragment connected to the stop/idle period that follows it.

Very small or uninformative fragments are removed using basic filters:

```text
minimum microtrip duration: 10 s
maximum idle duration inside a microtrip: 180 s
minimum maximum speed: 10 km/h
minimum distance: 0.01 km
```

### 6. Feature extraction

Each valid microtrip is represented using a set of driving-behavior features.

The extracted features include:

```text
duration
distance
maximum speed
mean speed
mean moving speed
speed standard deviation
idle percentage
acceleration percentage
deceleration percentage
cruise percentage
mean positive acceleration
mean absolute deceleration
acceleration standard deviation
speed-bin percentages
```

The driving-mode percentages are mutually exclusive:

```text
idle + acceleration + deceleration + cruise = 100%
```

Speed-bin percentages are calculated using the following bins:

```text
0–10 km/h
10–20 km/h
20–30 km/h
30–40 km/h
40–50 km/h
50–60 km/h
60–70 km/h
70–80 km/h
80+ km/h
```

### 7. PCA and K-means clustering

The microtrip feature table is standardized and reduced using Principal Component Analysis (PCA). K-means clustering is then applied in PCA space.

For the current implementation, the default number of clusters is three:

```text
low_speed
medium_speed
high_speed
```

Cluster names are assigned based on the average moving speed of each cluster.

### 8. Optimized microtrip subset selection

The final representative cycle is constructed by selecting a subset of valid microtrips.

The optimizer tries to match the following target characteristics:

```text
target duration
mean speed
mean moving speed
speed standard deviation
idle percentage
acceleration percentage
deceleration percentage
cruise percentage
speed-bin distribution
cluster duration proportions
mean positive acceleration
mean absolute deceleration
```

For small candidate sets, the script uses exhaustive subset search. This checks all valid combinations within the target duration window and selects the subset with the lowest score.

For larger candidate sets, the script can switch to random subset search.

### 9. Final cycle stitching

The selected microtrips are stitched together in chronological order. A short initial idle segment is added so the final driving cycle starts from rest.

After stitching, transition acceleration is checked and cleaned again using the same acceleration bounds.

The final output is a clean representative speed-time driving cycle.

---

## How to run

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python drive_cycle_pipeline.py --input data/cycle_v1.csv --output outputs --target-duration 1200
```

Useful optional arguments:

```bash
python drive_cycle_pipeline.py \
  --input data/cycle_v1.csv \
  --output outputs \
  --target-duration 1200 \
  --idle-speed-kmh 1.0 \
  --n-clusters 3 \
  --n-pcs 4
```

If the main Python file has a different name in the repository, update the command accordingly.

---

## Main output files

The pipeline writes outputs to the selected output folder.

Important output files:

```text
FINAL_optimized_representative_drive_cycle.csv
```

Final simplified driving cycle with time, speed, acceleration, and source microtrip labels.

```text
optimized_drive_cycle_full.csv
```

Full final cycle with additional metadata, original timestamp references, source microtrip IDs, and cluster labels.

```text
key_feature_comparison.csv
```

Comparison between target driving characteristics and the optimized constructed cycle.

```text
optimized_selected_microtrips.csv
```

Microtrips selected for the final representative cycle.

```text
microtrip_features.csv
```

Feature table for all valid microtrips.

```text
microtrip_clusters.csv
```

Microtrip feature table with PCA coordinates and K-means cluster labels.

```text
cluster_summary.csv
```

Summary of low-speed, medium-speed, and high-speed clusters.

```text
top_search_results.csv
```

Best candidate microtrip subsets ranked by optimization score.

```text
run_summary.json
```

Compact summary of the full pipeline run.

---

## Key result

The optimized cycle closely matches the target driving statistics.

| Feature | Target | Optimized constructed | Absolute error | Percent error |
|---|---:|---:|---:|---:|
| Duration (s) | 1200.000 | 1198.000 | -2.000 | -0.167% |
| Distance (km) | 8.762 | 8.726 | -0.037 | -0.417% |
| Maximum speed (km/h) | 74.650 | 72.900 | -1.750 | -2.344% |
| Mean speed (km/h) | 26.286 | 26.220 | -0.066 | -0.251% |
| Mean moving speed (km/h) | 34.977 | 34.968 | -0.009 | -0.024% |
| Speed standard deviation (km/h) | 22.789 | 22.554 | -0.234 | -1.029% |
| Idle time (%) | 24.875 | 25.042 | 0.167 | 0.670% |
| Acceleration time (%) | 29.373 | 29.633 | 0.260 | 0.886% |
| Deceleration time (%) | 28.373 | 28.631 | 0.258 | 0.909% |
| Cruise time (%) | 17.379 | 16.694 | -0.685 | -3.940% |
| Mean positive acceleration (m/s²) | 0.674 | 0.678 | 0.004 | 0.607% |
| Mean absolute deceleration (m/s²) | 0.674 | 0.673 | -0.001 | -0.083% |
| Acceleration standard deviation (m/s²) | 0.716 | 0.706 | -0.010 | -1.453% |

Compact result summary:

```text
target duration: 1200 s
final duration: 1198 s
distance error: 0.42%
mean speed error: 0.25%
moving mean speed error: 0.02%
idle percentage error: 0.67%
acceleration percentage error: 0.89%
deceleration percentage error: 0.91%
```

---

## Optimized constructed driving cycle

The final representative cycle is a 1198-second speed-time profile. It preserves the main stop-and-go, medium-speed, and higher-speed driving portions while compressing the original trip into a shorter representative cycle.

To show this figure in the README, place the image at:

```text
assets/optimized_constructed_driving_cycle.png
```

Then this Markdown image link will display it:

![Optimized constructed driving cycle](assets/optimized_constructed_driving_cycle.png)

---

## Speed-bin distribution comparison

The speed-bin comparison shows how the constructed cycle matches the original target distribution across different speed ranges.

The low-speed and higher-speed ranges are captured reasonably well. The main remaining mismatch appears around the 20–50 km/h range, where the optimized cycle slightly overrepresents 20–40 km/h and underrepresents 40–50 km/h. This is expected in a single-trip demonstration with a limited number of available candidate microtrips.

To show this figure in the README, place the image at:

```text
assets/speed_bin_distribution.png
```

Then this Markdown image link will display it:

![Speed-bin distribution comparison](assets/speed_bin_distribution.png)

---

## Recommended repository structure

```text
representative-drive-cycle-kmeans-ved/
│
├── drive_cycle_pipeline.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── assets/
│   ├── optimized_constructed_driving_cycle.png
│   └── speed_bin_distribution.png
│
├── results/
│   ├── key_feature_comparison.csv
│   ├── optimized_selected_microtrips.csv
│   └── run_summary.json
│
└── data/
    └── README.md
```

The `data/` folder is intended for local input files only. Raw VED data is not included in this repository.

---

## Notes on uploaded results

Only the most important result files should be uploaded to the repository.

Recommended result files:

```text
results/key_feature_comparison.csv
results/optimized_selected_microtrips.csv
results/run_summary.json
```

Recommended image files:

```text
assets/optimized_constructed_driving_cycle.png
assets/speed_bin_distribution.png
```

Intermediate files such as full cleaned time-series data, all phase-wise outputs, and raw input CSV files are not required in the repository.

---

## Scope and limitations

This repository is intended as a reproducible research-code pipeline for representative driving-cycle construction.

Current limitations:

- The demonstration result is based on one VED-style trip file.
- The method is designed to scale to more trips, where microtrip diversity and cluster stability should improve.
- The optimizer uses exhaustive subset search for small candidate sets and random subset search for larger candidate sets.
- This repository does not implement AMPSO.
- GPS coordinates are included in the input structure but are not map-matched.
- The final cycle is a research/analysis cycle, not an official regulatory driving cycle.
- Output quality depends on input trip diversity, selected thresholds, and target duration.

---

## Relation to previous driving-cycle methods

This implementation follows the general idea of microtrip-based representative driving-cycle construction.

The Fuzhou driving-cycle study used K-means clustering and AMPSO for representative driving-cycle development. This repository uses a related clustering-based structure, but the final selection step is implemented using exhaustive subset optimization for small candidate sets.

This choice is intentional. For the current demonstration size, exhaustive subset search is direct, reproducible, and sufficient.

---

## References

1. G. S. Oh, D. J. LeBlanc, and H. Peng,  
   **Vehicle Energy Dataset (VED), A Large-scale Dataset for Vehicle Energy Consumption Research**  
   arXiv:1905.02081  
   https://arxiv.org/abs/1905.02081

2. Vehicle Energy Dataset official repository  
   https://github.com/gsoh/VED

3. Minrui Zhao, Hongni Gao, Qi Han, Jiaang Ge, Wei Wang, and Jue Qu,  
   **Development of a Driving Cycle for Fuzhou Using K-Means and AMPSO**  
   Journal of Advanced Transportation, 2021  
   DOI: 10.1155/2021/5430137  
   https://onlinelibrary.wiley.com/doi/10.1155/2021/5430137

4. Yongjiang He,  
   **Research on the construction method of vehicle driving cycle based on Mean Shift clustering**  
   arXiv:2008.05070  
   https://arxiv.org/abs/2008.05070

---

## License

The code in this repository is released under the MIT License.

The VED dataset is not owned by this repository. Please follow the original dataset license and citation requirements from the official VED source.
