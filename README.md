# VLoLo: Variable-Wise Memory Networks with Local Features and Long-Term Periodic Patterns for Time-Series Anomaly Detection

## 📖 Introduction
In reconstruction-based time-series anomaly detection, memory-based approaches suppress excessive reconstruction of anomalous data by learning representative normal patterns as memory items and reconstructing inputs by referencing similar items. 
However, existing methods struggle to capture subtle and abrupt changes, trends, and variations in amplitude and time scales. 
Furthermore, low-frequency periodicities are often inadequately learned because they are dominated by high-frequency periodicities. 
These limitations can lead to false positives or missed detections. 
In multivariate time series, aggregation into a single embedding often results in the loss of variable-specific characteristics and intervariable relationships. 
To address these issues, we propose VLoLo, an anomaly detection method that integrates local-feature and long-term periodic memory networks for each variable. 
The local-feature memory network facilitates learning and reproduction of local temporal characteristics by incorporating engineered features and an intervariable attention mechanism. 
The long-term periodic memory network efficiently learns multiperiodic structures, including low-frequency periodicities, from a long time series that preserves global shape characteristics. 
Experiments on univariate and multivariate datasets from diverse domains demonstrate the effectiveness of each network and show that their integration achieves accurate reconstruction of normal time series while increasing reconstruction errors for anomalies.

### 🗂 Dataset Setup
- **SMD, MSL, SMAP, SWaT, PSM**: Download from [Google Drive](https://drive.google.com/drive/folders/1RaIJQ8esoWuhyphhmMaH-VCDh-WIluRR) and place into `../datasets/` folder.
- **UCR Anomaly Archive**: Download from [GitHub repository](https://github.com/thuml/Large-Time-Series-Model/tree/main/scripts/anomaly_detection) and place into `../datasets/UCR_Anomaly/` folder.

### 🚀 Training and Evaluation

Run the shell scripts in the `scripts/` folder for each dataset:
```bash
# SMD
bash scripts/run_smd.sh

# MSL
bash scripts/run_msl.sh

# SMAP
bash scripts/run_smap.sh

# PSM
bash scripts/run_psm.sh

# SWaT
bash scripts/run_swat.sh

# UCR Anomaly Archive
bash scripts/run_ucr.sh
```

Each script runs the full pipeline: First Step Training → Second Step Training → Testing.
