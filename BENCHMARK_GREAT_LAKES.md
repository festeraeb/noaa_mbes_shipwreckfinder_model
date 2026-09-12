# Great Lakes Real-Data Evaluation & ONNX Tuned Pipeline

## Overview
This contribution provides:
1. **ONNX Inference & Hillshade Configuration**: An optimized dual-channel configuration combining normalized depth with 315°/45° hillshade relief, suitable for fast cross-platform deployment.
2. **Real-World Great Lakes Benchmark**: Head-to-head testing on real NOAA NCEI Lake Michigan multibeam bathymetry (`H13941_MB_1m_LWD_1of2.bag`, 462M soundings).

### Real Soundings Benchmark Results (Lake Michigan H13941)

| Detection ID | Coordinates (Lat, Lon) | Anomaly Type | Height Above Floor | UNet Hillshade Max Prob | False Positive Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `H13941_phys000` | 41.7299, -86.9142 | Physical Wreck Target | **+3.45m** | High relief | Detected |
| `H13941_mask065` | 41.7394, -86.8901 | Masked Flat Texture Break | -1.15m | 0.0204 | Correctly Filtered Out |
| `H13941_mask1074` | 41.7009, -87.0179 | Masked Flat Texture Break | -0.47m | 0.0204 | Correctly Filtered Out |
| `H13941_mask294` | 41.7610, -86.9789 | Masked Flat Texture Break | +0.39m | 0.0204 | Correctly Filtered Out |

### Key Finding
The hillshade relief channel acts as an exceptional false-positive suppressor against flat bathymetric anomalies and boundary redactions. While raw edge detectors and texture breaks trigger on masked boundaries, the UNet Hillshade model remains selectively tuned to true 3D superstructure relief.

---

> [!TIP]
> **Note to UM Field Robotics team**: If you are interested in the NOAA masked / redaction reconstruction dataset or curvelet edge fusion benchmarks for training against sanitized bathymetry, please reach out! :-D
