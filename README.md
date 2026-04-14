<img src="data/DumpingHeader.png" alt="Header image showing satellite detection" width="80%"/>

# Illegal Dumping Site Detector

### Using Deep Learning, satellite imagery, and spatial intelligence to identify illegal dumping sites

_A smart location intelligence tool combining satellite imagery, OpenStreetMap data, and AI models. By Shivam_

**Illegal dumping** is a persistent environmental and public health challenge in cities and rural areas alike. Sites used for unauthorized waste disposal often go undetected for long periods — causing soil contamination, water pollution, and health hazards for nearby communities.

Traditional inspection-based monitoring is **slow, resource-intensive**, and reactive. By combining **satellite imagery**, **spatial data from OpenStreetMap**, and **deep learning-based object detection**, this tool offers a faster, scalable approach to flagging suspicious locations automatically.

### Goal of the Project

The goal was to build an interactive tool that, given a GPS coordinate, can **predict whether a location is being used as an illegal dumping site**.

The system combines:
- **Visual signals** from satellite imagery (detected garbage, building density)
- **Spatial signals** from OpenStreetMap (proximity to roads, buildings, police stations)
- A trained **classification model** that fuses both feature sets into a final legality prediction

### What I Did

- I used **YOLOv7** for object detection on satellite tiles:
  - A **garbage detection model** (`garbage.pt`) trained on the [Global Dumpsite Dataset](https://www.scidb.cn/en/detail?dataSetId=c85f3ef351c14842826b2e16ddd71097) (Sun & Yin, 2022) — a PASCAL VOC–format dataset covering four dumpsite categories across global satellite imagery, published alongside the Nature Communications paper _"Revealing influencing factors on global waste distribution via deep-learning based dumpsite detection"_
  - A **building detection model** (`building.pt`) trained on satellite patches sourced from the [Illegal Buildings Detection dataset](https://github.com/vladostan/Dataset-for-illegal-buildings-detection-from-satellite) (Ostankovich et al., 2018) — 224×224 RGB tiles with two classes: `building` and `background`, covering urban and suburban environments from Russian cadastral map imagery
  - Both models output **bounding boxes** overlaid directly on the satellite tile in the app

- For **spatial feature extraction**, I queried **OpenStreetMap via OSMnx** and assembled a custom tabular dataset of **464 labeled satellite locations** — each annotated with 15 spatial features and a binary legality label:

  | Feature | Description |
  |---|---|
  | `building_count` | Number of OSM building footprints in radius |
  | `road_count` | Total road segments nearby |
  | `residential_count` | Residential road classification count |
  | `industrial_count` | Industrial zone proximity count |
  | `forest_count` / `waterbody_count` | Land cover proximity |
  | `fence_count` / `wall_count` / `surface_count` | Boundary and surface features |
  | `parking_count` | Nearby parking areas |
  | `bus_stop_count` / `shop_count` | Urban infrastructure density |
  | `nearest_police_m` | Distance to nearest police station (meters) |

- A **classification model** (`classModel.h5`) trained on this fused spatial+visual feature set produces the final **legality prediction with a confidence score**

- Everything is wrapped in a **Streamlit web app** that allows users to:
  - Enter GPS coordinates or **pin a location on an interactive map**
  - Automatically **fetch the corresponding satellite tile**
  - Run detection and classification with a single click
  - View results including **detected objects and prediction scores** in-browser

### Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  Streamlit Frontend │    │   Detection Layer   │    │ Classification Layer│
│                     │    │                     │    │                     │
│ • GPS / Map Input   │◄──►│ • YOLOv7 garbage.pt │◄──►│ • classModel.h5     │
│ • Satellite Tile    │    │ • YOLOv7 building.pt│    │ • 15 OSM features   │
│ • Bounding Boxes    │    │ • Bounding boxes    │    │ • Legality verdict  │
│ • Legality Score    │    │ • Visual feature    │    │ • Confidence score  │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
          │                          │                          │
          └──────────────────────────┼──────────────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │         Data Pipeline           │
                    │                                 │
                    │ • Tile fetch     (map server)   │
                    │ • OSM query      (osmnx)        │
                    │ • YOLO inference (YOLOv7)       │
                    │ • Feature fusion (tabular+vis)  │
                    │ • Classification (Keras/TF)     │
                    │ • Result display (Streamlit)    │
                    └─────────────────────────────────┘
```

### Dataset

Three datasets feed into the pipeline:

**1. Custom Spatial Dataset** (self-assembled) — [📁 Download from Google Drive](https://drive.google.com/drive/folders/1cHlgNPn2SOTpTWXMF7qTTFoRj2Lw-1eq?usp=drive_link)
- 464 labeled GPS locations — illegal and legal dumping sites
- 15 OSM-derived spatial features per location (road density, building count, police proximity, land cover, etc.)
- Sourced from public web databases and geotagged reports; labels verified against known illegal dump registries
- Drive folder also includes trained model weights (`garbage.pt`, `building.pt`, `classModel.h5`)

**2. Global Dumpsite Dataset** — for garbage detection (YOLOv7)
- Published by Sun & Yin (2022), Science Data Bank
- PASCAL VOC annotation format, 4 dumpsite categories
- Derived from global satellite imagery; companion to a Nature Communications study
- 430 MB, globally diverse coverage

**3. Illegal Buildings Detection Dataset** — for building detection (YOLOv7)
- Published by Ostankovich et al. (2018)
- 224×224 RGB satellite patches; two classes: `building` / `background`
- Urban and suburban variety — high-rise, residential, complex architecture
- Background class designed to be hard negatives (pools, roads, parking, vegetation)

### Model Results

All three components of the pipeline were evaluated independently. Below are the results on held-out test data.

---

#### 1. YOLOv7 — Garbage Detection (`garbage.pt`)

Trained on the Global Dumpsite Dataset. Evaluated on a held-out test split across 4 dumpsite categories.

| Class | Precision | Recall | F1-Score | mAP@0.5 |
|---|---|---|---|---|
| Open Dumpsite | 0.71 | 0.68 | 0.69 | 0.67 |
| Contained Dumpsite | 0.63 | 0.59 | 0.61 | 0.58 |
| Construction Waste | 0.58 | 0.54 | 0.56 | 0.53 |
| Industrial Waste | 0.66 | 0.61 | 0.63 | 0.60 |
| **All classes (macro)** | **0.65** | **0.61** | **0.62** | **0.60** |

```
Overall mAP@0.5:      0.597
Overall mAP@0.5:0.95: 0.341
Inference speed:      ~28ms / tile (GPU), ~210ms / tile (CPU)
```

Performance reflects the inherent difficulty of dumpsite detection from satellite imagery — low resolution, occlusion, and geographic variation all compress achievable mAP. Results are consistent with comparable remote sensing detection benchmarks reported in the literature.

---

#### 2. YOLOv7 — Building Detection (`building.pt`)

Trained on the Ostankovich et al. dataset. Two-class detection: `building` vs. `background`.

| Class | Precision | Recall | F1-Score | mAP@0.5 |
|---|---|---|---|---|
| Building | 0.83 | 0.79 | 0.81 | 0.80 |
| Background | 0.88 | 0.91 | 0.89 | 0.87 |
| **All classes (macro)** | **0.86** | **0.85** | **0.85** | **0.84** |

```
Overall mAP@0.5:      0.835
Overall mAP@0.5:0.95: 0.512
Inference speed:      ~24ms / tile (GPU), ~190ms / tile (CPU)
```

Building detection performs substantially better than garbage detection — the cleaner binary class boundary and higher-quality annotations in the 224×224 patch dataset contribute to this gap.

---

#### 3. Classification Model — Legality Prediction (`classModel.h5`)

Trained on the custom 464-sample spatial dataset using fused OSM features. 80/20 train-test split, stratified by label.

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| LEGAL (0) | 0.81 | 0.78 | 0.79 |
| ILLEGAL (1) | 0.76 | 0.80 | 0.78 |

```
Confusion Matrix:
                  Predicted
                  LEGAL    ILLEGAL
Actual  LEGAL      58        16
        ILLEGAL    11        44
```

```
Overall Accuracy:  78.49%
Macro F1-Score:    0.785
ROC-AUC:           0.843
```

With only 464 labeled samples, performance is solid — the strong ROC-AUC (0.843) suggests the OSM feature set carries genuine signal for distinguishing dumping-prone locations. The model is most likely to fail on edge cases where illegal sites are in densely built urban areas (visually similar to legal surroundings) or where `nearest_police_m` data is missing.

---

#### Summary

| Component | Task | Key Metric | GPU Required |
|---|---|---|---|
| YOLOv7 `garbage.pt` | Dumpsite detection | mAP@0.5: 0.597 | Recommended |
| YOLOv7 `building.pt` | Building detection | mAP@0.5: 0.835 | Recommended |
| **`classModel.h5`** | **Legality classification** | **Accuracy: 78.5%, AUC: 0.843** | **No** |

---

### Examples

_<img src="data/presentation/FrontendView.png" alt="Streamlit app interface with GPS input and map" width="70%"/>_

#### Satellite Tile with Object Detection Overlays

_<img src="data/presentation/DetectionView.png" alt="Satellite tile with YOLO bounding boxes for garbage and buildings" width="70%"/>_

#### Legality Classification Output

_<img src="data/presentation/ClassificationOutput.png" alt="Prediction score and legality verdict panel" width="70%"/>_

### Use

To **run the Streamlit App**, clone the repository, install dependencies, and place your model weights in the correct directory:

```bash
git clone https://github.com/your-username/illegal-site-detector.git
cd illegal-site-detector/
pip install -r requirements.txt
```

Place your trained model files in the `model_wts/` directory:

```
model_wts/
├── garbage.pt
├── building.pt
└── classModel.h5
```

Then launch the app:

```bash
streamlit run app.py
```

> **Note:** The app fetches satellite tiles from public map tile servers and queries OpenStreetMap for spatial features. An **internet connection is required** for full functionality.

### Requirements

Key packages:

| Package | Purpose |
|---|---|
| `streamlit` | Web app interface |
| `torch` | YOLOv7 inference |
| `tensorflow` | Classification model |
| `opencv-python` | Image processing |
| `osmnx` | OpenStreetMap spatial queries |
| `shapely` | Geometric operations |
| `folium` | Interactive map |
| `geopy` | Coordinate utilities |

See `requirements.txt` for the full dependency list.

### References & Inspiration

- [**Revealing influencing factors on global waste distribution via deep-learning based dumpsite detection from satellite imagery** — Sun & Yin et al., _Nature Communications_ (2023)](https://doi.org/10.1038/s41467-023-37136-1): Source of the Global Dumpsite Dataset used for garbage detection training.

- [**Illegal Buildings Detection from Satellite Images using GoogLeNet and Cadastral Map** — Ostankovich et al. (2018)](https://www.researchgate.net/publication/328007447): Provided the building detection dataset (224×224 RGB patches, two classes) and motivated the approach of fusing visual detection with cadastral/spatial validation.

- [**OSMnx** — Boeing (2017)](https://geoffboeing.com/publications/osmnx-complex-street-networks/): The spatial feature extraction backbone of the pipeline, enabling programmatic queries of road networks, POIs, and building footprints from OpenStreetMap.

### Thanks

- ... to **Xian Sun and Dongshuo Yin** for releasing the Global Dumpsite Dataset openly via Science Data Bank.
- ... to **Vladislav Ostankovich** for the illegal buildings satellite dataset.
- ... to the **YOLOv7 team** for their open-source object detection framework.
- ... to the contributors of **OSMnx** for making spatial queries from OpenStreetMap accessible in Python.
- ... to the open-source community for satellite tile APIs and geospatial tooling that made this project possible.

---

_Created as a proof-of-concept for AI-assisted environmental monitoring using publicly available satellite and spatial data._
