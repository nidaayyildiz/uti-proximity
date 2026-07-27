# Proximity
 
> A social-group detection **package built for [NovaVision](https://github.com/novavision-ai)**.
 
## Overview
 
**Proximity** is a NovaVision component that detects social groups from a video stream. It takes person detections, pairwise distances, and facial-analysis data, then classifies groups of people into **Family**, **Parent-Child**, **Couple**, or **Friend Group** based on how close they are, how long they stay together, their age/gender profile, and optional pose (keypoint) signals.
 
## Pipeline
 
- **Inputs** — `inputDistances` (from `MeasureDistance`), `inputPersons` (pose estimation), `inputFacialAnalysis` (age & gender).
- **Output** — `outputGroups`: one detection per group, with a bounding box, a class label (`family`, `parent_child`, `couple`, `friend_group`), and a member count.
## How It Works
 
1. Build a person registry (keyed by tracker ID, or bbox centroid as fallback).
2. Enrich each person with age/gender from facial analysis.
3. Find pairs within the distance threshold and track how long they stay close.
4. Group stable pairs via union-find (connected components).
5. Classify each group by the enabled relationship rules.
## Classification
 
| Type | Rule (summary) |
|------|----------------|
| **Family** | Min. children + adults, with an adult-gender condition. |
| **Parent-Child** | Small group (≤3) with at least one child and one adult. |
| **Couple** | A pair — via time-together + formation, or a pose-based score (>0.6). |
| **Friend Group** | Similar-height members above a minimum size. |
 
## Key Configs
 
| Config | Default | Description |
|--------|---------|-------------|
| `configProximityThresholdCm` | 150.0 | Max distance (cm) to count as proximate. |
| `configProximityDurationSec` | 3.0 | Min seconds together to form a stable group. |
| `configFamilyDetection` | — | Toggle Family classification. |
| `configParentChildDetection` | — | Toggle Parent-Child classification. |
| `configCoupleDetection` | — | Toggle Couple classification. |
| `configFriendGroupDetection` | — | Toggle Friend Group classification. |
 
## Structure
 
```
src/
├── executors/SocialGroup.py   # Executor: proximity tracking, grouping, run loop
├── models/PackageModel.py     # Pydantic schemas (inputs, outputs, configs)
└── utils/
    ├── classifier.py          # Relationship classification & pose scoring
    └── response.py            # Response builder
```

## License
 
[MIT](LICENSE)
