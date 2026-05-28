# GeoFeat Mainline Framework Diagram

This diagram matches the converged mainline in [src/model/GeoFeatModel.py](src/model/GeoFeatModel.py): `train.py` is the only supported training entrypoint, and `GCDR` is the fixed descriptor refiner path.

```mermaid
flowchart LR
    I[Input Image] --> B[Backbone: Standard or RepVGG]
    B --> F[FPN-like Multi-Scale Fusion x3 x4 x5]

    F --> KH[Keypoint Head]
    F --> DH[Descriptor Head]
    F --> GH[Geo Head]

    GH --> GP{Geo Preprocess Mode}
    GP -->|unfold| GU[Unfold 8x8 to channel tokens]
    GP -->|no_unfold_lite| GN[Resize geo map to descriptor resolution]

    DH --> DE[Descriptor Encoder]
    GU --> GE[Per-branch Geo Encoders]
    GN --> GE
    GE --> ADD1((Add))
    DE --> ADD1

    ADD1 --> GCDR[GCDR Pre-Attention Refinement]
    GCDR --> AFT[AttentionalNN with AFT Blocks]
    AFT --> NORM[L2 Normalize]
    NORM --> DOUT[Descriptor Output 64D]

    NORM --> REP[Repeatability Head]
    REP --> PROJ[1x1 Projection to keypoint dim]
    KH --> MIX[Keypoint Logits Fusion with alpha]
    PROJ --> MIX
    MIX --> KOUT[Refined Keypoint Logits]

    style GCDR fill:#ffe9c6,stroke:#333,stroke-width:1.5px
    style AFT fill:#dceeff,stroke:#333,stroke-width:1.2px
```

## GCDR Internal Expansion

```mermaid
flowchart TD
    D0[Descriptor map] --> D1[Convert NHWC to NCHW]
    G0[Geo map] --> G1[Geo Projection 1x1 Conv + BN + GELU]

    G1 --> GF[Compressed geo feature]
    D1 --> CAT[Concat desc and geo]
    GF --> CAT

    CAT --> REL[Reliability Head]
    REL --> R[Sigmoid reliability map]
    R --> GATE[Gate with floor]

    GF --> FILM[FiLM generator]
    FILM --> MOD[Residual build]
    GATE --> OUT[Desc + residual_gain * gate * residual]
    MOD --> OUT
    OUT --> LN[LayerNorm]
    LN --> DOUT2[Refined descriptor]
    R --> ROUT[Geo reliability map]

    style REL fill:#ffe9c6,stroke:#333,stroke-width:1.2px
    style FILM fill:#ffe9c6,stroke:#333,stroke-width:1.2px
    style LN fill:#dceeff,stroke:#333,stroke-width:1.2px
```

## Mainline Config Mapping

- `MODEL.REFINER_TYPE`: current mainline uses `GCDR`.
- `MODEL.GCDR.GEO_COMPRESS`: geometry compression width.
- `MODEL.GCDR.RELIABILITY_HIDDEN`: hidden width of the reliability head.
- `MODEL.GCDR.GATE_FLOOR`, `RESIDUAL_GAIN`, `RESIDUAL_GATE_FLOOR`: residual injection strength and lower bounds.
- `LOSS.RELIABILITY_LOSS_WEIGHT`, `RELIABILITY_CONSISTENCY_WEIGHT`, `RELIABILITY_TARGET_SCALE`: reliability supervision terms used by the training loss.
