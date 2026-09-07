<<<<<<< HEAD
Geospatial Engine – Experimental Earth Data Processing Project
This repository contains an experimental geospatial engine developed to test multi‑source satellite ingestion, global mosaic generation, and cubemap reprojection workflows. The project represents substantial progress in building a full Earth‑processing pipeline capable of handling raw satellite data, cloud mosaics, land mosaics, and GPU‑ready texture outputs.

The codebase includes two major components:

Cloud Processing Pipeline
A satellite‑driven workflow that:

Downloads and caches raw data from multiple geostationary and polar‑orbiting satellites.

Decodes Level‑1/Level‑2 products using SatPy.

Reprojects imagery into six gnomonic cube faces.

Applies blending, gap‑filling, smoothing, and caching logic.

Exports cloud cubemap faces and optional GPU‑compressed textures.

This pipeline is designed for testing global cloud field generation and evaluating reprojection accuracy across different satellite sources.

Land Mosaic Pipeline
A tile‑based workflow that:

Fetches VIIRS corrected reflectance tiles from NASA GIBS over extended date ranges.

Stacks and filters tiles to reduce cloud contamination.

Assembles a global equirectangular mosaic.

Splits the mosaic into cubemap faces.

Optionally compresses outputs into KTX2 textures.

This component is used to test cloud‑free land generation, tile compositing, and cubemap reprojection methods.

Repository Structure
Code
geospatial-engine/
│
├── cloud_mosaic/                     # Cloud cubemap outputs
├── earth_land_mosaic/                # Land mosaic and KTX2 outputs
│   └── ktx2_faces/
│
├── tiles/                            # Cached NASA GIBS tiles
├── data_cache/                       # Raw satellite cache
├── data_cache_vs/                    # Meteosat raw cache
│
├── cloud_pipeline.py                 # Satellite cloud processing engine
├── earth_mosaic.py                   # Land mosaic and cubemap engine
├── telegram_sources.txt              # External news source list used for testing
├── requirements.txt
└── README.md
Purpose
This repository serves as a geospatial testing project.
It is not a finished product or a polished framework.
The goal is to experiment with:

satellite ingestion

reprojection techniques

mosaic generation

cubemap workflows

GPU texture compression

caching and data management

The project demonstrates significant progress in building a functional Earth‑processing pipeline from scratch.
=======
# geospatial-testing-2025
This is a memorable project on geospatial testing with advanced python that i made in 2025(Age 15).
>>>>>>> fb1c17c298dd1eff7baf59fc76183e3165146768
