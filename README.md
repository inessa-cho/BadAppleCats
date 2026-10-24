# BadAppleCats

## Overview

Turns video frames into cat-image mosaics using CLIP embeddings, FAISS retrieval and brightness matching.

## Stack

Python, PyTorch, Transformers/CLIP, FAISS, OpenCV, NumPy, Pillow and tqdm.

## Getting Started

The source pipeline is image preparation → make_embeddings.py → build_faiss.py → render_final.py. Run from the project root. Original video and image datasets must be supplied separately; make_embeddings.py contains an absolute input path that must match your environment. No pinned dependency manifest exists, so a clean installation and full render are not verified.

## Implementation and limitations

The existing rendered video is evidence of an artifact, not a new successful reproduction. No automated functional suite was found. Original media, cat datasets, frames, embeddings and indexes are excluded from this code snapshot. Their redistribution rights are not established.

See [PRESERVATION.md](PRESERVATION.md) for the maintenance record, excluded files and verification scope.
