# BadAppleCats

## Overview

CLIP embeddings, FAISS nearest-neighbour retrieval and brightness constraints turn video frames into cat-image mosaics.

## Stack

Python, PyTorch, Transformers/CLIP, FAISS, OpenCV, NumPy, Pillow and tqdm.

## Running the legacy project

The source pipeline is image preparation → make_embeddings.py → build_faiss.py → render_final.py. Run from the project root. Original video and image datasets must be supplied separately; make_embeddings.py contains a legacy absolute input path. No pinned dependency manifest exists, so a clean installation and full render are not verified.

## Implementation and limitations

The existing rendered video is evidence of an artifact, not a new successful reproduction. No automated functional suite was found. Original media, cat datasets, frames, embeddings and indexes are excluded from this code snapshot. Their redistribution rights are not established.

## Maintenance status

Development and maintenance are discontinued. This is a preservation snapshot, not a new release or feature-development plan. Running services and scheduled jobs are not automatically shut down by this status.

This repository is intended for Private preservation. Public release requires a separate review and approval. See PRESERVATION.md for excluded material and verification limits.
