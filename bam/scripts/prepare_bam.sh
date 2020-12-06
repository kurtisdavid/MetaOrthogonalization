#!/bin/bash

git clone https://github.com/google-research-datasets/bam.git
pip install bam-intp
source scripts/download_models.sh
source scripts/download_datasets.sh
python scripts/construct_bam_dataset.py
