#!/usr/bin/env bash

python -m pip install --upgrade torch
python download_punkt.py
python -m qanta.server web
