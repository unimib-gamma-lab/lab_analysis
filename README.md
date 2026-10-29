# Gamma Spectroscopy & Quantum Physics Laboratory

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue?logo=c%2B%2B&logoColor=white)](https://isocpp.org/)
[![ROOT](https://img.shields.io/badge/CERN%20ROOT-v6.28%2B-orange?logo=cern&logoColor=white)](https://root.cern/)

> Repository for the Advanced Physics Laboratory course focusing on gamma spectroscopy, Compton scattering (Klein-Nishina cross-section), and the Wu-Shaknov experiment on quantum entanglement.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Environment Setup](#-environment-setup)
- [Workflow & Conventions](#-workflow--conventions)
- [Git & GitHub Beginner's Guide](#-git--github-beginners-guide)
- [Contributors](#-contributors)

---

## 🔬 Overview

This repository contains the analysis code, data processing pipelines, and documentation for three main laboratory experiments:

1. **Gamma Spectroscopy & Energy Calibration**: Energy and resolution calibration of scintillation detectors using $^{22}\text{Na}$.
2. **Compton Formula**: Demonstration of Compton Formula
3. **Compton Scattering & Klein-Nishina Formula**: Measurement of differential cross-sections of scattered gamma photons at different angles and comparison with quantum electrodynamics predictions.
4. **Wu-Shaknov Experiment**: Experimental verification of quantum mechanical correlations in polarization states of annihilation photon pairs.

---

## 📁 Repository Structure

```text
.
├── data/                  # Raw experimental spectra (.Spe, .root, .csv)
│   ├── calibration/       # Calibration sources data
│   ├── compton/           # Compton scattering measurements by angle
│   └── wu-shaknov/        # Coincidence spectra for polarization
├── docs/                  # Experiment manuals, reference papers, datasheets
├── notebooks/             # Jupyter Notebooks for interactive data exploration
├── results/               # Generated figures (.png, .pdf), tables, and fitted parameters
├── src/                   # Main source code
│   ├── cpp/               # CERN ROOT C++ macros (.C) and compiled scripts
│   └── python/            # Data processing modules, fitting procedures, utilities
├── .gitignore             # Files and directories ignored by Git
├── environment.yml        # Conda environment definition file
├── LICENSE                # License information
└── README.md              # Project documentation
