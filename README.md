# ICP-Merging Project

## Acknoledgements

This project was developed to explore the potential of Gaussian Splatting techniques for cultural heritage preservation, as part of the project exam for the course [Augmented and Virtual Reality](https://corsi.unige.it/off.f/2023/ins/66562) during Master's degree in Computer Engineering - Artifical Intelligence at the University of Genova.

## Overview

This project implements the **Iterative Closest Point (ICP)** algorithm to merge two 3D models in Splat PLY format, those generated using **Gaussian Splatting**. The ICP algorithm aligns and merges the models by minimizing the differences between their point clouds. The resulting merged model highlights differences between the two input models, with **green** indicating "new" parts and **red** indicating "eroded" parts. This is particularly useful for **cultural heritage** applications, where it can help visualize changes over time.

The project manipulates Gaussian Splatting PLY files using a custom class called `SplatPLYHandler`, performing manual extraction, merging, custom file creation and additional features (require higher computational resources).

---

## Features

- **ICP Alignment**: Aligns two 3D models using the Iterative Closest Point algorithm.
- **Splat PLY File Support**: Handles standard Gaussian Splatting PLY files (Splat PLY).
- **Difference Highlighting**: Visualizes differences between the models using colors:
  - **Green**: Represents "new" parts (additions).
  - **Red**: Represents "eroded" parts (removals).
- **Gaussian Splatting Compatibility**: Supports advanced Gaussian Splatting PLY files for enhanced 3D representation.
- **Interactive GUI**: Using a small GUI you can interact with the project without having to code anything.

---

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/Martin-Martuccio/ICP-Merging.git
    cd ICP-Merging
    ```

2. Install tkinter (required for the Interactive GUI):

  - For Ubuntu users:

    ```sh
    apt-get install python3-tk
    ```

  - For Fedora users:

    ```sh
    dnf install python3-tkinter
    ```

3. Install Anaconda (if you prefer Miniconda just change every "Anaconda3" with "Miniconda3"):

    ```sh
    wget https://repo.anaconda.com/archive/Anaconda3-latest-Linux-x86_64.sh
    bash ~/Anaconda3-latest-Linux-x86_64.sh
    ```

4. Create conda enviroment in order to install the required dependencies:

    ```sh
    conda env create -f environment.yml
    conda activate icp_merging
    ```

# Usage

## Prepare Input Models

Place your PLY files (standard or Gaussian Splatting) in the input folder (`data/input/`).

## Run the ICP-Merging Script

- Inside `src/` run the script in order to open the GUI:
  
  ```sh
    python main.py
  ```

<p align="center">
  <img src="images/GUI_Example.png" alt="Interactive GUI" width="300"/>
</p>

- Click on the buttons `Load Model 1` and `Load Model 2` in order to choose the two Splat PLY files to confront.
- Customize each Parameter (optional).
- Click on the button `Process Models` and wait (for files of ∼200 MB it takes about 15 seconds).

## Output

- The merged model will be saved as `merged_model.ply` in the output folder (`data/output/`).
  
- By default, differences between the models will be highlighted in green (new parts) and red (eroded parts).
  
# Example Workflow

We took the [Satiro e Baccante](https://poly.cam/capture/5621D36B-36BF-4655-AE5C-B2D68DF0A851?) statue (sculpted by James Pradier), from `poly.cam` website, as example.

## Input Models

- `model1.ply`: Represents the initial state of a cultural heritage site.
  
- `model2.ply`: Represents the current state of the same site.

## Merging Process

- The ICP algorithm aligns the two models.

<div align="center">

  | <img src="images/ICP_Before.png" alt="Before ICP" width="300"/> | <img src="images/ICP_After.png" alt="After ICP" width="200"/> |
  |-------------------------------------------------------------|---------------------------------------------------------------|

</div>

- Differences are highlighted:
  
  - Green: New structures or additions.

  - Red: Eroded or missing structures.

<p align="center">
  <img src="images/Example_Broken.png" alt="Satiro e Baccante" width="300"/>
</p>

## Output

- A single Splat PLY file (`merged_model.ply`) is created, showing the aligned models with highlighted differences.

- To see the merged model this custom viewer is available: [Gaussian Splatting WebGL](https://github.com/biaperass/Gaussian-Splatting-WebGL)

# Gaussian Splatting Support

This project supports Gaussian Splatting PLY files, which are advanced 3D representations that include additional features like density and splatting parameters. These files are particularly useful for high-quality 3D reconstructions but require more computational resources.

For more information on Gaussian Splatting, refer to:

- [Gaussian Splatting - Wikipedia](https://en.wikipedia.org/wiki/Gaussian_splatting)
  
- [Gaussian Splatting Paper (arXiv)](https://arxiv.org/abs/2308.04079)
  
# Collaborators

This project was developed by:

- [@Martin-Martuccio](https://github.com/Martin-Martuccio) - Martin Martuccio
  
- [@PSamK](https://github.com/PSamK) - Samuele Pellegrini
  
- [@biaperass](https://github.com/biaperass) - Bianca Perasso
  
- [@LorenzoMesi](https://github.com/LorenzoMesi) - Lorenzo Mesi
  
## Contact

For questions, issues, or collaborations, please contact:

- Martin Martuccio: [[martinmartuccio@gmail.com](Martin:martinmartuccio@gmail.com)]

- Samuele Pellegrini: [[pellegrini.samuele.1@gmail.com](Samuele:pellegrini.samuele.1@gmail.com)]

- Bianca Perasso: [[bianca.perasso@gmail.com](Bianca:bianca.perasso@gmail.com)]

- Lorenzo Mesi: [[mesilorenzo@gmail.com](Lorenzo:mesilorenzo@gmail.com)]

# License

This project is licensed under the MIT License. See the `LICENSE` file for details.

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python: 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
