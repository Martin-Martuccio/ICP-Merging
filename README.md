# ICP-Merging Project

## Overview

This project implements the Iterative Closest Point (ICP) algorithm to merge two PLY models created using Gaussian Splatting. The ICP algorithm is used to align and merge 3D point clouds by minimizing the difference between them.

## Features

- Implementation of the ICP algorithm
- Support for PLY file format
- Merging of two 3D models

## Requirements

- Python 3.x
- NumPy
- Open3D

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/ICP-Merging.git
    ```
2. Install the required dependencies:
    ```sh
    pip install numpy open3d
    ```

## Usage

1. Place your PLY models in the `models` directory.
2. Run the ICP merging script:
    ```sh
    python merge.py models/model1.ply models/model2.ply
    ```
3. The merged model will be saved as `merged_model.ply` in the `output` directory.

## License

This project is licensed under the MIT License.

## Acknowledgements

- Gaussian Splatting technique
- Open3D library for 3D data processing

## Contact

For any questions or issues, please contact [your email].
