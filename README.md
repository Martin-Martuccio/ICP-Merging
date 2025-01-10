# ICP-Merging Project

## Overview

This project implements the **Iterative Closest Point (ICP)** algorithm to merge two 3D models in PLY format, including those generated using **Gaussian Splatting**. The ICP algorithm aligns and merges the models by minimizing the differences between their point clouds. The resulting merged model highlights differences between the two input models, with **green** indicating "new" parts and **red** indicating "eroded" parts. This is particularly useful for **cultural heritage** applications, where it can help visualize changes over time.

The project manipulates PLY files using the `PlyData` and `PlyFile` libraries, performing manual extraction, merging, and custom PLY file creation. It also supports Gaussian Splatting PLY files, which include additional features but require higher computational resources.

---

## Features

- **ICP Alignment**: Aligns two 3D models using the Iterative Closest Point algorithm.
- **PLY File Support**: Handles both standard PLY files and Gaussian Splatting PLY files.
- **Difference Highlighting**: Visualizes differences between the models using colors:
  - **Green**: Represents "new" parts (additions).
  - **Red**: Represents "eroded" parts (removals).
- **Custom PLY Creation**: Manually extracts, merges, and creates PLY files with custom properties.
- **Gaussian Splatting Compatibility**: Supports advanced Gaussian Splatting PLY files for enhanced 3D representation.

---

## Requirements

- **Python 3.x**
- **NumPy**: For numerical operations.
- **Open3D**: For 3D data processing and visualization.
- **PlyFile**: For reading and writing PLY files.

Install the required dependencies using:
```bash
pip install numpy open3d plyfile
```

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/ICP-Merging.git
    cd ICP-Merging
    ```
2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

# Usage

## Prepare Input Models:
- Place your PLY files (standard or Gaussian Splatting) in the input folder.
  
- Ensure the files are in ASCII format for compatibility.

## Run the ICP-Merging Script:

- Open the `main.py` file and update the paths to your input models:
  
  ```sh
    source_path = "input/model1.ply"
    target_path = "input/model2.ply"
  ```
  
- Run the script:
  
  ```sh
    python main.py
  ```

## Output:

- The merged model will be saved as `merged_model.ply` in the output folder.
  
- Differences between the models will be highlighted in green (new parts) and red (eroded parts).
  

# Example Workflow

## Input Models:

- `model1.ply`: Represents the initial state of a cultural heritage site.
  
- `model2.ply`: Represents the current state of the same site.
  

## Merging Process:

- The ICP algorithm aligns the two models.
  
- Differences are highlighted:
  
  - Green: New structures or additions.
    
  - Red: Eroded or missing structures.
    

## Output:

- A single PLY file (`merged_model.ply`) is created, showing the aligned models with highlighted differences.

# Gaussian Splatting Support

This project supports Gaussian Splatting PLY files, which are advanced 3D representations that include additional features like density and splatting parameters. These files are particularly useful for high-quality 3D reconstructions but require more computational resources.

For more information on Gaussian Splatting, refer to:

- [Gaussian Splatting - Wikipedia](https://en.wikipedia.org/wiki/Gaussian_splatting)
  
- [Gaussian Splatting Paper (arXiv)](https://arxiv.org/abs/2112.10670)
  

# Collaborators

This project was developed by:

- [@Martin-Martuccio](https://github.com/Martin-Martuccio) - Martin Martuccio
  
- [@PSamK](https://github.com/PSamK) - PSamK
  
- [@biaperass](https://github.com/biaperass) - Bianca Perasso
  
- [@LorenzoMesi](https://github.com/LorenzoMesi) - Lorenzo Mesi
  

# License

This project is licensed under the MIT License. See the `LICENSE` file for details.

# Acknowledgements

- Gaussian Splatting: For advanced 3D representation techniques.
  
- Open3D: For 3D data processing and visualization.
  
- PlyFile: For PLY file manipulation.
  

# Contact

For questions, issues, or collaborations, please contact:

- Your Name: [[martinmartuccio@gmail.com](Martin:martinmartuccio@gmail.com)]
  
- Project Repository: https://github.com/.../ICP-Merging
  

# Future Work

- **Support for Splat PLY Files:** Enhance the project to handle Gaussian Splatting PLY files with additional features.
  
- **Performance Optimization:** Reduce computational costs for large datasets.
  
- **User Interface:** Develop a graphical interface for easier model selection and visualization.