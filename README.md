# CV-Road_Segmentation

## Overview
This repository is dedicated to the exploration and application of computer vision techniques for two primary tasks: road line extraction and road segmentation. We leverage a combination of techniques to effectively analyze road images and extract useful patterns.

## Features
- **Region of Interest (ROI)** focusing on specific horizon lines.
- **Thresholding**: Image binarization based on the HSV color scale.
- **Morphological Operations**: Includes erosion, dilation, opening, and closing for improving the Thresholding results.
- **Clustering**: Used in segmentation scenarios.
- **Canny Edge Detection**: Applied for edge detection in images.

## Repository Structure
CV-Road_Segmentation
│
├── Clustering_codes/ # Scripts for clustering algorithms  
├── Images/ # VArious images for testing and obtained results  
├── KITTI_roadDataset/ # KITTI_road dataset images   
├── Results/ # Result images for different approaches  
├── Videos_results/ # Result videos for different approaches  
├── __pycache__/ # Python cache files  
├── carina_dataset/ # Local testing images  
├── git_img/ # Images needed to the README.ms file  
├── Line_extractionOnVideo.py # Script for video line extraction  
├── RoadLine_Extraction.py # Script for extracting road lines  
├── Road_Segmentation.py # Script for segmenting roads  
├── histogram_plt.py # Script for plotting histograms  
├── lower_upper_HSV.py # Script for HSV range adjustment  
└── morphological_op.py # Scripts for applying morphological operations  


## Results
### Orginal image:
<p align="center">
  <img src="Images/BP6.png" alt="Gráfico de Velocidade 1" width="65%" />
</p>

### Thresholding na escala HSV:
<p align="center">
  <img src="Images/processed.png" alt="Gráfico de Velocidade 1" width="45%" />
  <img src="git_img/Th_sg.png" alt="Gráfico de Velocidade 2" width="45%" />
</p>


### Clustering:
<p align="center">
  <img src="Images/segmentation.png" alt="Mean_shift++" width="45%" />
  <img src="Images/kmean_seg.png" alt="K_means" width="45%" />
</p>


### Canny Edge Detection:
<p align="center">
  <img src="git_img/Canny.png" alt="Canny" width="65%" />
</p>


### Line Extraction:
<p align="center">
  <img src="git_img/lines.png" alt="Line" width="65%" />
</p>


### Segmentation:
<p align="center">
  <img src="git_img/sg.png" alt="Segmentation" width="65%" />
</p>
