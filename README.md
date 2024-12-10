# EventPointMesh: Human Mesh Recovery Solely from Event Point Clouds

[![Visit Project Page](https://img.shields.io/badge/Project%20Page-Visit%20Here-blue)](https://ryosukehori.github.io/EPM_ProjectPage/)

![Teaser Image](imgs/Fig1.png)

## Abstract
How much can we infer about human shape using an event camera that only detects the pixel 
position where the luminance changed and its timestamp? This neuromorphic vision technology 
captures changes in pixel values at ultra-high speeds, regardless of the variations in 
environmental lighting brightness. Existing methods for human mesh recovery (HMR) from event 
data need to utilize intensity images captured with a generic frame-based camera, rendering 
them vulnerable to low-light conditions, energy/memory constraints, and privacy issues. 
In contrast, we explore the potential of solely utilizing event data to alleviate these 
issues and ascertain whether it offers adequate cues for HMR, as illustrated in Fig.1. 
This is a quite challenging task due to the substantially limited information ensuing from 
the absence of intensity images. To this end, we propose EventPointMesh, a framework which 
treats event data as a three-dimensional (3D) spatio-temporal point cloud for reconstructing 
the human mesh. By employing a coarse-to-fine pose feature extraction strategy, we extract 
both global features and local features. The local features are derived by processing the 
spatio-temporally dispersed event points into groups associated with individual body segments. 
This combination of global and local features allows the framework to achieve a more accurate 
HMR, capturing subtle differences in human movements. Experiments demonstrate that our method 
with only sparse event data outperforms baseline methods.



## Dataset
The dataset used in this project will be made available soon.

![Image](imgs/Fig4.png)


## Code
The source code for this project is currently under preparation and will be released soon. Please check back later for updates.

