# 3D Post-Processing Pipeline for Fresco Reconstruction
Pipeline and scripts for digitizing the scanned frescoes. This repository provides the framework and tools for the post-processing pipeline used to reconstruct 3D models of fresco fragments. The pipeline involves several stages of processing, as shown in the diagram.
![repair_3d_postprocessing_pipeline_2](https://github.com/user-attachments/assets/b0978359-892b-4132-88e7-0c8db5db0200)

# Scanning
For the scanning procedure we used the following sensoring:
1. the Polyga H3 3D scanner
2. the sony α7c mirroless digital camera

For digitizing the acquired data from the Polyga H3 3D scanner we used the accompanied software Flex3DScan and a rotary table as peripheral, in order to capture the fragments from multiple perspectives. Moreover, the usage of a lighting box to ensure consistent isometric ambient lighting conditions.

Each frescoe piece is scanned both from the bottom and and upper part, the bottom and upper parts are identified from the corresponding postfixes. For the bottom is <span style="background-color:gray">`RPf_<id>a.ply`</span> and the upper <span style="background-color:gray">`RPf_<id>b.ply`</span> respectively.

# Digitization
Once we have both the bottom and upper part of each frescoe piece we need to align them in order to create a unique piece. This is done by performing  a Truncated Least Squares
(TLS) registration and using the Teaser++ library.

The alignment process involves extracting robust putative correspondences $(\mathbf{a}_i,\mathbf{b}_i), i = 1, . . . , N$ and determining the optimal transformation to align the top, $\mathbf{a}$, and bottom, $\mathbf{b}$ point clouds.
```math
\min_{s>0, {R}\in SO(3), \mathbf{t}\in \mathbb{R}^3}\sum_{i=1}^{N}min\left (\frac{1}{\beta_{i}^{2}} \left \| \mathbf{b}_i - sR\mathbf{a_i} - \mathbf{t}\right \|^2, \bar{c}^2 \right )
```
Where $s$, $R$, and $t$ are the unknown (to-be-computed) scale, rotation, and translation respectively and which are computed through a least-squares solution of measurements with small residuals $(\frac{1}{\beta_{i}^2}\left \|\mathbf{b}_i - sR\mathbf{a}_i - \mathbf{t}\right \|^2 \leq \bar{c}^2)$ and a given noise bound $\beta_i$.
The (RANSAC) algorithm is then leveraged to handle outlier correspondences, ensuring the determination of the best transformation parameters.

For the high resolution maps we utilize the high-resolution images from the digital camera to generate a secondary 3D model, facilitating the mapping of high-resolution texture information onto the 3D model reconstructed from the 3D scanner.
For this purpose, we employ Structure from Motion (SfM).
```math
    \min_{\{\mathbf{P}_j\}, \{C_i\}}\sum_{i\sim j}^{}\left ( x_{i,j} - \frac{C_{i1}^{T}\mathbf{P}_j}{C_{i3}^{T}\mathbf{P}_j} \right )^2 + \left ( y_{i,j} - \frac{C_{i2}^{T}\mathbf{P}_j}{C_{i3}^{T}\mathbf{P}_j} \right )^2
```
where, $(x_{ij} , y_{ij} ) \in \mathbb{R}^2$ denote the
computed projection of point $\mathbf{P}_j$ (in homogeneous coordinates $`[\mathbf{P}_j , 1] \in \mathbb{R}^4`$) onto the image plane of the camera $`C_i$, with $C_{ik}\in \mathbb{R}^4`$ denoting the $k$'th row of the $C_i (1 \leq k \leq 3) 3\times 4$ camera matrix, and $i\sim j$ indicating that the $j$'th scene point is visible by the $i$'th camera.

To achieve our goal of reconstructing only the fragments and not the entire scanning scene, using segmentation masks allows us to avoid reconstructing the background environment surrounding each fragment during the scanning process.


To start the aligning process run the <span style="background-color:gray">`align_meshes_v1.py`</span> script. You will need to specify the path were the upper and bottom parts are located, they need to be in the same folder. This will create and save the unique piece as as <span style="background-color:gray">`RPf_<id>.ply`</span> in the same folder where the individual pieces <span style="background-color:gray">`a`</span> and <span style="background-color:gray">`b`</span> are. This mesh model though comes with a low resolution texture map.

For mapping the high resolution texture map acquired from the Sony α7c camera to the mesh model extracted from the Polyga scanner we need to have a set of RGB data, i.e. sample images, covering the frescoe from multiple views around it. If the images are already captured directly then no need for extra pre-processing is needed. If the RGB data are captured as a video then we need to extract the individual frames from this video. This is done with the <span style="background-color:gray">`video2imgs.py`</span> script. Thereafter we create the mask of the frescoes for each of these images which will help to have a better reconstruction model from the photogrammetry pipeline. To extract the mask of the object use the <span style="background-color:gray">`create_bg_fg_mask.py`</span> script.

Once the masks are created you can run the <span style="background-color:gray">`bundler_extractor_metashape_v1.py`</span> script which will launch metashape though comand line and will create the photogrammetric mesh model of the fragment with the high resolution texture map. Thereafter, we just need to map the high resolution map to the original 3D scanner sensor based mesh model. We can do that by running the script <span style="background-color:gray">`texture_mapping_from_images.py`</span>. This will give you the final high resolution textured map mesh file.


# References
[TEASER++: fast & certifiable 3D registration](https://github.com/MIT-SPARK/TEASER-plusplus)

[PyMeshLab](https://github.com/cnr-isti-vclab/PyMeshLab)

[Metashape](https://github.com/agisoft-llc)

[Open3D](https://github.com/isl-org/Open3D)

---

## Acknowledgements

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 964854

---

## Citation

If you use this code in your research, please cite the following paper:

```
@inproceedings{repair2024,
title={Re-assembling the Past: The RePAIR Dataset and Benchmark for Realistic 2D and 3D Puzzle Solving},
author={Tsesmelis, Theodore and Palmieri, Luca and Khoroshiltseva, Marina and Islam, Adeela and Elkin, Gur and Shahar, Ofir Itzhak and Scarpellini, Gianluca and Fiorini, Stefano and Ohayon, Yaniv and Alal, Nadav and Aslan, Sinem and Moretti, Pietro and Vascon, Sebastiano and Gravina, Elena and Napolitano, Maria Cristina and Scarpati, Giuseppe and Zuchtriegel, Gabriel and Spühler, Alexandra and Fuchs, Michel E. and James, Stuart and Ben-Shahar, Ohad and Pelillo, Marcello and Del Bue, Alessio},
booktitle={NeurIPS},
year={2024}
}
```
