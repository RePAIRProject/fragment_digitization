# fragment_digitization
Pipeline and scripts for digitizing the scanned frescoes.

# Scanning
For the scanning procedure we used the following sensoring:
1. the Polyga H3 3D scanner
2. the sony α7c mirroless digital camera

For digitizing the acquired data from the Polyga H3 3D scanner we used the accompanied software Flex3DScan and a rotary table as peripheral.

Each frescoe piece is scanned both from the bottom and and upper part, the bottom and upper parts are identified from the corresponding postfixes. For the bottom is <span style="background-color:gray">`RPf_<id>a.ply`</span> and the upper <span style="background-color:gray">`RPf_<id>b.ply`</span> respectively.

# Digitization
Once we have both the bottom and upper part of each frescoe piece we need to align them in order to create a unique piece. This is done with the <span style="background-color:gray">`align_meshes_v1.py`</span> script. This will create and save the unique piece as as <span style="background-color:gray">`RPf_<id>.ply`</span> in the same folder where the individual pieces <span style="background-color:gray">`a`</span> and <span style="background-color:gray">`b`</span> are. This mesh model though comes with a low resolution texture map. Which we try to improve by taking into action a photogrammetry pipeline.

For mapping the high resolution texture map acquired from the Sony α7c camera to the mesh model extracted from the Polyga scanner we need to have a set of RGB data, i.e. sample images, covering the frescoe from multiple views around it. If the images are already captured directly then no need for extra pre-processing is needed. If the RGB data are captured as a video then we need to extract the individual frames from this video. This is done with the <span style="background-color:gray">`video2imgs.py`</span> script. Thereafter we create the mask of the frescoes for each of these images which will help to have a better reconstruction model from the photogrammetry pipeline.



# Scripts description

