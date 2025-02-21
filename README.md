# DeepDarts

Code for the CVSports 2021 paper: [DeepDarts: Modeling Keypoints as Objects for Automatic Scorekeeping in Darts using a Single Camera](https://arxiv.org/abs/2105.09880)

## Prerequisites
Python 3.5-3.8, CUDA >= 10.1, cuDNN >= 7.6

## Setup
1. [Install Anaconda or Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Create a new conda environment with Python 3.7: ```$ conda create -n deep-darts python==3.7```. Activate the environment: ```$ conda activate deep-darts```
4. Clone this repo: ```$ git clone https://github.com/wmcnally/deep-darts.git```
5. Go into the directory and install the dependencies: ```$ cd deep-darts && pip install -r requirements.txt```
6. Download ```images.zip``` from [IEEE Dataport](https://ieee-dataport.org/open-access/deepdarts-dataset) 
   and extract in the ```dataset``` directory. Crop the images: ```$ python crop_images.py --size 800```. This step could
   take a while. Alternatively, you can download the 800x800 cropped images directly from IEEE Dataport. 
   If you choose this option, extract ```cropped_images.zip``` in the ```dataset``` directory.
8. Download ```models.zip``` from IEEE Dataport and extract in the main directory.

## Pipeline

    Take photo
    Save to images/utrecht_{month}{day}{year}
    Run "annotate" on folder above using --img-folder
    Annotate images manually
    Run "combine_labels"
    Run "crop_images" with --size=800 and appropriate --labels-path


## Sample Test Predictions

Dataset 1:\
![alt text](./d1_pred.JPG)

Dataset 2:\
![alt text](./d2_pred.JPG)

# Notes

Latest image id: 1055 (791 in d3_26_12_2024_random)

## Dataset holo_v1

### train: 891 imgs
- D1: 98 
- D2: 150 
- D3: 643

  ### val: 144 imgs
  - D1: 15
  - D2: 23
  - D3: 106
 
### test: 67 imgs
- D1: 12
- D2: 24
- D3: 31

## holo_v2:

train: additional 203 => 1094 imgs

val: additional 70 => 214

test: 
   1. full game: 119
   2. full game 3 darts only: 19
   3. front facing 3 darts hard: 28
   4. taken on samsung: 21
   5. zoomed: 15
   6. different darts: 46
   7. easy one dart: 43 (each field once)
   8. testing occlusions + angles: 30
   9. occlusion only: 21
   10. precision test: 20 (right on section borders)
   11. test rim: 15 (took away black tape covering board ads)

=> totals: 377
  
