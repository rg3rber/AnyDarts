
![anydarts_logo](https://github.com/user-attachments/assets/ccfae7dd-3321-46b6-a94b-61b34b145d84)

- The model weights and datasets are available here: [AnyDarts Google Drive Shared Folder](https://drive.google.com/drive/folders/1og8SZbe6Yn7kWXbzq2ANEtsk-o_F5Cou?usp=drive_link)
- The Android Project repository and the APK can be found here: [AnyDarts Android](https://github.com/rg3rber/AnyDarts-Android)
- The related repository and heavily influencing this project: [DeepDarts Repository](https://github.com/wmcnally/deep-darts)


## Prerequisites
Python >=3.12

## Setup
1. [Install Miniforge (or another conda distribution)](https://github.com/conda-forge/miniforge)
2. Clone this repo: ```$ git clone https://github.com/rg3rber/AnyDarts.git```
3. Create a new conda environment from the environment.yaml file: ```conda env create -f environment.yml```
4. Activate the conda environment: ```conda activate anydarts```
5. Download the datasets and weights from [AnyDarts Google Drive Shared Folder](https://drive.google.com/drive/folders/1og8SZbe6Yn7kWXbzq2ANEtsk-o_F5Cou?usp=drive_link)

## Pipeline to increase datasets

1. Take photos
2. Save in session folder to dataset/images
3. run image renaming script (excluding the images that are already annotated) using start id where you left off last time
4. Run "annotate" on folder above using --img-folder
5. Annotate images manually per folder
6. Run "combine_labels" 
7. Run "crop_images" with --size=800 and appropriate --labels-path
8. run "dump_pkl_to_txts" to get labels as txt
9. put cropped images and labels into your dataset with YOLO structure

!- DO NOT RENAME IMAGES AFTER

## Board Detection

<img src="https://github.com/user-attachments/assets/dc2f8158-d366-4948-87ad-4460a0d0438b" alt="Description" width="400"/>

## Scoring Pipeline

<img src="https://github.com/user-attachments/assets/01c92796-66b0-4c30-94c6-1302c631391d" alt="scoring-pipeline" width="800"/>

## Scoring Results examples



<img src="https://github.com/user-attachments/assets/1d83cde3-683a-416a-ac99-a6ff43d3cb83" alt="thesis-board-pred" width="400"/>

<img src="https://github.com/user-attachments/assets/e9e0976d-9215-4c22-8213-aaa43acc3a90" alt="generalization-board-pred" width="400"/>


# Datasets 

Located at: https://drive.google.com/drive/folders/1og8SZbe6Yn7kWXbzq2ANEtsk-o_F5Cou?usp=drive_link

## AnyDarts_v1

### train: 889 (891 initial) imgs
- D1: 98 
- D2: 150 
- D3: 643
! 2 deepdarts d1 training images had corrupt labels:

train: Scanning /local/home/AnyDarts/dataset/any_v1/train/labels.cache... 891 images, 0 backgrounds, 2 corrupt: 100%|████████
train: WARNING ⚠️ /local/home/AnyDarts/dataset/any_v1/train/images/d1_02_04_2020_IMG_1092.JPG: ignoring corrupt image/label: non-normalized or out of bounds coordinates [1.0023872]
train: WARNING ⚠️ /local/home/AnyDarts/dataset/any_v1/train/images/d1_02_04_2020_IMG_1093.JPG: ignoring corrupt image/label: non-normalized or out of bounds coordinates [1.0023872]

  ### val: 144 imgs
  - D1: 15
  - D2: 23
  - D3: 106
 
### test: 67 imgs
- D1: 12
- D2: 24
- D3: 31

## AnyDarts_v2:

train: additional 203 => 1092 imgs

val: additional 70 => 214

## Test suite Dataset:

- **Front facing 3 darts hard:** 21 challenging front-facing images with 3 darts.
- **Occlusion only:** 11 images focusing solely on dart occlusions.
- **Test v1:** The 67 images from the AnyDarts_v1 dataset.
- **Testing occlusion angles:** 30 images taken on repeated setups from different angles introducing and removing occlusions to compare different angles.
- **Precision test:** 20 images with darts placed exactly on segment borders to test detection and scoring precision.
- **Full game:** 102 images covering full two-player gameplay.
- **Easy one dart:** 43 simpler images, with one dart in each field (once per field).
- **Test generalization:** 13 images of various boards, including unseen ones.
- **Test rim:** 15 images without the black tape normally used to cover board ads, revealing the full rim.
- **Full game 3 darts only:** 19 images showing only the 3-dart portion of full games.
- **Taken on Samsung:** 20 images captured using a Samsung Galaxy S9+ camera.
- **DeepDarts D2:** The 150 images making up the DeepDarts *D2* test set.
- **Different darts:** 25 images using dart types and colors differing from the main blue set.

=> totals: 536 (might include duplicates) 
  
