# Modeling Fashion Recommendations with Sketch-Based Model

This repository contains our solution to [FARFETCH Fashion Recommendations Challenge](https://www.ffrecschallenge.com/ecmlpkdd2021/)  that achieved 3rd place.

The aim is to predicit the products clicked by a user from a list of selected recommendations.
First, we use [Cleora](https://arxiv.org/abs/2102.02302) - our graph embedding method - to represent products as a directed graph and learn their vector representation. Products are embeded in two relations:
- Only clicked products in a given session  (`clicked-modality`)
- Viewed products in a given session    (`viewed-modality`)

As a results we obtain two embeddings. No all products were clikced or viewed, so for a small number of proucts we do not have a vector representation from Cleora.
Next, we apply [EMDE](https://arxiv.org/abs/2006.01894) to predict the product based on previously clicked and viewed products. We also add some features associated with each user.

Model takes as an input 4 sketches:
1. Sketch of all products clicked in the previous sessions (`clicked-modality`)
2. Sketch of all products clicked in the current sessions apart from the current query (`clicked-modality`)
3. Sketch of all products viewed in the current sessions apart from the current query (`viewed-modality`)
4. Sketch of all products displayed in the current query (`viewed-modality`)

Model return a sketch of a product clicked in the current query. Output sketch is from `viewed-modality` as it contains more product embedding/codes. The output sketch is then scored against all product sketch from `viewd-modality` and click probiablity is obtained.


## Requirements
* Download binary [Cleora release](https://github.com/Synerise/cleora/releases/download/v1.1.0/cleora-v1.1.0-x86_64-unknown-linux-gnu). Then add execution permission to run it. Refer to [cleora github webpage](https://github.com/Synerise/cleora) for more details about Cleora.
* Python 3.7
* Install requirments: `pip install -r requirements.txt`
* GPU for training


## Getting Started

1. Create `data` directory in `src` folder:
    ```bash
    mkdir src/data
    ```
2. Put `train.parquet`, `validation.parquet` and `test.parquet` into `src/data` folder
3. Change directory to `src`
    ```bash
    cd src
    ```
4. Transform all parquest files to CSV with sequential-like form. It also creates input files to Cleora:
    ```
    python transform_to_sequential_data.py --data-dir data          

    ```
    This script will create three CSV files: `data/train_original_processed_reproducing.csv`, `data/val_original_processed_reproducing.csv` and `data/test_original_processed_reproducing.csv`
    And two input files for Cleora algorithm
    `data/cleoraInput_sessionIdGrouped_viewed`, `data/cleoraInput_sessionIdGrouped_onlyClicked`.
    Script also creates and saves dict with products2attributes data; at `data/products_dict_reproducing`.

5. Create datapoints for running the model:
    ```
    python create_datapoints.py --data-dir data          

    ```
    This script will create three files: `data/train_datapoints_sequential_reproducing`, `data/validation_datapoints_sequential_reproducing` and `data/test_datapoints_sequential_reproducing`.

6. Compute product sketches using [Cleora](https://github.com/Synerise/cleora) and [EMDE](https://arxiv.org/abs/2006.01894)
    ```
    python encode.py --data-dir data    
    ```
    This script will create LSH codes for each product from `viewed-modality` and `clicked-modality`.
    Codes are saved to `data/codes_viewed` and `codes_clicked`
7. Run training
    ```
    python train.py --data-dir data    
    ```
    Logs are saved to: `src/logs/runs`
8. Download trained model checkpoint:
    https://drive.google.com/file/d/1vnuKZGdEGHzGkBrVUx7JNbyOcqE-OK5o/view?usp=sharing    
9. Run test. Use flag `checkpoint-path` to specify trained model path; `model_trained.ckpt` by default. Flag    `--subset-to-use` to specify whether to use `validation` or `test` subset; `test` by default.
    ```
    python test.py --data-dir data    
    ```