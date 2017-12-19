# ProductImageSegmentation
Wait to add more information.

## Basic Installation

### Windows Prerequisites

In order to run the project, you will need Python, pip and the relative libraries.

#### Prerequisites

  - [Git for windows](https://git-for-windows.github.io/)
  - [Python 3.5+ 64-bit](https://www.python.org/downloads/)
  - [LabelImg](https://github.com/tzutalin/labelImg)

    Download the zipped repository directly or install by `pip3 install labelImg`, then install
      - [PyQt4+](https://www.riverbankcomputing.com/software/pyqt/download)
        `pip3 install pyqt5` or
        `pip3 install PyQt4‑4.11.4‑cp36‑cp36m‑win_amd64.whl`
      - [lxml](http://lxml.de/installation.html)
        `pip3 install lxml`

    **Note**: Use `pip3` just in case two versions of Python are installed (Check it).

  - [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)

    Install required libraries, such as pandas, Pillow:

    `py -3 -m pip install pandas` or `pip3 install pandas`
    `py -3 -m pip install Pillow` or `pip3 install Pillow`

#### Download the Application

To run a copy from the latest master branch in git you can clone the repository:

```
git clone git@github.com:weizh888/ProductImageSegmentation.git
```

## Usage

### Creating the dataset

1. Manually Labelling

    Create labels based on https://shopee.sg/search/?keyword=women+apparel&subcategory.

    Testd the category *Pants_Leggings* First.

    simply run
    ```
    LabelImg
    ```
    **Open Dir, Change Save Dir, View->Autosaving**
    ```
    python labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
    ```

    For example, use `python labelImg.py 'C:\ProductImageSegmentation\samples' '.\data\predefined_classes.txt'`

2. Use the dataset on http://www.vision.ee.ethz.ch/~lbossard/projects/accv12/index.html

#### Generating TFRecord files for TensorFlow

1. Convert XML to CSV

    ```
    python xml_to_csv.py
    ```
2. Split the dataset to training and testing (default ratio: 4:1)

    ```
    python split_dataset.py
    ```
3. Generate .tfrecords files

    ```
    py -3 generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record
    py -3 generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record
    ```


### Running Locally (Windows 10)
#### Train the Model
  ```
  # From tensorflow\models\research\
  py -3 object_detection\train.py \
    --train_dir TRAIN_DIR \
    --pipeline_config_path PIPELINE_CONFIG_PATH
  ```

  Graph Visualization: `tensorboard --logdir=TRAIN_DIR`

#### Evaluate the Model
  ```
  # From tensorflow\models\research\
  py -3 object_detection\eval.py \
    --eval_dir EVAL_DIR \
    --pipeline_config_path PIPELINE_CONFIG_PATH \
    --checkpoint_dir TRAIN_DIR
  ```

  Graph Visualization: `tensorboard --logdir=EVAL_DIR --port 6007`

### Running on Google Cloud Platform (Ubuntu)
  **Note**: Follow [GCP documentation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_cloud.md)
  and solve the bugs of [missing module](https://github.com/tensorflow/models/issues/2739) before submitting the jobs.

  ```
  export PROJECT=$(gcloud config list project --format "value(core.project)")
  export GCS_BUCKET="weizh888"
  export JOB_NAME="imgSeg_$(date +%s)"
  export TRAIN_DIR="${GCS_BUCKET}/training"
  export EVAL_DIR="${GCS_BUCKET}/evaluation"
  export PIPELINE_CONFIG_PATH="${GCS_BUCKET}/config/ssd_mobilenet_v1_gcs.config"
  ```
#### Submit Training Job
```
gcloud ml-engine jobs submit training ${JOB_NAME} \
    --module-name object_detection.train \
    --job-dir=gs://${TRAIN_DIR} \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz \
    --staging-bucket gs://${GCS_BUCKET} \
    --region us-central1 \
    --config object_detection/samples/cloud/cloud.yml \
    -- \
    --train_dir gs://${TRAIN_DIR} \
    --pipeline_config_path=gs://${PIPELINE_CONFIG_PATH} \
    --eval_data_paths gs://${GCS_BUCKET}/preproc/eval* \
    --train_data_paths gs://${GCS_BUCKET}/preproc/train*
```

  Monitor Training Logs: `gcloud ml-engine jobs stream-logs ${JOB_NAME}`

  Graph Visualization: `tensorboard --logdir=gs://${TRAIN_DIR} --port 8080`

#### Evaluate the Model
```
gcloud ml-engine jobs submit training ${JOB_NAME}_eval \
    --job-dir=gs://${TRAIN_DIR} \
    --module-name object_detection.eval \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz \
    --region us-central1 \
    --scale-tier BASIC_GPU \
    -- \
    --checkpoint_dir=gs://${TRAIN_DIR} \
    --eval_dir=gs://${EVAL_DIR} \
    --pipeline_config_path=gs://${PIPELINE_CONFIG_PATH}
```
  Monitor Training Logs: `gcloud ml-engine jobs stream-logs ${JOB_NAME}_eval`

  Graph Visualization: `tensorboard --logdir=gs://${EVAL_DIR} --port 8081`

#### Export the Model

`gsutil cp gs://weizh888/training/model.ckpt-${CHECKPOINT_NUMBER}.* ~/`

```
# From tensorflow/models/research/
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path gs://${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix gs://${TRAIN_DIR}/model.ckpt-${CHECKPOINT_NUMBER} \
    --output_directory exported_model
```

## Results
### Total Loss and Precision
<img src="/model_six_classes/Total_Loss.PNG" width="300" height="300"> <img src="/model_six_classes/mAP.PNG" width="300" height="300">

The total loss reaches ~1.0 in the training dataset and the mAP@IoU≥0.5 reaches ~0.5 for the validation dataset. (The mAP@IoU≥0.5 reaches more than 0.9 in the one-class situation.)

### Evaluation Examples
#### Examples in one-class model

<img src="/model_one_class/images/00.png" width="300" height="300"> <img src="/model_one_class/images/01.png" width="300" height="300">

For more examples, check [examples](/model_one_class/images).

#### Examples in 6-class model

  The classes that are manually labeled from the examples:
  ```
  'Pants_Leggings'
  'Dresses'
  'Skirts'
  'Tops'
  'Shorts'
  'Lingerie'
  ```
  <img src="/model_six_classes/images/00.png" width="300" height="300"> <img src="/model_six_classes/images/06.png" width="300" height="300">
  <img src="/model_six_classes/images/57.png" width="300" height="300"> <img src="/model_six_classes/images/96.png" width="300" height="300">

  `Pants_Leggings`, `Dresses` and `Tops` are relatively easy to segment, but sometimes the model cannot identify `Dresses` and `Skirts`, `Skirts` and `Shorts`.

### Conclusion and More Thoughts
  The possible reasons are:
  ```
  1. In some scenes/images, some clothes are hard to identify even by human beings.
  2. There are less training examples for the category of `Skirts`, `Shorts` and `Lingerie`
  3. The model is not very tuned (due to time and computing resource limitation).
  ``
