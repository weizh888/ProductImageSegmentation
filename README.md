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

### Training the Model

### Using Google Cloud Platform
  **Note**: Following [GCP documentation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_cloud.md)
  and solve the bugs of [missing module](https://github.com/tensorflow/models/issues/2739) before submitting the jobs.

  ```
  export PROJECT=$(gcloud config list project --format "value(core.project)")
  export YOUR_GCS_BUCKET="gs://weizh888"
  export JOB_ID="imgSeg_$(date +%s)"
  ```
#### Submit Training Job
  ```
  gcloud ml-engine jobs submit training ${JOB_ID} \
      --module-name object_detection.train \
      --job-dir=${YOUR_GCS_BUCKET}/train \
      --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz \
      --staging-bucket ${YOUR_GCS_BUCKET} \
      --region us-central1 \
      --config object_detection/samples/cloud/cloud.yml \
      -- \
      --train_dir ${YOUR_GCS_BUCKET}/training \
      --pipeline_config_path=${YOUR_GCS_BUCKET}/ssd_mobilenet_v1_gcs.config \
      --eval_data_paths ${YOUR_GCS_BUCKET}/preproc/eval* \
      --train_data_paths ${YOUR_GCS_BUCKET}/preproc/train*
  ```

#### Monitor Training Logs
`gcloud ml-engine jobs stream-logs ${JOB_ID}`

#### Graph Visualization
`tensorboard --logdir=${YOUR_GCS_BUCKET}/training`
