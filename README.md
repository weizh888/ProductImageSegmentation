# ProductImageSegmentation
Wait to add more information.

## Basic Installation

### Windows Prerequisites

In order to run the project, you will need Python, pip and the relative libraries.

#### Prerequisites

  - [Git for windows](https://git-for-windows.github.io/)
  - [Python 3.5+ 64-bit](https://www.python.org/downloads/)
  - [LabelImg](https://github.com/tzutalin/labelImg)

    Install by `pip3 install labelImg`
      - [PyQt4](https://www.riverbankcomputing.com/software/pyqt/download)
        `pip3 install PyQt4-4.11.4-cp27-cp27m-win32.whl`
      - [lxml](http://lxml.de/installation.html)
        `pip3 install lxml`

    **Note**: Use `pip3` just in case two versions of Python are installed (Check it).

  - [TensorFlow's Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)

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

### Training the model
