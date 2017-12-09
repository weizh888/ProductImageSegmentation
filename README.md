# ProductImageSegmentation
Wait to add more information.

## Basic Installation

### Windows Prerequisites

In order to run the project, you will need Python, pip and the relative libraries.

#### Prerequisites

  - [Git for windows](https://git-for-windows.github.io/)
  - [Python 2.7.14 or Python 3.6.3](https://www.python.org/downloads/)
  - [LabelImg](https://github.com/tzutalin/labelImg)

    Install by `pip install labelImg`
      - [PyQt4](https://www.riverbankcomputing.com/software/pyqt/download)
        `pip install PyQt4-4.11.4-cp27-cp27m-win32.whl`
      - [lxml](http://lxml.de/installation.html)
        `pip install lxml`

  - [TensorFlow's Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)

#### Download the Application

To run a copy from the latest master branch in git you can clone the repository:

```
git clone git@github.com:weizh888/ProductImageSegmentation.git
```

## Usage

### Creating the dataset

#### Manually Labelling

Create labels based on (https://shopee.sg/search/?keyword=women+apparel&subcategory).

simply run
```
LabelImg
```
> Open Dir, Change Save Dir, View->Autosaving
```
labelImg [IMAGE_PATH] [PRE-DEFINED CLASS FILE] (It doesn't work for now)
```

#### Generating TFRecord files for TensorFlow

### Training the model
