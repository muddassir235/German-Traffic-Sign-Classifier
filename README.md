## German Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Overview

In this project, I used deep neural networks and convolutional neural networks to classify traffic signs. I trained my model using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After training I was able to get a [validation set](https://en.wikipedia.org/wiki/Test_set#Validation_set) accuracy of approximately 99.8% and test set Accuracy of **95.8%.**

![](https://github.com/muddassir235/German-Traffic-Sign-Classifier/blob/master/Files/conv_net.png?raw=true)
### Model Specs
- A model of about **13 million** neurons
- It has **7 layers**, two of which are convolutional, four are fully connected and an output layer.
- **Relu** is used as an activation function throughout.
- It has a **dropout** regularization of **0.5** throughout the fully connected layers.
- **L2** regularization of **1e-6** is also applied.
- **[Batch Normalization](http://jmlr.org/proceedings/papers/v37/ioffe15.pdf)** is also employed throughout the network, in order to get better regularization and normalization. It also helps the network converge more quickly and makes it more independent of it's initial parameters.
- To get numerical stability each channel of each pixel is divided by the maximum value i.e. **255**.
- The **mean image** of the training dataset is subtracted from all the image to achieve normalization.
- **[Xavier Initializer](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)** is used to initialize all the weights in the network.

#### **Details of Layers**
- _**Layer 1**_: Convolutional (30 5x5 filers)
- _**Layer 2**_: Convolutional (200 5x5 filters)
- _**Layer 3**_: Fully connected (2200 depth)
- _**Layer 4**_: Fully connected (1000 depth)
- _**Layer 5**_: Fully connected (500 depth)
- _**Layer 6**_: Fully connected (120 depth)
- _**Layer 7**_: Output Layer (43 as we have 43 different classes of traffic signs.)

### Dependencies

This project requires **Python 3.5** and the following Python libraries installed:

- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/)
- [TensorFlow](http://tensorflow.org)
- [Matplotlib](http://matplotlib.org/)
- [Pandas](http://pandas.pydata.org/) (Optional)

Run this command at the terminal prompt to install [OpenCV](http://opencv.org/). Useful for image processing:

- `conda install -c https://conda.anaconda.org/menpo opencv3`

### Dataset

1. [Download the dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip). This is a pickled dataset in which we've already resized the images to 32x32.
2. Clone the project and start the notebook.
```
git clone https://github.com/muddassir235/German-Traffic-Sign-Classifier
cd CarND-Traffic-Signs
jupyter notebook Traffic_Signs_Recognition.ipynb
```
3. You can now run and modify the code in python jupyter notebook :-).
4. Please feel free to make a pull request for sharing improvements to the code. That will make me realy happy :-).

### License
This project is distributed under the Apache 2.0 [license](https://github.com/muddassir235/German-Traffic-Sign-Classifier/blob/master/license.md).
