Python Machine Learning - Code Examples


##  Chapter 15 - Classifying Images with Deep Convolutional Neural Networks

### Chapter Outline

- Building blocks of convolutional neural networks
  - Understanding CNNs and learning feature hierarchies
  - Performing discrete convolutions
    - Performing a discrete convolution in one dimension
    - The effect of zero-padding in a convolution
    - Determining the size of the convolution output
    - Performing a discrete convolution in 2D
  - Subsampling
- Putting everything together to build a CNN 508
  - Working with multiple input or color channels 508
  - Regularizing a neural network with dropout 512
- Implementing a deep convolutional neural network using TensorFlow
  - The multilayer CNN architecture 514 Loading and preprocessing the data
  - Implementing a CNN in the TensorFlow low-level API
  - Implementing a CNN in the TensorFlow layers API
- Summary

### A note on using the code examples

The recommended way to interact with the code examples in this book is via Jupyter Notebook (the `.ipynb` files). Using Jupyter Notebook, you will be able to execute the code step by step and have all the resulting outputs (including plots and images) all in one convenient document.

![](../ch02/images/jupyter-example-1.png)



Setting up Jupyter Notebook is really easy: if you are using the Anaconda Python distribution, all you need to install jupyter notebook is to execute the following command in your terminal:

    conda install jupyter notebook

Then you can launch jupyter notebook by executing

    jupyter notebook

A window will open up in your browser, which you can then use to navigate to the target directory that contains the `.ipynb` file you wish to open.

**More installation and setup instructions can be found in the [README.md file of Chapter 1](../ch01/README.md)**.

**(Even if you decide not to install Jupyter Notebook, note that you can also view the notebook files on GitHub by simply clicking on them: [`ch15.ipynb`](ch15.ipynb))**

In addition to the code examples, I added a table of contents to each Jupyter notebook as well as section headers that are consistent with the content of the book. Also, I included the original images and figures in hope that these make it easier to navigate and work with the code interactively as you are reading the book.

![](../ch02/images/jupyter-example-2.png)


When I was creating these notebooks, I was hoping to make your reading (and coding) experience as convenient as possible! However, if you don't wish to use Jupyter Notebooks, I also converted these notebooks to regular Python script files (`.py` files) that can be viewed and edited in any plaintext editor. 