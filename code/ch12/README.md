Python Machine Learning - Code Examples


##  Chapter 12: Implementing a Multilayer Arti cial Neural
Network from Scratch

### Chapter Outline

- Modeling complex functions with arti cial neural networks
  - Single-layer neural network recap
  - Introducing the multilayer neural network architecture
  - Activating a neural network via forward propagation
- Classifying handwritten digits
  - Obtaining the MNIST dataset
  - Implementing a multilayer perceptron
- Training an artificial neural network
  - Computing the logistic cost function
  - Developing your intuition for backpropagation
  - Training neural networks via backpropagation
- About the convergence in neural networks
- A few last words about the neural network implementation
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

**(Even if you decide not to install Jupyter Notebook, note that you can also view the notebook files on GitHub by simply clicking on them: [`ch12.ipynb`](ch12.ipynb))**

In addition to the code examples, I added a table of contents to each Jupyter notebook as well as section headers that are consistent with the content of the book. Also, I included the original images and figures in hope that these make it easier to navigate and work with the code interactively as you are reading the book.

![](../ch02/images/jupyter-example-2.png)


When I was creating these notebooks, I was hoping to make your reading (and coding) experience as convenient as possible! However, if you don't wish to use Jupyter Notebooks, I also converted these notebooks to regular Python script files (`.py` files) that can be viewed and edited in any plaintext editor. 