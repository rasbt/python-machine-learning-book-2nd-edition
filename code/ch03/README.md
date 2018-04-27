Python Machine Learning - Code Examples


##  Chapter 3: A Tour of Machine Learning Classifiers Using scikit-learn

### Chapter Outline

- Choosing a classification algorithm
- First steps with scikit-learn -- training a perceptron
- Modeling class probabilities via logistic regression
  - Logistic regression intuition and conditional probabilities
  - Learning the weights of the logistic cost function
  - Converting an Adaline implementation into an algorithm for logistic regression
  - Training a logistic regression model with scikit-learn
  - Tackling over tting via regularization
- Maximum margin classification with support vector machines
  - Maximum margin intuition
  - Dealing with a nonlinearly separable case using slack variables
  - Alternative implementations in scikit-learn
- Solving nonlinear problems using a kernel SVM
  - Kernel methods for linearly inseparable data
  - Using the kernel trick to find separating hyperplanes in high-dimensional space 
- Decision tree learning
  - Maximizing information gain – getting the most bang for your buck
  - Building a decision tree
  - Combining multiple decision trees via random forests
- K-nearest neighbors – a lazy learning algorithm
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

**(Even if you decide not to install Jupyter Notebook, note that you can also view the notebook files on GitHub by simply clicking on them: [`ch03.ipynb`](ch03.ipynb))**

In addition to the code examples, I added a table of contents to each Jupyter notebook as well as section headers that are consistent with the content of the book. Also, I included the original images and figures in hope that these make it easier to navigate and work with the code interactively as you are reading the book.

![](../ch02/images/jupyter-example-2.png)


When I was creating these notebooks, I was hoping to make your reading (and coding) experience as convenient as possible! However, if you don't wish to use Jupyter Notebooks, I also converted these notebooks to regular Python script files (`.py` files) that can be viewed and edited in any plaintext editor. 