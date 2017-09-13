Python Machine Learning - Code Examples


##  Chapter 4: Building Good Training Sets – Data Preprocessing

### Chapter Outline

- Dealing with missing data
  - Identifying missing values in tabular data
  - Eliminating samples or features with missing values
  - Imputing missing values
  - Understanding the scikit-learn estimator API
- Handling categorical data
  - Nominal and ordinal features
  - Creating an example dataset
  - Mapping ordinal features
  - Encoding class labels
  - Performing one-hot encoding on nominal features
- Partitioning a dataset into separate training and test sets
- Bringing features onto the same scale
- Selecting meaningful features
  - L1 and L2 regularization as penalties against model complexity
  - A geometric interpretation of L2 regularization
  - Sparse solutions with L1 regularization
  - Sequential feature selection algorithms
- Assessing feature importance with random forests
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

**(Even if you decide not to install Jupyter Notebook, note that you can also view the notebook files on GitHub by simply clicking on them: [`ch04.ipynb`](ch04.ipynb))**

In addition to the code examples, I added a table of contents to each Jupyter notebook as well as section headers that are consistent with the content of the book. Also, I included the original images and figures in hope that these make it easier to navigate and work with the code interactively as you are reading the book.

![](../ch02/images/jupyter-example-2.png)


When I was creating these notebooks, I was hoping to make your reading (and coding) experience as convenient as possible! However, if you don't wish to use Jupyter Notebooks, I also converted these notebooks to regular Python script files (`.py` files) that can be viewed and edited in any plaintext editor. 