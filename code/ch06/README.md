Python Machine Learning - Code Examples


##  Chapter 6: Learning Best Practices for Model Evaluation and Hyperparameter Tuning

### Chapter Outline

- Streamlining workflows with pipelines
  - Loading the Breast Cancer Wisconsin dataset
  - Combining transformers and estimators in a pipeline
- Using k-fold cross-validation to assess model performance
  - The holdout method
  - K-fold cross-validation
- Debugging algorithms with learning and validation curves
  - Diagnosing bias and variance problems with learning curves
  - Addressing over- and underfitting with validation curves
- Fine-tuning machine learning models via grid search
  - Tuning hyperparameters via grid search
  - Algorithm selection with nested cross-validation
- Looking at different performance evaluation metrics
  - Reading a confusion matrix
  - Optimizing the precision and recall of a classification model
  - Plotting a receiver operating characteristic
  - Scoring metrics for multiclass classification
- Dealing with class imbalance
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

**(Even if you decide not to install Jupyter Notebook, note that you can also view the notebook files on GitHub by simply clicking on them: [`ch06.ipynb`](ch06.ipynb))**

In addition to the code examples, I added a table of contents to each Jupyter notebook as well as section headers that are consistent with the content of the book. Also, I included the original images and figures in hope that these make it easier to navigate and work with the code interactively as you are reading the book.

![](../ch02/images/jupyter-example-2.png)


When I was creating these notebooks, I was hoping to make your reading (and coding) experience as convenient as possible! However, if you don't wish to use Jupyter Notebooks, I also converted these notebooks to regular Python script files (`.py` files) that can be viewed and edited in any plaintext editor. 