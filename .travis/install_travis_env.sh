#!/usr/bin/env bash

# Installing MINICONDA

set -e

echo ${TRAVIS_PYTHON_VERSION};

if [ "${TRAVIS_PYTHON_VERSION}" == "2.7" ]; then
  wget http://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh;
else
  wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
fi


bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a


# Setting up the Test Environment

if [ "${LATEST}" = "true" ]; then
  conda create -q -n testenv --yes python=$TRAVIS_PYTHON_VERSION numpy scipy matplotlib scikit-learn pandas nltk;
else
  conda create -q -n testenv --yes python=$TRAVIS_PYTHON_VERSION numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION matplotlib=$MATPLOTLIB_VERSION scikit-learn=$SKLEARN_VERSION pandas=$PANDAS_VERSION nltk=$NLTK_VERSION;
fi

source activate testenv

conda install -q -y pip jupyter notebook nbconvert;

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import matplotlib; print('matplotlib %s' % matplotlib.__version__)"
python -c "import sklearn; print('sklearn %s' % sklearn.__version__)"
python -c "import pandas; print('pandas %s' % pandas.__version__)"


pip install --upgrade pip

if [ "${COVERAGE}" = "true" ]; then
    pip install coveralls
fi

pip install watermark;
pip install pyprind;
pip install nbformat;
pip install pydotplus;
pip install seaborn;
pip install pillow;

if [ "${LATEST}" = "true" ]; then
  pip install tensorflow;
else
  pip install tensorflow==$TENSORFLOW_VERSION;
fi

python -c "import tensorflow; print('tensorflow %s' % tensorflow.__version__)"
python -c "import os; print(os.environ)"

