import unittest
import os
import subprocess
import tempfile
import sys


def run_ipynb(path):
    if (sys.version_info >= (3, 0)):
        kernel_name = 'python3'
    else:
        kernel_name = 'python2'
    # error_cells = []
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to",
                "notebook", "--execute",
                "--ExecutePreprocessor.kernel_name=%s" % kernel_name,
                "--output", fout.name, path]
        subprocess.check_output(args)


class TestNotebooks(unittest.TestCase):

    def test_ch01(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_ipynb(os.path.join(this_dir,
                               '../ch01/ch01.ipynb'))

    def test_ch02(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_ipynb(os.path.join(this_dir,
                               '../ch02/ch02.ipynb'))

    def test_ch03(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_ipynb(os.path.join(this_dir,
                               '../ch03/ch03.ipynb'))

    def test_ch04(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_ipynb(os.path.join(this_dir,
                               '../ch04/ch04.ipynb'))

    def test_ch05(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_ipynb(os.path.join(this_dir,
                               '../ch05/ch05.ipynb'))

    def test_ch06(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_ipynb(os.path.join(this_dir,
                               '../ch06/ch06.ipynb'))

    def test_ch07(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_ipynb(os.path.join(this_dir,
                               '../ch07/ch07.ipynb'))

    # add 8 & 9

    def test_ch10(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_ipynb(os.path.join(this_dir,
                               '../ch10/ch10.ipynb'))

    def test_ch11(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_ipynb(os.path.join(this_dir,
                               '../ch11/ch11.ipynb'))

    def test_ch12(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        for gz in ['t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz',
                   'train-images-idx3-ubyte.gz',
                   'train-labels-idx1-ubyte.gz']:
            gz_path = os.path.join(this_dir, '../ch12/%s' % gz)
        subprocess.call(['gunzip', gz_path])
        run_ipynb(os.path.join(this_dir,
                               '../ch12/ch12.ipynb'))


if __name__ == '__main__':
    unittest.main()
