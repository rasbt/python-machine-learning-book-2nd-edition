import unittest
import os
import subprocess
import tempfile
import watermark
import sys


def run_ipynb(path):
    if (sys.version_info >= (3, 0)):
        kernel_name = 'python3'
    else:
        kernel_name = 'python2'
    error_cells = []
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to",
                "notebook", "--execute",
                "--ExecutePreprocessor.kernel_name=%s" % kernel_name,
                "--output", fout.name, path]
        subprocess.check_output(args)


class TestNotebooks(unittest.TestCase):

    def test_appendix_g_tensorflow_basics(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_ipynb(os.path.join(this_dir,
                               '../ch01/ch01.ipynb',
                               '../ch02/ch02.ipynb'))


if __name__ == '__main__':
    unittest.main()
