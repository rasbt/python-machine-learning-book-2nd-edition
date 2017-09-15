import unittest
import os
import subprocess
import sys
import logging


LOG_FORMAT = '[%(asctime)s %(levelname)s] %(message)s'
LOGGER = logging.getLogger(__file__)


def run_ipynb(path):

    nb_dir = os.path.dirname(path)
    nb_path = os.path.basename(path)
    orig_dir = os.getcwd()
    os.chdir(nb_dir)

    if (sys.version_info >= (3, 0)):
        kernel_name = 'python3'
    else:
        kernel_name = 'python2'
    #  error_cells = []

    args = ["jupyter", "nbconvert",
            "--execute", "--inplace",
            "--debug",
            "--ExecutePreprocessor.timeout=5000",
            "--ExecutePreprocessor.kernel_name=%s" % kernel_name,
            nb_path]

    if (sys.version_info >= (3, 0)):
        try:
            subprocess.check_output(args)
        except TimeoutError:
            sys.stderr.write('%s timed out\n' % path)
            sys.stderr.flush()

    else:
        subprocess.check_output(args)

    os.chdir(orig_dir)


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

    def test_ch08(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))

        # run only on Py3, because of Py2 unicode handling
        if (sys.version_info >= (3, 0)):
            run_ipynb(os.path.join(this_dir,
                                   '../ch08/ch08.ipynb'))

    def test_ch09(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))

        # run only on Py3, because of the Py3 specific pickle files
        if (sys.version_info >= (3, 0)):
            run_ipynb(os.path.join(this_dir, '../ch09/ch09.ipynb'))
        else:
            pass

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
        run_ipynb(os.path.join(this_dir,
                               '../ch12/ch12.ipynb'))

    def test_ch13(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_ipynb(os.path.join(this_dir,
                               '../ch13/ch13.ipynb'))

    def test_ch14(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_ipynb(os.path.join(this_dir,
                               '../ch14/ch14.ipynb'))

    def test_ch15(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        run_ipynb(os.path.join(this_dir,
                               '../ch15/ch15.ipynb'))

    def test_ch16(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        # run only on Py3, because of Py2 unicode handling
        if (sys.version_info >= (3, 0)):
            run_ipynb(os.path.join(this_dir,
                                   '../ch16/ch16.ipynb'))


if __name__ == '__main__':
    unittest.main()
