# Simple helper script to convert
# a Jupyter notebook to Python
#
# Sebastian Raschka, 2017


import argparse
import os
import subprocess


def convert(input_path, output_path):
    subprocess.call(['jupyter', 'nbconvert', '--to', 'script',
                     input_path, '--output', output_path])


def cleanup(path):

    skip_lines_startwith = ('Image(filename=',
                            'get_ipython()',
                            '# <br>',
                            'from __future__ import print_function')

    clean_content = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith(skip_lines_startwith):
                if line.startswith('from __future__ import print_function'):
                    clean_content.insert(0,
                                         'from __future__ import '
                                         'print_function\n\n\n')
                continue
            else:
                clean_content.append(line)

    with open(path, 'w') as f:
        for line in clean_content:
            f.write(line)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            description='Convert Jupyter notebook to Python script.',
            formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--input',
                        required=True,
                        help='Path to the Jupyter Notebook file')

    parser.add_argument('-o', '--output',
                        required=True,
                        help='Path to the Python script file')

    parser.add_argument('-v', '--version',
                        action='version',
                        version='v. 0.1')

    args = parser.parse_args()

    convert(input_path=args.input,
            output_path=os.path.splitext(args.output)[0])

    cleanup(args.output)
