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
                            '# In[',
                            '# <hr>',
                            'from IPython.display import Image',
                            'get_ipython()',
                            '# <br>')

    clean_content = []
    imports = []
    existing_imports = set()
    with open(path, 'r') as f:
        next(f)
        next(f)
        for line in f:
            line = line.rstrip(' ')
            if line.startswith(skip_lines_startwith):
                continue
            if line.startswith('import ') or (
                    'from ' in line and 'import ' in line):
                if 'from __future__ import print_function' in line:
                    if line != imports[0]:
                        imports.insert(0, line)
                else:
                    if line.strip() not in existing_imports:
                        imports.append(line)
                        existing_imports.add(line.strip())
            else:
                clean_content.append(line)

    clean_content = ['# coding: utf-8\n\n\n'] + imports + clean_content

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
