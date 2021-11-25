import os
import sys


# Add hpt/knockpy to path
file_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.split(file_directory)[0]
sys.path.insert(0, os.path.abspath(parent_directory + '/hpt'))
sys.path.insert(0, os.path.abspath(parent_directory + '/pyblip'))
sys.path.insert(0, os.path.abspath(parent_directory + '/knockpy'))

# Ensure we are using the right version of knockpy
import pyblip
import hpt
import knockpy
print(f"Using version {knockpy.__version__} of knockpy")