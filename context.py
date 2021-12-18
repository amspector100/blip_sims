import os
import sys


# Add hpt/knockpy to path
file_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.split(file_directory)[0]
sys.path.insert(0, os.path.abspath(parent_directory + '/pyblip'))

# Ensure we are using the right version of knockpy
import pyblip