import os
import sys

# Add blip_sims to path
file_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.split(file_directory)[0]
sys.path.insert(0, os.path.abspath(parent_directory))
import blip_sims

# Get rid of this when pyblip is published
grandparent_dir = os.path.split(parent_directory)[0]
sys.path.insert(0, os.path.abspath(grandparent_dir + '/pyblip'))
import pyblip