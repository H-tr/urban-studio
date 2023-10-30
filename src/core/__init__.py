import sys
import os

# Get the path to the project's root directory (urban_studio)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
base_path = os.path.join(project_root, "urban_studio")
utils_path = os.path.join(project_root, "utils")

# Check if the base folder path is not already in sys.path
if base_path not in sys.path:
    sys.path.append(base_path)

# Check if the utils folder path is not already in sys.path
if utils_path not in sys.path:
    sys.path.append(utils_path)
