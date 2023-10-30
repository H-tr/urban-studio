import unittest
from src.core import segmentation
from PIL import Image
import requests


class TestSegmentation(unittest.TestCase):
    def test_seg_image(self):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        img = segmentation.seg_image(image)
