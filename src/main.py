import os
import yaml
import argparse
from utils.segmentation_utils import ade_classes
from core.segmentation import inference

import warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config file",
        default="configs/segmentation.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r") as fin:
        config = yaml.safe_load(fin)

    input_path = config["input_path"]
    output_path = config["output_path"]
    model_config_file = config["model"]["config_file"]
    model_checkpoint_file = config["model"]["checkpoint_file"]
    device = config["model"]["device"]

    segmentor = inference.Segmentor(
        model_config_file, model_checkpoint_file, output_path, device
    )
    # Read all the images in the input path
    for image in os.listdir(input_path):
        image_path = os.path.join(input_path, image)
        # If you want to visualize the segmentation results, set display = True
        result = segmentor.inference(image_path, display = False)
        print("=====================================================")
        print(f"Image: {image_path}")
        for id in inference.Segmentor.interested_classes():
            percentage = segmentor.get_percentage(id, result)
            if percentage > 0:
                print(f"\tThe percentage of {ade_classes()[id]} is {100 * percentage: .2f}%")


if __name__ == "__main__":
    main()
