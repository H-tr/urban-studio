import yaml
import argparse
from core.segmentation import inference


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
    segmentor.inference(input_path, True)


if __name__ == "__main__":
    main()
