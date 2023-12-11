# Urban Studio

![Urban Studio Logo](imgs/logo.png)

Urban Studio is a powerful and user-friendly toolkit for urban design and analysis. It provides a suite of tools and functionalities to aid in urban environment analysis, from image segmentation for building detection to planform generation and 3D reconstruction. Additionally, the project includes a graphical user interface (GUI) for easy and convenient operation.

## 📚 Table of Contents

- [Urban Studio](#urban-studio)
  - [📚 Table of Contents](#-table-of-contents)
  - [🚀 Features](#-features)
  - [🛠 Installation](#-installation)
  - [🖥 Usage](#-usage)
  - [📖 Documentation](#-documentation)
  - [🤝 Contributing](#-contributing)
  - [📝 To-Do List](#-to-do-list)
  - [📄 License](#-license)

## 🚀 Features

- **Image Segmentation**: Quickly and accurately segment urban environments to analyze the percentage of buildings in the area.

- **Graphical User Interface (GUI)**: Urban Studio features an intuitive GUI for easy and efficient interaction with the toolkit.

## 🛠 Installation

To use Urban Studio, follow these steps:

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/h-tr/urban-studio.git
   ```

2. Create virtual environment (conda is recommanded)

   ```bash
   conda create -n urban-studio -y python=3.10
   conda activate urban-studio
   ```

3. Install the required dependencies and build the project by running:

   ```bash
   pip install -e .
   ```

   We are using mmcv as dependency, so we need to install the pretrained model and config files
   
   ```bash
   mim download mmsegmentation --config segformer_mit-b4_8xb2-160k_ade20k-512x512 --dest src/models
   ```

   By default, we are using the segformer trained on ade20k dataset. Feel free to try other models!

4. Launch the application by running:

   ```bash
   python urban_studio.py
   ```

## 🖥 Usage

1. **Image Segmentation**: Open an urban image in the Urban Studio GUI. Click the "Segment" button to perform image segmentation and analyze the building percentage.

2. **Planform Generation**: Use the "Planform Generator" feature to create various urban layouts based on your design parameters.

3. **3D Reconstruction**: Load 2D urban images and convert them into detailed 3D models using the "3D Reconstruction" tool.

For detailed instructions and tutorials on how to use specific features, please refer to the [documentation](#documentation).

## 📖 Documentation

For comprehensive documentation and detailed guides on using Urban Studio, please visit our [Documentation](/docs) folder. You can find in-depth information on each feature, as well as usage examples.

## 🤝 Contributing

We welcome contributions from the community. If you'd like to contribute to Urban Studio, please follow our [Contributing Guidelines](CONTRIBUTING.md). Your input and enhancements are greatly appreciated!

## 📝 To-Do List

Here's a list of planned features and improvements for Urban Studio:

- [ ] Implement a feature for urban green space analysis.
- [ ] Create tutorials for advanced use cases.

Feel free to contribute to any of these tasks or propose your own!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.