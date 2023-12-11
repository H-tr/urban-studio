import os

import cv2
import mmcv
import torch
import numpy as np
from utils.segmentation_utils import wait_continue, ade_classes, ade_palette
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
from mmseg.apis import inference_model, init_model


class Segmentor:
    """Segmentor class for performing semantic segmentation using a pre-trained model.

    Args:
        config_file (str): Path to the model's configuration file.
        checkpoint_file (str): Path to the model's checkpoint file.
        save_dir (str, optional): Directory to save the segmentation results. Defaults to None.
        device (str, optional): Device to use for inference (e.g., "cuda:0" for GPU). Defaults to "cuda:0".

    Attributes:
        config_file (str): Path to the model's configuration file.
        checkpoint_file (str): Path to the model's checkpoint file.
        save_dir (str): Directory to save the segmentation results.
        device (str): Device to use for inference.
        model: Initialized segmentation model.

    """

    def __init__(
        self,
        config_file: str,
        checkpoint_file: str,
        save_dir: str = None,
        device: str = "cuda:0",
    ) -> None:
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.save_dir = save_dir
        self.fig_show_cfg = dict(frameon=False)
        if self.save_dir is not None and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.device = device
        self.model = init_model(
            self.config_file, self.checkpoint_file, device=self.device
        )

    @staticmethod
    def interested_classes() -> list:
        """Get the list of interested classes.

        Returns:
            list: List of interested classes.

        """
        return [1, 2, 4, 6, 9, 11, 13, 16, 17, 21]

    def set_model(self, config_file: str, checkpoint_file: str):
        """Set the model with new configuration and checkpoint files.

        Args:
            config_file (str): Path to the new model's configuration file.
            checkpoint_file (str): Path to the new model's checkpoint file.

        """
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.model = init_model(
            self.config_file, self.checkpoint_file, device=self.device
        )

    def inference(self, img, display: bool = False) -> SegDataSample:
        """Perform inference on an input image.

        Args:
            img: Input image for inference.
            display (bool, optional): Whether to display the segmentation results. Defaults to False.

        Returns:
            SegDataSample: Segmentation result.

        """
        result = inference_model(self.model, img)

        if self.save_dir is not None:
            # append img name to save_dir
            out_file = os.path.join(self.save_dir, img.split("/")[-1])
            # visualize the results in a new window
            # you can change the opacity of the painted segmentation map in (0, 1].

            self.get_result(img, result, show=display, out_file=out_file, opacity=0.5)
        return result

    def get_percentage(self, id, result: SegDataSample) -> float:
        """Get the percentage of the semantic segmentation.

        Args:
            id (int): Class id.
            result (SegDataSample): Segmentation result.

        Returns:
            float: Percentage of the semantic segmentation.

        """
        sem_seg = result.pred_sem_seg.cpu().data
        mask = sem_seg[0] == id
        return np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])

    def get_result(
        self,
        img: str,
        result: SegDataSample,
        show: bool = False,
        out_file: str = None,
        opacity: float = 0.5,
    ):
        """Visualize the results.

        Args:
            img (str): Path to the input image.
            result (SegDataSample): Segmentation result.
            show (bool, optional): Whether to display the segmentation results. Defaults to False.
            out_file (str, optional): Path to save the segmentation result. Defaults to None.
            opacity (float, optional): Opacity of the painted segmentation map. Defaults to 0.5.

        """
        image = mmcv.imread(img, channel_order="rgb")
        classes = ade_classes()
        pred_img_data = self._draw_sem_seg(image, result.pred_sem_seg, classes, opacity)
        if self.save_dir is not None:
            mmcv.imwrite(mmcv.rgb2bgr(pred_img_data), out_file)
        if show:
            self.show(pred_img_data)

    def _get_center_loc(self, mask: np.ndarray) -> np.ndarray:
        """Get the center coordinate of the semantic segmentation.

        Args:
            mask (np.ndarray): Semantic segmentation mask.

        Returns:
            np.ndarray: Center coordinate of the semantic segmentation.

        """
        loc = np.argwhere(mask == 1)

        loc_sort = np.array(sorted(loc.tolist(), key=lambda row: (row[0], row[1])))
        y_list = loc_sort[:, 0]
        unique, _, counts = np.unique(y_list, return_index=True, return_counts=True)
        y_loc = unique[counts.argmax()]
        y_most_freq_loc = loc[loc_sort[:, 0] == y_loc]
        center_num = len(y_most_freq_loc) // 2
        x = y_most_freq_loc[center_num][1]
        y = y_most_freq_loc[center_num][0]
        return np.array([x, y])

    def _draw_sem_seg(
        self,
        image: np.ndarray,
        sem_seg: PixelData,
        classes: None | list,
        opacity: float = 0.5,
    ):
        """Draw the semantic segmentation on the input image.

        Args:
            image (np.ndarray): Input image.
            sem_seg (PixelData): Semantic segmentation result.
            classes (None | list): List of class names.
            opacity (float, optional): Opacity of the painted segmentation map. Defaults to 0.5.

        Returns:
            np.ndarray: Image with the painted segmentation map.

        """
        num_classes = len(classes)
        sem_seg = sem_seg.cpu().data
        ids = np.unique(sem_seg)[::-1]
        legal_indices = ids < num_classes
        ids = ids[legal_indices]
        ids = np.array([id for id in ids if id in Segmentor.interested_classes()])
        labels = np.array(ids, dtype=np.int64)
        palette = ade_palette()
        colors = [palette[label] for label in labels]

        mask = np.zeros_like(image, dtype=np.uint8)
        for label, color in zip(labels, colors):
            mask[sem_seg[0] == label, :] = color

        font = cv2.FONT_HERSHEY_SIMPLEX
        # (0,1] to change the size of the text relative to the image
        scale = 0.05
        fontScale = min(image.shape[0], image.shape[1]) / (25 / scale)
        fontColor = (255, 255, 255)
        if image.shape[0] < 300 or image.shape[1] < 300:
            thickness = 1
            rectangleThickness = 1
        else:
            thickness = 2
            rectangleThickness = 2
        lineType = 2

        if isinstance(sem_seg[0], torch.Tensor):
            masks = sem_seg[0].numpy() == labels[:, None, None]
        else:
            masks = sem_seg[0] == labels[:, None, None]
        masks = masks.astype(np.uint8)
        for mask_num in range(len(labels)):
            classes_id = labels[mask_num]
            classes_color = colors[mask_num]
            loc = self._get_center_loc(masks[mask_num])
            text = classes[classes_id]
            (label_width, label_height), baseline = cv2.getTextSize(
                text, font, fontScale, thickness
            )
            mask = cv2.rectangle(
                mask,
                loc,
                (loc[0] + label_width + baseline, loc[1] + label_height + baseline),
                classes_color,
                -1,
            )
            mask = cv2.rectangle(
                mask,
                loc,
                (loc[0] + label_width + baseline, loc[1] + label_height + baseline),
                (0, 0, 0),
                rectangleThickness,
            )
            mask = cv2.putText(
                mask,
                text,
                (loc[0], loc[1] + label_height),
                font,
                fontScale,
                fontColor,
                thickness,
                lineType,
            )
        color_seg = (image * (1 - opacity) + mask * opacity).astype(np.uint8)
        return color_seg

    def _init_manager(self, win_name: str) -> None:
        """Initialize the matplot manager.

        Args:
            win_name (str): The window name.
        """
        from matplotlib.figure import Figure
        from matplotlib.pyplot import new_figure_manager

        if getattr(self, "manager", None) is None:
            self.manager = new_figure_manager(
                num=1, FigureClass=Figure, **self.fig_show_cfg
            )

        try:
            self.manager.set_window_title(win_name)
        except Exception:
            self.manager = new_figure_manager(
                num=1, FigureClass=Figure, **self.fig_show_cfg
            )
            self.manager.set_window_title(win_name)

    def show(
        self,
        drawn_img: None | np.ndarray,
        win_name: str = "image",
        wait_time: float = 0.0,
        continue_key: str = " ",
        backend: str = "matplotlib",
    ) -> None:
        """Show the drawn image.

        Args:
            drawn_img (np.ndarray, optional): The image to show. If drawn_img
                is None, it will show the image got by Visualizer. Defaults
                to None.
            win_name (str):  The image title. Defaults to 'image'.
            wait_time (float): Delay in seconds. 0 is the special
                value that means "forever". Defaults to 0.
            continue_key (str): The key for users to continue. Defaults to
                the space key.
            backend (str): The backend to show the image. Defaults to
                'matplotlib'. `New in version 0.7.3.`
        """
        if backend == "matplotlib":
            import matplotlib.pyplot as plt

            is_inline = "inline" in plt.get_backend()
            img = self.get_image() if drawn_img is None else drawn_img
            self._init_manager(win_name)
            fig = self.manager.canvas.figure
            # remove white edges by set subplot margin
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            fig.clear()
            ax = fig.add_subplot()
            ax.axis(False)
            ax.imshow(img)
            self.manager.canvas.draw()

            # Find a better way for inline to show the image
            if is_inline:
                return fig
            wait_continue(fig, timeout=wait_time, continue_key=continue_key)
        elif backend == "cv2":
            # Keep images are shown in the same window, and the title of window
            # will be updated with `win_name`.
            cv2.namedWindow(winname=f"{id(self)}")
            cv2.setWindowTitle(f"{id(self)}", win_name)
            cv2.imshow(
                str(id(self)), self.get_image() if drawn_img is None else drawn_img
            )
            cv2.waitKey(int(np.ceil(wait_time * 1000)))
        else:
            raise ValueError(
                'backend should be "matplotlib" or "cv2", ' f"but got {backend} instead"
            )
