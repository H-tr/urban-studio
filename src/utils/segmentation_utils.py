def wait_continue(figure, timeout: float = 0, continue_key: str = " ") -> int:
    """Show the image and wait for the user's input.

    This implementation refers to
    https://github.com/matplotlib/matplotlib/blob/v3.5.x/lib/matplotlib/_blocking_input.py

    Args:
        timeout (float): If positive, continue after ``timeout`` seconds.
            Defaults to 0.
        continue_key (str): The key for users to continue. Defaults to
            the space key.

    Returns:
        int: If zero, means time out or the user pressed ``continue_key``,
            and if one, means the user closed the show figure.
    """  # noqa: E501
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import CloseEvent

    is_inline = "inline" in plt.get_backend()
    if is_inline:
        # If use inline backend, interactive input and timeout is no use.
        return 0

    if figure.canvas.manager:  # type: ignore
        # Ensure that the figure is shown
        figure.show()  # type: ignore

    while True:
        # Connect the events to the handler function call.
        event = None

        def handler(ev):
            # Set external event variable
            nonlocal event
            # Qt backend may fire two events at the same time,
            # use a condition to avoid missing close event.
            event = ev if not isinstance(event, CloseEvent) else event
            figure.canvas.stop_event_loop()

        cids = [
            figure.canvas.mpl_connect(name, handler)  # type: ignore
            for name in ("key_press_event", "close_event")
        ]

        try:
            figure.canvas.start_event_loop(timeout)  # type: ignore
        finally:  # Run even on exception like ctrl-c.
            # Disconnect the callbacks.
            for cid in cids:
                figure.canvas.mpl_disconnect(cid)  # type: ignore

        if isinstance(event, CloseEvent):
            return 1  # Quit for close.
        elif event is None or event.key == continue_key:
            return 0  # Quit for continue.


def ade_classes():
    """ADE20K class names for external use."""
    return [
        "wall",
        "building",
        "sky",
        "floor",
        "tree",  # 0-4
        "ceiling",
        "road",
        "bed",
        "windowpane",
        "grass",  # 5-9
        "cabinet",
        "sidewalk",
        "person",
        "earth",
        "door",  # 10-14
        "table",
        "mountain",
        "plant",
        "curtain",
        "chair",  # 15-19
        "car",
        "water",
        "painting",
        "sofa",
        "shelf",  # 20-24
        "house",
        "sea",
        "mirror",
        "rug",
        "field",  # 25-29
        "armchair",
        "seat",
        "fence",
        "desk",
        "rock",  # 30-34
        "wardrobe",
        "lamp",
        "bathtub",
        "railing",
        "cushion",  # 35-39
        "base",
        "box",
        "column",
        "signboard",
        "chest of drawers",  # 40-44
        "counter",
        "sand",
        "sink",
        "skyscraper",
        "fireplace",  # 45-49
        "refrigerator",
        "grandstand",
        "path",
        "stairs",
        "runway",  # 50-54
        "case",
        "pool table",
        "pillow",
        "screen door",
        "stairway",  # 55-59
        "river",
        "bridge",
        "bookcase",
        "blind",
        "coffee table",  # 60-64
        "toilet",
        "flower",
        "book",
        "hill",
        "bench",  # 65-69
        "countertop",
        "stove",
        "palm",
        "kitchen island",
        "computer",  # 70-74
        "swivel chair",
        "boat",
        "bar",
        "arcade machine",
        "hovel",  # 75-79
        "bus",
        "towel",
        "light",
        "truck",
        "tower",  # 80-84
        "chandelier",
        "awning",
        "streetlight",
        "booth",
        "television receiver",  # 85-89
        "airplane",
        "dirt track",
        "apparel",
        "pole",
        "land",  # 90-94
        "bannister",
        "escalator",
        "ottoman",
        "bottle",
        "buffet",  # 95-99
        "poster",
        "stage",
        "van",
        "ship",
        "fountain",  # 100-104
        "conveyer belt",
        "canopy",
        "washer",
        "plaything",
        "swimming pool",  # 105-109
        "stool",
        "barrel",
        "basket",
        "waterfall",
        "tent",  # 110-114
        "bag",
        "minibike",
        "cradle",
        "oven",
        "ball",  # 115-119
        "food",
        "step",
        "tank",
        "trade name",
        "microwave",  # 120-124
        "pot",
        "animal",
        "bicycle",
        "lake",
        "dishwasher",  # 125-129
        "screen",
        "blanket",
        "sculpture",
        "hood",
        "sconce",  # 130-134
        "vase",
        "traffic light",
        "tray",
        "ashcan",
        "fan",  # 135-139
        "pier",
        "crt screen",
        "plate",
        "monitor",
        "bulletin board",  # 140-144
        "shower",
        "radiator",
        "glass",
        "clock",
        "flag",  # 145-149
    ]


def ade_palette():
    """ADE20K palette for external use."""
    return [
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ]
