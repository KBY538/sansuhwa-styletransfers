sansu2coco = {
                8: 
                    {
                        "feature": "풀",
                        "matching_coco": [64, 97, 119, 124, 134, 142, 163],
                        "color": (128, 128, 240)
                    },
                6:
                    {
                        "feature": "땅",
                        "matching_coco": [111, 114, 115, 116, 117, 118, 126, 131, 136, 140, 145, 149, 152, 154, 159],
                        "color": (0, 165, 255)
                    },
                5:
                    {
                        "feature": "하늘",
                        "matching_coco": [0, 106, 120, 130, 157],
                        "color": (250, 206, 135)
                    },
                2:
                    {
                        "feature": "앞산",
                        "matching_coco": [127, 135],
                        "color": (144, 238, 144)
                    },
                4:
                    {
                        "feature": "가까운 나무",
                        "matching_coco": [94, 122, 129, 169],
                        "color": (19, 69, 139)
                    },
                3:
                    {
                        "feature": "먼 나무",
                        "matching_coco": [182],
                        "color": (143, 143, 188)
                    },
                9:
                    {
                        "feature": "물",
                        "matching_coco": [148, 155, 178, 179],
                        "color": (225, 105, 65)
                    },
                7:
                    {
                        "feature": "바위",
                        "matching_coco": [125, 144, 147, 150, 162],
                        "color": (153, 136, 119)
                    },
                10:
                    {
                        "feature": "건물",
                        "matching_coco": [92, 95, 96, 112, 113, 128, 146, 151, 158, 161, 164, 166, 171, 172, 173, 174, 175, 176, 177, 180, 181],
                        "color": (0, 0, 0)
                    }
            }

def for_sansu_label(coco_label):
    if coco_label == 0:
        return 0

    for class_num, class_elements in sansu2coco.items():
        if coco_label in class_elements["matching_coco"]:
            return class_num
    
    return 11

def sansu_label_color(class_num):
    if class_num in sansu2coco:
        
        return sansu2coco[class_num]["color"]
    else:
        return (128, 128, 128)

coco_labels = {
    0: "unlabeled",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    12: "street sign",		# Removed from COCO.
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    26: "hat",		# Removed from COCO.
    27: "backpack",
    28: "umbrella",
    29: "shoe",		# Removed from COCO.
    30: "eye glasses",		# Removed from COCO.
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    45: "plate",		# Removed from COCO.
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    66: "mirror",		# Removed from COCO.
    67: "dining table",
    68: "window",		# Removed from COCO.
    69: "desk",		# Removed from COCO.
    70: "toilet",
    71: "door",		# Removed from COCO.
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    83: "blender",		# Removed from COCO.
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
    91: "hair brush",		# Removed from COCO.
    92: "banner",	# Any large sign, especially if constructed of soft material or fabric, often seen in stadiums and advertising.
    93: "blanket", #	A loosely woven fabric, used for warmth while sleeping.
    94: "branch", #	The woody part of a tree or bush, arising from the trunk and usually dividing.
    95: "bridge", #	A manmade construction that spans a divide (incl. train bridge, river bridge).
    96: "building-other", #	Any other type of building or structures.
    97: "bush", #	A woody plant distinguished from a tree by its multiple stems and lower height (incl. hedge, scrub).
    98: "cabinet", #	A storage closet, often hanging on the wall.
    99: "cage", #	An enclosure made of bars, often seen in zoos.
    100: "cardboard", #	A wood-based material resembling heavy paper, used in the manufacture of boxes, cartons and signs.
    101: "carpet", #	A fabric used as a floor covering.
    102: "ceiling-other", #	Other types of ceilings (incl. industrial ceilings, painted ceilings).
    103: "ceiling-tile", #	A ceiling made of regularly-shaped slabs.
    104: "cloth", #	A piece of cloth used for a particular purpose. (incl. cleaning cloth).
    105: "clothes", #	Items of clothing or apparel, not currently worn by a person.
    106: "clouds", #	A visible mass of water droplets suspended in the air.
    107: "counter", #	A surface in the kitchen or bathroom, often built into a wall or above a cabinet, which holds the washbasin or surface to prepare food.
    108: "cupboard", #	A piece of furniture used for storing dishware or a wardrobe for clothes, sometimes hanging on the wall.
    109: "curtain", #	A piece of cloth covering a window, bed or shower to offer privacy and keep out light.
    110: "desk-stuff", #	A piece of furniture with a flat surface and typically with drawers, at which one can read, write, or do other work.
    111: "dirt", #	Soil or earth (incl. dirt path).
    112: "door-stuff", #	A portal of entry into a building, room or vehicle, consisting of a rigid plane movable on a hinge (incl. the frame, replaces door).
    113: "fence", #	A thin, human-constructed barrier which separates two pieces of land.
    114: "floor-marble", #	The supporting surface of a room or outside, made of marble.
    115: "floor-other", #	Any other type of floor (incl. rubber-based floor).
    116: "floor-stone", #	The supporting surface of a room or outside, made of stone (incl. brick floor).
    117: "floor-tile", #	The supporting surface of a room or outside, made of regularly-shaped slabs (incl. tiled stone floor, tiled marble floor).
    118: "floor-wood", #	The supporting surface of a room or outside, made of wood (incl. wooden tiles, parquet, laminate, wooden boards).
    119: "flower", #	The seed-bearing part of a plant (incl. the entire flower).
    120: "fog", #	A thick cloud of tiny water droplets suspended in the atmosphere near the earth's surface.
    121: "food-other", #	Any other type of food.
    122: "fruit", #	The sweet and fleshy product of a tree or other plant.
    123: "furniture-other", #	Any other type of furniture (incl. oven).
    124: "grass", #	Vegetation consisting of typically short plants with long, narrow leaves (incl. lawn, pasture).
    125: "gravel", #	A loose aggregation of small water-worn or pounded stones.
    126: "ground-other", #	Any other type of ground found outside a building.
    127: "hill", #	A naturally raised area of land, not as high as a mountain, viewed at a distance and may be covered in trees, snow or grass.
    128: "house", #	A smaller size building for human habitation.
    129: "leaves", #	A structure of a higher plant, typically green and blade-like, that is attached to a stem or stalk.
    130: "light", #	A source of illumination, especially a lamp (incl. ceiling lights).
    131: "mat", #	A piece of coarse material placed on a floor for people to wipe their feet on.
    132: "metal", #	A raw metal material (incl. a pile of metal).
    133: "mirror-stuff", #	A glass coated surface which reflects a clear image (incl. the frame, replaces mirror).
    134: "moss", #	A small flowerless green plant which lacks true roots, growing in in damp habitats.
    135: "mountain", #	A large natural elevation rising abruptly from the surrounding level, viewed at a distance and may be covered in trees, snow or grass.
    136: "mud", #	A soft, sticky matter resulting from the mixing of earth and water.
    137: "napkin", #	A piece of cloth or paper used at a meal to wipe the fingers or lips.
    138: "net", #	An open-meshed fabric twisted, knotted, or woven together at regular intervals.
    139: "paper", #	A material manufactured in thin sheets from the pulp of wood.
    140: "pavement", #	A typically raised paved path for pedestrians at the side of a road.
    141: "pillow", #	A rectangular cloth bag stuffed with soft materials to support the head.
    142: "plant-other", #	Any other type of plant.
    143: "plastic", #	Raw plastic material.
    144: "platform", #	A raised level surface on which people or things can stand (incl. railroad platform).
    145: "playingfield", #	A ground marked off for various games (incl. indoor and outdoor).
    146: "railing", #	A fence or barrier made of typically metal rails.
    147: "railroad", #	A track made of steel rails along which trains run (incl. the wooden beams).
    148: "river", #	A stream of flowing water.
    149: "road", #	A paved way leading from one place to another.
    150: "rock", #	The solid mineral material forming part of the surface of the earth.
    151: "roof", #	The structure forming the upper covering of a building.
    152: "rug", #	A floor covering of thick woven material, typically not extending over the entire floor.
    153: "salad", #	A cold dish of various mixtures of raw or cooked vegetables.
    154: "sand", #	A loose granular substance, typically pale yellowish brown, resulting from erosion (incl. beach).
    155: "sea", #	Expanse of water that covers most of the earth's surface.
    156: "shelf", #	An open piece of furniture that provides a surface for the storage or display of objects.
    157: "sky-other", #	Any other type of sky (incl. blue sky).
    158: "skyscraper", #	A very tall building of many storeys.
    159: "snow", #	Atmospheric water vapour frozen into ice crystals, falling or lying on the ground.
    160: "solid-other", #	Any other type of solid material.
    161: "stairs", #	A set of steps leading from one floor to another (incl. stairs inside or outside a building).
    162: "stone", #	A piece of stone shaped for a purpose.
    163: "straw", #	Dried stalks of grain.
    164: "structural-other", #	Any other type of structural connection (incl. arcs, pillars).
    165: "table", #	A piece of furniture with a flat top and one or more legs.
    166: "tent", #	A portable shelter made of cloth.
    167: "textile-other", #	Any other type of textile.
    168: "towel", #	A piece of thick absorbent cloth used for drying oneself.
    169: "tree", #	A woody plant, typically having a single trunk growing to a considerable height and bearing lateral branches at some distance from the ground.
    170: "vegetable", #	A part of a plant used as food.
    171: "wall-brick", #	A building wall made of bricks of clay.
    172: "wall-concrete", #	A building wall made of concrete.
    173: "wall-other", #	Any other type of wall.
    174: "wall-panel", #	A panel that is attached to a wall.
    175: "wall-stone", #	A building wall made of stone.
    176: "wall-tile", #	A building wall made of tiles, such as used in bathrooms and kitchens.
    177: "wall-wood", #	A building wall made of wooden material.
    178: "water-other", #	Any other type of water (incl. lake).
    179: "waterdrops", #	Sprinkles or drops of water not connected to a larger body of water.
    180: "window-blind", #	Blinds and shutters that cover a window.
    181: "window-other", #	Any type of window that must be visible in the image (replaces window).
    182: "wood"
    }