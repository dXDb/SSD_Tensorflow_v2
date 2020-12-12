import os
import math
import random
import numpy              as np
import tensorflow         as tf
import matplotlib.pyplot  as plt
import matplotlib.patches as patches

from PIL       import Image
from box_utils import compute_iou


class ImageVisualizer(object):
    """ Class for visualizing image

    Attributes:
        idx_to_name: list to convert integer to string label
        class_colors: colors for drawing boxes and labels
        save_dir: directory to store images
    """

    def __init__(self, idx_to_name, class_colors=None, save_dir=None):
        self.idx_to_name = idx_to_name
        if class_colors is None or len(class_colors) != len(self.idx_to_name):
            self.class_colors = [[0, 255, 0]] * len(self.idx_to_name)
        else:
            self.class_colors = class_colors

        if save_dir is None:
            self.save_dir = './'
        else:
            self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

    def save_image(self, img, boxes, labels, name):
        """ Method to draw boxes and labels
            then save to dir

        Args:
            img: numpy array (width, height, 3)
            boxes: numpy array (num_boxes, 4)
            labels: numpy array (num_boxes)
            name: name of image to be saved
        """
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        save_path = os.path.join(self.save_dir, name)

        for i, box in enumerate(boxes):
            idx = labels[i] - 1
            cls_name = self.idx_to_name[idx]
            top_left = (box[0], box[1])
            bot_right = (box[2], box[3])
            ax.add_patch(patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor=(0., 1., 0.),
                facecolor="none"))
            plt.text(
                box[0],
                box[1],
                s=cls_name,
                color="white",
                verticalalignment="top",
                bbox={"color": (0., 1., 0.), "pad": 0},
            )

        plt.axis("off")
        # plt.gca().xaxis.set_major_locator(NullLocator())
        # plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
        plt.close('all')


## WJ ##
def random_patching(img, boxes, labels):
    max_trials = 20
    threshold = np.random.choice([0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    breaking = False

    for i in range(max_trials):
        scale = np.random.uniform(0.3, 1)
        min_aspect_ratio = 0.5
        max_aspect_ratio = 2.0

        aspect_ratio = np.random.uniform(min_aspect_ratio, max_aspect_ratio)

        aspect_ratio = max(aspect_ratio, scale**2)
        aspect_ratio = min(aspect_ratio, 1/(scale**2))

        width = scale * math.sqrt(aspect_ratio)
        height = scale / math.sqrt(aspect_ratio)

        patch_xmin = np.random.uniform(0, 1 - width)
        patch_ymin = np.random.uniform(0, 1 - height)
        patch_xmax = patch_xmin + width
        patch_ymax = patch_ymin + height
        
        patch = np.array([[patch_xmin, patch_ymin, patch_xmax, patch_ymax]], dtype=np.float32)
        patch = np.clip(patch, 0.0, 1.0)        

        ious = compute_iou(tf.constant(patch), boxes)
        ious = ious.numpy()
        iou = np.max(ious)

        if iou >= threshold:
            breaking = True
            break

    if not breaking: return img, boxes, labels

    img_width, img_height = img.width, img.height

    after_img_width_xmin = int(img_width * patch_xmin)
    after_img_width_xmax = int(img_width * patch_xmax)
    after_img_width_ymin = int(img_height * patch_ymin)
    after_img_width_ymax = int(img_height * patch_ymax)
    
    after_boxes, after_labels = box_changed(boxes, labels, img_width, img_height, after_img_width_xmin, after_img_width_xmax, after_img_width_ymin, after_img_width_ymax)
    if len(after_boxes) == 0: return img, boxes, labels

    img = tf.keras.preprocessing.image.img_to_array(img)[after_img_width_ymin:after_img_width_ymax, after_img_width_xmin:after_img_width_xmax, :].copy()
    img = tf.keras.preprocessing.image.array_to_img(img)

    return img, tf.constant(after_boxes), tf.constant(after_labels)


def box_changed(boxes, labels, img_width, img_height, after_img_width_xmin, after_img_width_xmax, after_img_width_ymin, after_img_width_ymax):
    after_boxes = []
    after_labels = []
    index = 0

    for box in boxes.numpy():
        # [xmin, ymin, xmax, ymax]
        xmin = img_width * box[0] / (after_img_width_xmax - after_img_width_xmin)
        xmax = img_width * box[2] / (after_img_width_xmax - after_img_width_xmin)
        ymin = img_height * box[1] / (after_img_width_ymax - after_img_width_ymin)
        ymax = img_height * box[3] / (after_img_width_ymax - after_img_width_ymin)
        
        if xmin > 1 or ymin > 1 or xmax < 0 or ymax < 0: continue
        after_box = [xmin, ymin, xmax, ymax]
        after_box = np.clip(after_box, 0.0, 1.0)

        after_boxes.append(after_box.tolist())
        after_labels.append(labels.numpy()[index])
        index += 1

    return after_boxes, after_labels


def random_brightness(img, boxes, labels, delta=0.32):
    p = random.uniform(0, 1)
    if p < 0.5: return img, boxes, labels

    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.image.random_brightness(img, max_delta=delta)
    img = tf.keras.preprocessing.image.array_to_img(img)
    return img, boxes, labels


def random_contrast(img, boxes, labels, lower=0.5, upper=1.5):
    p = random.uniform(0, 1)
    if p < 0.5: return img, boxes, labels

    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.image.random_contrast(img, lower=lower, upper=upper)
    img = tf.keras.preprocessing.image.array_to_img(img)
    return img, boxes, labels


def random_hue(img, boxes, labels, delta=0.18):
    p = random.uniform(0, 1)
    if p < 0.5: return img, boxes, labels

    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.image.random_hue(img, max_delta=delta)
    img = tf.keras.preprocessing.image.array_to_img(img)
    return img, boxes, labels


def random_saturation(img, boxes, labels, lower=0.5, upper=1.5):
    p = random.uniform(0, 1)
    if p < 0.5: return img, boxes, labels

    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.image.random_saturation(img, lower=lower, upper=upper)
    img = tf.keras.preprocessing.image.array_to_img(img)
    return img, boxes, labels


def horizontal_flip(img, boxes, labels):
    p = random.uniform(0, 1)
    if p < 0.5: return img, boxes, labels

    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    boxes = tf.stack([
        1 - boxes[:, 2],
        boxes[:, 1],
        1 - boxes[:, 0],
        boxes[:, 3]], axis=1)

    return img, boxes, labels
## WJ ##