from __future__ import annotations

import math
import cv2
import numpy as np
from PIL import Image, ImageEnhance


def downsample(img, factor):
    """
    Downsample an image along both dimensions by some factor
    """

    assert img.shape[0] % factor == 0
    assert img.shape[1] % factor == 0

    img = img.reshape(
        [img.shape[0] // factor, factor, img.shape[1] // factor, factor, 3]
    )
    img = img.mean(axis=3)
    img = img.mean(axis=1)

    return img


def fill_coords(img, fn, color):
    """
    Fill pixels of an image with coordinates matching a filter function
    """

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]
            xf = (x + 0.5) / img.shape[1]
            if fn(xf, yf):
                img[y, x] = color

    return img

def rotate_fn(fin, cx, cy, theta):
    def fout(x, y):
        x = x - cx
        y = y - cy

        x2 = cx + x * math.cos(-theta) - y * math.sin(-theta)
        y2 = cy + y * math.cos(-theta) + x * math.sin(-theta)

        return fin(x2, y2)

    return fout


def point_in_line(x0, y0, x1, y1, r):
    p0 = np.array([x0, y0], dtype=np.float32)
    p1 = np.array([x1, y1], dtype=np.float32)
    dir = p1 - p0
    dist = np.linalg.norm(dir)
    dir = dir / dist

    xmin = min(x0, x1) - r
    xmax = max(x0, x1) + r
    ymin = min(y0, y1) - r
    ymax = max(y0, y1) + r

    def fn(x, y):
        # Fast, early escape test
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return False

        q = np.array([x, y])
        pq = q - p0

        # Closest point on line
        a = np.dot(pq, dir)
        a = np.clip(a, 0, dist)
        p = p0 + a * dir

        dist_to_line = np.linalg.norm(q - p)
        return dist_to_line <= r

    return fn


def point_in_circle(cx, cy, r):
    def fn(x, y):
        return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r

    return fn


def point_in_rect(xmin, xmax, ymin, ymax):
    def fn(x, y):
        return x >= xmin and x <= xmax and y >= ymin and y <= ymax

    return fn


def point_in_triangle(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    def fn(x, y):
        v0 = c - a
        v1 = b - a
        v2 = np.array((x, y)) - a

        # Compute dot products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Check if point is in triangle
        return (u >= 0) and (v >= 0) and (u + v) < 1

    return fn


def highlight_img(img, color=(255, 255, 255), alpha=0.30):
    """
    Add highlighting to an image
    """

    blend_img = img + alpha * (np.array(color, dtype=np.uint8) - img)
    blend_img = blend_img.clip(0, 255).astype(np.uint8)
    img[:, :, :] = blend_img

def overlay_img(img, overlay):
    """
    Overlay one image on top of another with proper alpha blending.
    """
    # Resize the overlay to match the input image dimensions
    overlay = cv2.resize(overlay, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Separate the RGB and alpha channels
    overlay_rgb = overlay[:, :, :3]
    if overlay.shape[2] == 4:
        overlay_alpha = overlay[:, :, 3] / 255.0  # Normalize alpha channel to range [0, 1]
    else:
        overlay_alpha = np.ones((overlay.shape[0], overlay.shape[1]), dtype=np.float32)

    # Ensure alpha is in range [0, 1] and has correct dimensions
    overlay_alpha = np.expand_dims(overlay_alpha, axis=-1)
    
    # Premultiply alpha in the overlay and blend it with the img
    img[:] = (overlay_rgb * overlay_alpha + img * (1 - overlay_alpha)).astype(np.uint8)

    return img

def extract_sprite_by_index(spritesheet_path, sprite_width, sprite_height, index):
    """
    Extract the sprite from the sprite sheet by index (row-major order).
    This function assumes the sprite sheet is filled with equally sized sprites.
    """

    # Open the sprite sheet
    spritesheet = Image.open(spritesheet_path)
    spritesheet_array = np.array(spritesheet)

    # Calculate how many sprites fit per row in the sprite sheet
    sprites_per_row = spritesheet_array.shape[1] // sprite_width

    # Calculate the row and column based on the index
    row = index // sprites_per_row
    col = index % sprites_per_row

    # Calculate the coordinates of the sprite in the sheet
    top = row * sprite_height
    left = col * sprite_width
    bottom = top + sprite_height
    right = left + sprite_width

    # Extract and return the sprite
    return spritesheet_array[top:bottom, left:right]

def find_nearest_neighbor(proximity_grid, Keys):
    max_matches = -1
    best_match = Keys[0]
    for key in Keys:
        matches = sum([1 for i, j in zip(proximity_grid, key) if i == j])
        if matches > max_matches:
            max_matches = matches
            best_match = key
    return best_match

def recolor_sprite(sprite: np.ndarray, base_color: tuple):
    """
    Converts the sprite to grayscale and recolors it based on the given color.
    """
    # Convert to grayscale
    grayscale = np.mean(sprite[..., :3], axis=2, keepdims=True).astype(np.uint8)
    
    # Normalize and recolor in one step
    recolored_rgb = (grayscale / 255.0 * np.array(base_color)).astype(np.uint8)

    # Combine recolored RGB with original alpha
    recolored_sprite = np.dstack((recolored_rgb, sprite[..., 3]))

    return recolored_sprite


