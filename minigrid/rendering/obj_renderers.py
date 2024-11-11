import math
import numpy as np
from minigrid.core.constants import (
    TILE_PIXELS,
    COLOR_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
)
from minigrid.utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_line,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)

class ObjRenderer():

    def __init__(self):
        self.render_state = 0
        self.loop = 0

    def rendering_encoding(self):
            return (self.render_state, self.loop)

    def render(self, img: np.ndarray, encoding: tuple):
        """
        Takes an image and an encoding of the object to render,
        and renders the object on top of the passed image.
        """
        raise NotImplementedError

class AgentRenderer(ObjRenderer):

    def __init__(self):
        super().__init__()

    def render(self, img: np.ndarray, agent_dir: int):
        """
        Render the agent object
        """
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )

        # Rotate the agent based on its direction
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
        fill_coords(img, tri_fn, (255, 0, 0))

class GoalRenderer(ObjRenderer):
    
    def render(self, img: np.ndarray, encoding: tuple):
        """
        Render the goal object
        """
        color_str = IDX_TO_COLOR[encoding[1]]
        color = COLORS[color_str]
        fill_coords(img, point_in_rect(0, 1, 0, 1), color)

class FloorRenderer(ObjRenderer):
    
    def render(self, img: np.ndarray, encoding: tuple):
        """
        Render the floor object
        """
        color_str = IDX_TO_COLOR[encoding[1]]
        color = COLORS[color_str]
        fill_coords(img, point_in_rect(0, 1, 0, 1), color)

class LavaRenderer(ObjRenderer):
    
    def render(self, img: np.ndarray, encoding: tuple):
        """
        Render the lava object
        """
        color_str = IDX_TO_COLOR[encoding[1]]
        color = COLORS[color_str]

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), color)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))

class WallRenderer(ObjRenderer):
        
    def render(self, img: np.ndarray, encoding: tuple):
        """
        Render the wall object
        """
        color_str = IDX_TO_COLOR[encoding[1]]
        color = COLORS[color_str]
        fill_coords(img, point_in_rect(0, 1, 0, 1), color)

class DoorRenderer(ObjRenderer):
        
    def render(self, img: np.ndarray, encoding: tuple):
        """
        Render the door object
        """
        color_str = IDX_TO_COLOR[encoding[1]]
        color = COLORS[color_str]
        state = encoding[2]

        is_open = state == 0
        is_locked = state == 2

        if is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), color)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
            return

        # Door frame and door
        if is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), color)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(color))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), color)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), color)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), color)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), color)

class KeyRenderer(ObjRenderer):
        
    def render(self, img: np.ndarray, encoding: tuple):
        """
        Render the key object
        """
        color_str = IDX_TO_COLOR[encoding[1]]
        color = COLORS[color_str]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), color)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), color)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), color)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), color)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))

class BallRenderer(ObjRenderer):

    def render(self, img: np.ndarray, encoding: tuple):
        """
        Render the ball object
        """
        color_str = IDX_TO_COLOR[encoding[1]]
        color = COLORS[color_str]

        # Draw the ball
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), color)

class BoxRenderer(ObjRenderer):

    def render(self, img: np.ndarray, encoding: tuple):
        """
        Render the box object
        """
        color_str = IDX_TO_COLOR[encoding[1]]
        color = COLORS[color_str]

        # Draw the box
        fill_coords(img, point_in_rect(0.1, 0.9, 0.1, 0.9), color)

