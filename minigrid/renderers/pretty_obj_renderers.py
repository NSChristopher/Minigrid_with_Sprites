import math
import numpy as np
from PIL import Image
from minigrid.core.constants import TILE_PIXELS, COLORS
from minigrid.renderers.obj_renderer import ObjRenderer
import cv2
from minigrid.utils.rendering import overlay_img

WALL_TYPES = {
    (0, 0, 0, 
     0, 1, 0, 
     0, 0, 0): 0,
    (0, 0, 0, 
     0, 1, 1, 
     0, 0, 0): 1,
    (0, 0, 0, 
     1, 1, 1, 
     0, 0, 0): 2,
    (0, 0, 0, 
     1, 1, 0, 
     0, 0, 0): 3,
    (0, 0, 0, 
     0, 1, 0, 
     0, 1, 0): 4,
    (0, 0, 0, 
     0, 1, 1, 
     0, 1, 0): 5,
    (0, 0, 0, 
     1, 1, 1, 
     0, 1, 0): 6,
    (0, 0, 0, 
     1, 1, 0, 
     0, 1, 0): 7,
    (0, 1, 0, 
     0, 1, 0, 
     0, 1, 0): 8,
    (0, 1, 0, 
     0, 1, 1, 
     0, 1, 0): 9,
    (0, 1, 0, 
     1, 1, 1, 
     0, 1, 0): 10,
    (0, 1, 0, 
     1, 1, 0, 
     0, 1, 0): 11,
    (0, 1, 0, 
     0, 1, 0, 
     0, 0, 0): 12,
    (0, 1, 0, 
     0, 1, 1, 
     0, 0, 0): 13,
    (0, 1, 0, 
     1, 1, 1, 
     0, 0, 0): 14,
    (0, 1, 0, 
     1, 1, 0, 
     0, 0, 0): 15,
}

class PrettyAgentRenderer(ObjRenderer):

    DIR_TO_ORDINAL = {
        0: 2,
        1: 0,
        2: 3,
        3: 1,
    }

    def __init__(self, obj = None):
        super().__init__(obj)
        self.agent_sprites_dir = 'figures/sprites/Players.png'
        self.agent_sprites = np.array(Image.open(self.agent_sprites_dir))

        # Sprite indexing parameters
        self.sprite_row = 4
        self.sprites_per_row = 16
        sprite_index = 0

        # Walk animation loop parameters
        self.loop = 0
        self.loop_res = 2

        # Agent position and direction
        self.agent_dir = 0

    def render(self, img: np.ndarray):
        """
        Takes an image and renders the agent on top of the passed image.
        """
        # Extract the sprite with the alpha channel
        agent_sprite = extract_sprite_by_index(self.agent_sprites_dir, 16, 16, self.sprite_index)

        # Overlay the sprite on the image
        overlay_img(img, agent_sprite)

    def update_render_state(self, action: int, agent_dir: int | None = None):
        if action == 2:
            self.loop = (self.loop + 1) % self.loop_res

        if agent_dir is not None:
            self.agent_dir = agent_dir
        self.compute_index(self.agent_dir)

    def compute_index(self, agent_dir: int):
        self.sprite_index = self.sprite_row * self.sprites_per_row + self.DIR_TO_ORDINAL[agent_dir] * self.loop_res + self.loop
        return self.sprite_index

    def rendering_encoding(self):
        return (self.sprite_index,)

class PrettyFloorRenderer(ObjRenderer):
    def __init__(self, obj = None):
        super().__init__(obj)
        self.floor_sprites_dir = 'figures/sprites/lightblue_bricks_only.png'
        self.floor_sprites = np.array(Image.open(self.floor_sprites_dir))

    def render(self, img: np.ndarray):
        """
        Takes an image and renders the object on top of the passed image.
        """
        # Extract the sprite with the alpha channel
        floor_sprite = extract_sprite_by_index(self.floor_sprites_dir, 16, 16, 17)

        # Overlay the sprite on the image
        overlay_img(img, floor_sprite)

class PrettyWallRenderer(ObjRenderer):
    
        def __init__(self, obj):
            super().__init__(obj)
            self.wall_sprites_dir = 'figures/sprites/lightblue_bricks_only.png'
            self.wall_sprites = np.array(Image.open(self.wall_sprites_dir))

        def set_render_state(self, proximity_grid):
            """
            recieves a flattened numpy array representing the wall type and sets the render state
            using the WALL_TYPES dictionary
            """
            if proximity_grid in WALL_TYPES:
                self.render_state = WALL_TYPES[proximity_grid]
            else:
                best_match = find_nearest_neighbor(proximity_grid, list(WALL_TYPES.keys()))
                self.render_state = WALL_TYPES[best_match]
    
        def render(self, img: np.ndarray):
            """
            Takes an image and renders the object on top of the passed image.
            """
            index = self.render_state

            wall_sprite = extract_sprite_by_index(self.wall_sprites_dir, 16, 16, index)

            overlay_img(img, wall_sprite)

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