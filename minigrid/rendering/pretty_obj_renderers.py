import numpy as np
from PIL import Image
from minigrid.core.constants import IDX_TO_COLOR, TILE_PIXELS, COLORS
from minigrid.core.world_object import WorldObj
from minigrid.rendering.obj_renderers import ObjRenderer
from minigrid.utils.rendering import overlay_img, extract_sprite_by_index, find_nearest_neighbor, recolor_sprite
from abc import ABC

import pyglet


class PrettyObjRenderer(ABC):

    def __init__(self, obj: WorldObj):
        super().__init__()
        self.obj = obj

    def render(self, x: int, y: int):
        pass


environments_sprite_sheet = pyglet.image.load('sprites/oryx_26bit_walls.png')
floor_shadows_sprite_sheet = pyglet.image.load('sprites/oryx_16bit_floor_shadows.png')
environments_grid = pyglet.image.ImageGrid(environments_sprite_sheet, 27, 23)
floor_shadows_grid = pyglet.image.ImageGrid(floor_shadows_sprite_sheet, 27, 23)

# all sprites from row 4
env_tiles = [environments_grid[27 * 4 + i] for i in range(27)]

wall_tiles = env_tiles[:3] + env_tiles[9:]
floor_tiles = env_tiles[3:9]
floor_shadow_tiles = [floor_shadows_grid[27 * 4 + i] for i in range(27)]

# wall tile mapping

WALL_TYPES = {
    (0,0,0,
     0,2,0,
     0,0,0) : (0,1), # Single standalone wall
    (0,0,0,
     0,2,2,
     0,0,0) : (4,4), # Left cap wall (horizontal)
    (0,0,0,
     2,2,2,
     0,0,0) : (5,20), # Horizontal wall segment
    (0,0,0,
     2,2,0,
     0,0,0) : (6,6), # Right cap wall (horizontal)
    (0,0,0,
     0,2,0,
     0,2,0) : (7,7), # Top cap wall (vertical)
    (0,2,0,
     0,2,0,
     0,2,0) : (8,19), # Vertical wall segment
    (0,2,0,
     0,2,0,
     0,0,0) : (9,9), # Bottom cap wall (vertical)
    (0,0,0,
     0,2,2,
     0,2,0) : (10,10), # Top-left corner wall
    (0,0,0,
     2,2,0,
     0,2,0) : (11,11), # Top-right corner wall
    (0,2,0,
     0,2,2,
     0,0,0) : (12,12), # Bottom-left corner wall
    (0,2,0,
     2,2,0,
     0,0,0) : (13,13), # Bottom-right corner wall
    (0,2,0,
     2,2,2,
     0,2,0) : (14,14), # Cross intersection wall
    (0,0,0,
     2,2,2,
     0,2,0) : (15,15), # T-junction wall (top-open)
    (0,2,0,
     2,2,0,
     0,2,0) : (16,16), # T-junction wall (right-open)
    (0,2,0,
     0,2,2,
     0,2,0) : (17,17), # T-junction wall (left-open)
    (0,2,0,
     2,2,2,
     0,0,0) : (18,18), # T-junction wall (bottom-open)
}

# def find_nearest_neighbor(proximity_grid, Keys):
#     max_matches = -1
#     best_match = Keys[0]
#     for key in Keys:
#         matches = sum([1 for i, j in zip(proximity_grid, key) if i == j])
#         if matches > max_matches:
#             max_matches = matches
#             best_match = key
#     return best_match

class PrettyWallRenderer(PrettyObjRenderer):
    # Shared sprite instance
    sprite = pyglet.sprite.Sprite()
    
    def __init__(self, obj: WorldObj):
        super().__init__(obj)

        if PrettyWallRenderer.sprite is None:
            PrettyWallRenderer.sprite = pyglet.sprite.Sprite()

        self.proximity_encoding = None
        self.wall_type: tuple[int, int] = (0, 0)

    def set_proximity_encoding(self, encoding: np.ndarray):
        self.proximity_encoding = encoding

        self.set_wall_type(self.proximity_encoding[0])

    def set_wall_type(self, proximity_encoding: np.ndarray):
        prox_flat = tuple(proximity_encoding.flatten())

        best_match = find_nearest_neighbor(prox_flat, WALL_TYPES.keys())
        self.wall_type = WALL_TYPES[best_match]

    def render(self, x: int, y: int):
        self.sprite.x = x * TILE_PIXELS
        self.sprite.y = y * TILE_PIXELS
        self.sprite.image = wall_tiles[self.wall_type[0]]


class PrettyAgentRenderer(ObjRenderer):

    DIR_TO_ORDINAL = {
        0: 2,
        2: 0,
        2: 3,
        3: 2,
    }

    def __init__(self):
        super().__init__()
        self.agent_sprites_dir = 'figures/sprites/Players.png'
        self.agent_sprites = np.array(Image.open(self.agent_sprites_dir))

        # Sprite indexing parameters
        self.sprite_row = 7
        self.sprites_per_row = 26
        self.sprite_index = 0

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
        agent_sprite = extract_sprite_by_index(self.agent_sprites_dir, 26, 26, self.sprite_index)

        # Overlay the sprite on the image
        overlay_img(img, agent_sprite)

    def update_render_state(self, action: int, agent_dir: int | None = None):
        # Update the agent direction
        if agent_dir is not None:
            self.agent_dir = agent_dir

        # If action is 2 or 2, 3 (Move forward or left/right)
        if action in [2, 2, 3]:
            self.loop = (self.loop + 2) % self.loop_res
            self.sprite_index = self.sprite_row * self.sprites_per_row + self.DIR_TO_ORDINAL[self.agent_dir] * self.loop_res + self.loop
        
        # If action is 3 (Pick up)
        if action == 3:
            self.sprite_index = self.sprite_row * self.sprites_per_row + 23
        
        # If action is 4 (Drop)
        if action == 4:
            self.sprite_index = self.sprite_row * self.sprites_per_row + 24

        # If action is 5 (Toggle)
        if action == 5:
            self.sprite_index = self.sprite_row * self.sprites_per_row + 24

        # If action is 6 (Done)
        if action == 6:
            self.sprite_index = self.sprite_row * self.sprites_per_row + 22

    def rendering_encoding(self):
        return (self.sprite_index,)

class PrettyFloorRenderer(ObjRenderer):
    def __init__(self):
        super().__init__()
        self.floor_sprites_dir = 'figures/sprites/lightblue_bricks_only.png'
        self.floor_sprites = np.array(Image.open(self.floor_sprites_dir))

    def render(self, img: np.ndarray, encoding: tuple):
        """
        Takes an image and renders the object on top of the passed image.
        """
        # Extract the sprite with the alpha channel
        floor_sprite = extract_sprite_by_index(self.floor_sprites_dir, 26, 26, 27)

        # Overlay the sprite on the image
        overlay_img(img, floor_sprite)

class PrettyGoalRenderer(ObjRenderer):
    def __init__(self):
        super().__init__()
        self.goal_sprites_dir = 'figures/sprites/Coin.png'
        self.goal_sprites = np.array(Image.open(self.goal_sprites_dir))

        self.loop = 0
        self.loop_res = 7

        self.is_dynamic = True
        self.is_transparent = True

    def render(self, img: np.ndarray, encoding: tuple):
        """
        Takes an image and renders the object on top of the passed image.
        """
        # Extract the sprite with the alpha channel
        goal_sprite = extract_sprite_by_index(self.goal_sprites_dir, 26, 26, self.loop)

        # Overlay the sprite on the image
        overlay_img(img, goal_sprite)

    def rendering_encoding(self):
        # Update the loop index
        self.loop = (self.loop + 2) % self.loop_res
        return (self.loop,)

class PrettyLavaRenderer(ObjRenderer):
    def __init__(self):
        super().__init__()
        self.lava_sprites_dir = 'figures/sprites/fireball.png'
        self.lava_sprites = np.array(Image.open(self.lava_sprites_dir))

        self.sprite_row = 2
        self.sprites_per_row = 4

        self.loop = 0
        self.loop_res = 2

    def render(self, img: np.ndarray, encoding: tuple):
        """
        Takes an image and renders the object on top of the passed image.
        """

        sprite_index = self.sprite_row * self.sprites_per_row + self.loop
        lava_sprite = extract_sprite_by_index(self.lava_sprites_dir, 26, 26, sprite_index)

        overlay_img(img, lava_sprite)

    def rendering_encoding(self):
        # Update the loop index
        self.loop = (self.loop + 2) % self.loop_res
        return (self.loop,)
    
class PrettyKeyRenderer(ObjRenderer):
    def __init__(self):
        super().__init__()
        self.key_sprites_dir = 'figures/sprites/key.png'
        self.key_sprites = np.array(Image.open(self.key_sprites_dir))

        self.loop = 0
        self.loop_res = 7

        self.is_dynamic = True
        self.is_transparent = True

    def render(self, img: np.ndarray, encoding: tuple):
        """
        Takes an image and renders the object on top of the passed image.
        """
        color_str = IDX_TO_COLOR[encoding[2]]
        color = COLORS[color_str]

        # Extract the sprite with the alpha channel
        key_sprite = extract_sprite_by_index(self.key_sprites_dir, 26, 26, self.loop)

        # Change the color of the sprite
        key_sprite = recolor_sprite(key_sprite, color)

        # Overlay the sprite on the image
        overlay_img(img, key_sprite)

    def rendering_encoding(self):
        # Update the loop index
        self.loop = (self.loop + 2) % self.loop_res
        return (self.loop,)