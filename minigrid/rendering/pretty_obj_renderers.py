import numpy as np
from PIL import Image
from minigrid.core.constants import IDX_TO_COLOR, TILE_PIXELS, COLORS
from minigrid.rendering.obj_renderers import ObjRenderer
from minigrid.utils.rendering import overlay_img, extract_sprite_by_index, find_nearest_neighbor, recolor_sprite

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

    def __init__(self):
        super().__init__()
        self.agent_sprites_dir = 'figures/sprites/Players.png'
        self.agent_sprites = np.array(Image.open(self.agent_sprites_dir))

        # Sprite indexing parameters
        self.sprite_row = 7
        self.sprites_per_row = 16
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
        agent_sprite = extract_sprite_by_index(self.agent_sprites_dir, 16, 16, self.sprite_index)

        # Overlay the sprite on the image
        overlay_img(img, agent_sprite)

    def update_render_state(self, action: int, agent_dir: int | None = None):
        # Update the agent direction
        if agent_dir is not None:
            self.agent_dir = agent_dir

        # If action is 2 or 1, 3 (Move forward or left/right)
        if action in [1, 2, 3]:
            self.loop = (self.loop + 1) % self.loop_res
            self.sprite_index = self.sprite_row * self.sprites_per_row + self.DIR_TO_ORDINAL[self.agent_dir] * self.loop_res + self.loop
        
        # If action is 3 (Pick up)
        if action == 3:
            self.sprite_index = self.sprite_row * self.sprites_per_row + 13
        
        # If action is 4 (Drop)
        if action == 4:
            self.sprite_index = self.sprite_row * self.sprites_per_row + 14

        # If action is 5 (Toggle)
        if action == 5:
            self.sprite_index = self.sprite_row * self.sprites_per_row + 14

        # If action is 6 (Done)
        if action == 6:
            self.sprite_index = self.sprite_row * self.sprites_per_row + 11

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
        floor_sprite = extract_sprite_by_index(self.floor_sprites_dir, 16, 16, 17)

        # Overlay the sprite on the image
        overlay_img(img, floor_sprite)

class PrettyWallRenderer(ObjRenderer):
    
        def __init__(self):
            super().__init__()
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
    
        def render(self, img: np.ndarray, encoding: tuple):
            """
            Takes an image and renders the object on top of the passed image.
            """
            index = self.render_state

            wall_sprite = extract_sprite_by_index(self.wall_sprites_dir, 16, 16, index)

            overlay_img(img, wall_sprite)

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
        goal_sprite = extract_sprite_by_index(self.goal_sprites_dir, 16, 16, self.loop)

        # Overlay the sprite on the image
        overlay_img(img, goal_sprite)

    def rendering_encoding(self):
        # Update the loop index
        self.loop = (self.loop + 1) % self.loop_res
        return (self.loop,)

class PrettyLavaRenderer(ObjRenderer):
    def __init__(self):
        super().__init__()
        self.lava_sprites_dir = 'figures/sprites/fireball.png'
        self.lava_sprites = np.array(Image.open(self.lava_sprites_dir))

        self.sprite_row = 1
        self.sprites_per_row = 4

        self.loop = 0
        self.loop_res = 2

    def render(self, img: np.ndarray, encoding: tuple):
        """
        Takes an image and renders the object on top of the passed image.
        """

        sprite_index = self.sprite_row * self.sprites_per_row + self.loop
        lava_sprite = extract_sprite_by_index(self.lava_sprites_dir, 16, 16, sprite_index)

        overlay_img(img, lava_sprite)

    def rendering_encoding(self):
        # Update the loop index
        self.loop = (self.loop + 1) % self.loop_res
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
        color_str = IDX_TO_COLOR[encoding[1]]
        color = COLORS[color_str]

        # Extract the sprite with the alpha channel
        key_sprite = extract_sprite_by_index(self.key_sprites_dir, 16, 16, self.loop)

        # Change the color of the sprite
        key_sprite = recolor_sprite(key_sprite, color)

        # Overlay the sprite on the image
        overlay_img(img, key_sprite)

    def rendering_encoding(self):
        # Update the loop index
        self.loop = (self.loop + 1) % self.loop_res
        return (self.loop,)