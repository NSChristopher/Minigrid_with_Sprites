import numpy as np
from PIL import Image
from minigrid.core.constants import IDX_TO_COLOR, OBJECT_TO_IDX, TILE_PIXELS, COLORS
from minigrid.core.world_object import WorldObj
from minigrid.rendering.obj_renderers import ObjRenderer
from minigrid.utils.rendering import overlay_img, extract_sprite_by_index, find_nearest_neighbor, recolor_sprite
from abc import ABC
from minigrid.core.actions import Actions

import pyglet
from pyglet.image import ImageGrid
from pyglet.image.animation import Animation, AnimationFrame


class PrettyObjRenderer(ABC):

    def __init__(self, obj: WorldObj | None):
        super().__init__()
        self.obj = obj

    def update(self, x: int, y: int):
        pass


environments_sprite_sheet = pyglet.image.load('sprites/oryx_16bit_walls.png')
floor_shadows_sprite_sheet = pyglet.image.load('sprites/oryx_16bit_floor_shadows.png')
environments_grid = ImageGrid(environments_sprite_sheet, rows=23, columns=27, row_padding=0, column_padding=0)
floor_shadows_grid = ImageGrid(floor_shadows_sprite_sheet, rows=1, columns=8, row_padding=0, column_padding=0)

# all sprites from row 19 starting from bottom left corner
access_row = 15
env_tiles = [environments_grid[(27 * access_row) + i] for i in range(27)]

wall_tiles = env_tiles[:3] + env_tiles[9:]
floor_tiles = env_tiles[3:9]
floor_shadow_tiles = [floor_shadows_grid[i] for i in range(8)]

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

    R_TILE_PIXELS = 24
    
    def __init__(self, obj: WorldObj, x: int, y: int, batch: pyglet.graphics.Batch, group: pyglet.graphics.Group, scale_factor: int = 1):
        super().__init__(obj)
        self.obj = obj
        self.sprite = pyglet.sprite.Sprite(img=wall_tiles[0], x=x, y=y, batch=batch, group=group)
        self.sprite.update(scale=scale_factor)

        # Proximity encoding of surrounding objects
        self.proximity_encoding = None

        # Wall type 
        self.type: tuple[int, int] = (0, 0)
        self.image = None

    def set_proximity_encoding(self, encoding: np.ndarray):
        """
        Set the proximity encoding of the wall object, set the wall type 
        and set the sprite location.
        """
        self.proximity_encoding = encoding

        self.set_wall_type(self.proximity_encoding)

    def set_wall_type(self, proximity_encoding: np.ndarray):
        """
        Set the wall type based on the proximity encoding.
        """
        mapped_proximity = np.where(proximity_encoding == 2, 2, 0)
        prox_flat = tuple(int(val) for val in mapped_proximity.flatten())

        self.prox_flat = prox_flat

        if prox_flat in WALL_TYPES:
            self.type = WALL_TYPES[prox_flat]
        else:
            best_match = find_nearest_neighbor(prox_flat, list(WALL_TYPES.keys()))
            self.type = WALL_TYPES[best_match]

        random = np.random.rand(1)
        if random > 0.05:
            self.image = wall_tiles[self.type[0]]
        else:
            self.image = wall_tiles[self.type[1]]

        self.sprite.image = self.image

    def update(self, x: int, y: int):
        """
        Update state.
        """
        # update the sprite location
        self.sprite.update(x=x, y=y)

class PrettyFloorRenderer(PrettyObjRenderer):
    
        R_TILE_PIXELS = 24
    
        def __init__(self, obj: WorldObj | None, x: int, y: int, batch: pyglet.graphics.Batch, group: pyglet.graphics.Group, scale_factor: int = 1):
            super().__init__(obj)
            self.obj = obj
            self.sprite = pyglet.sprite.Sprite(img=floor_tiles[1], x=x, y=y, batch=batch, group=group)
            self.sprite.update(scale=scale_factor)
            self.shadow_sprite = pyglet.sprite.Sprite(img=floor_shadow_tiles[1], x=x, y=y, batch=batch, group=group)
            self.shadow_sprite.update(scale=scale_factor)

            self.image = None

        def set_proximity_encoding(self, encoding: np.ndarray):
            """
            Set the proximity encoding of the floor object.
            """
            self.proximity_encoding = encoding

            self.set_floor_type(self.proximity_encoding)

        def set_floor_type(self, proximity_encoding: np.ndarray):
            """
            Set the floor type based on the proximity encoding.
            """
            is_wall_above = proximity_encoding[0, 1] == OBJECT_TO_IDX['wall']

            if is_wall_above == False:
                self.shadow_sprite.delete()

        def update(self, x: int, y: int):
            """
            Update state.
            """
            pass

door_sprite_sheet = pyglet.image.load('sprites/oryx_16bit_doors.png')
door_grid = ImageGrid(door_sprite_sheet, rows=1, columns=14)

unlocked_door = door_grid[0]
opened_door = door_grid[1]
locked_door = door_grid[2]

class PrettyDoorRenderer(PrettyObjRenderer):
    
        def __init__(self, obj: WorldObj, x: int, y: int, batch: pyglet.graphics.Batch, group: pyglet.graphics.Group, scale_factor: int = 1):
            super().__init__(obj)
            self.obj = obj
            self.sprite = pyglet.sprite.Sprite(img=unlocked_door, x=x, y=y, batch=batch, group=group)
            self.sprite.update(scale=scale_factor)

            self.type = None
            self.image = None

        def update(self, x: int, y: int):
            """
            Update state.
            """
            # update the sprite location
            self.sprite.update(x=x, y=y)

            if self.obj.is_open:
                self.sprite.image = opened_door
            elif self.obj.is_locked == False:
                self.sprite.image = unlocked_door
            else:
                self.sprite.image = locked_door

chests_sprite_sheet = pyglet.image.load('sprites/oryx_16bit_chests.png')
chests_grid = ImageGrid(chests_sprite_sheet, rows=1, columns=6)

open_chest_gold = chests_grid[5]

class PrettyGoalRenderer(PrettyObjRenderer):
        
        def __init__(self, obj: WorldObj, x: int, y: int, batch: pyglet.graphics.Batch, group: pyglet.graphics.Group, scale_factor: int = 1):
            super().__init__(obj)
            self.obj = obj
            self.sprite = pyglet.sprite.Sprite(img=open_chest_gold, x=x, y=y, batch=batch, group=group)
            self.sprite.update(scale=scale_factor)

        def update(self, x: int, y: int):
            """
            Update state.
            """
            # update the sprite location
            self.sprite.update(x=x, y=y)

rows = 8
columns = 24
mage_sprite_sheet = pyglet.image.load('sprites/character_base.png')
mage_grid = ImageGrid(mage_sprite_sheet, rows=rows, columns=columns)

# center all sprites
for img in mage_grid:
    img.anchor_x = img.width // 2
    img.anchor_y = img.height // 2

# walking sprites
frame_sequence = [2,3,2,3,1]
row=7
mage_down = [mage_grid[(columns * row) + i] for i in range(columns)]
walking_down_animation = Animation([AnimationFrame(mage_down[idx], 0.1) for idx in frame_sequence])
row=5
mage_right = [mage_grid[(columns * row) + i] for i in range(columns)]
walking_right_animation = Animation([AnimationFrame(mage_right[idx], 0.1) for idx in frame_sequence])
row=3
mage_up = [mage_grid[(columns * row) + i] for i in range(columns)]
walking_up_animation = Animation([AnimationFrame(mage_up[idx], 0.1) for idx in frame_sequence])
row=1
mage_left = [mage_grid[(columns * row) + i] for i in range(columns)]
walking_left_animation = Animation([AnimationFrame(mage_left[idx], 0.1) for idx in frame_sequence])

# idel sprites
standing_right = mage_right[1]
standing_left = mage_left[1]
standing_up = mage_up[1]
standing_down = mage_down[1]


# standing sprites (8 directions)
column = 0
standing = [mage_grid[(columns * i + 1) + column] for i in range(rows)]

# turning clockwise
turning_left_to_up_animation = Animation([AnimationFrame(standing[2], 0.1), AnimationFrame(standing[3], None)])
turning_up_to_right_animation = Animation([AnimationFrame(standing[4], 0.1), AnimationFrame(standing[5], None)])
turning_right_to_down_animation = Animation([AnimationFrame(standing[6], 0.1), AnimationFrame(standing[7], None)])
turning_down_to_left_animation = Animation([AnimationFrame(standing[0], 0.1), AnimationFrame(standing[1], None)])
# turning counter-clockwise
turning_left_to_down_animation = Animation([AnimationFrame(standing[0], 0.1), AnimationFrame(standing[7], None)])
turning_down_to_right_animation = Animation([AnimationFrame(standing[6], 0.1), AnimationFrame(standing[5], None)])
turning_right_to_up_animation = Animation([AnimationFrame(standing[4], 0.1), AnimationFrame(standing[3], None)])
turning_up_to_left_animation = Animation([AnimationFrame(standing[2], 0.1), AnimationFrame(standing[1], None)])

# Toggle animation
sequence = [15,16,17,15,1]
toggle_down_animation = Animation([AnimationFrame(mage_down[idx], 0.1) for idx in sequence])
toggle_right_animation = Animation([AnimationFrame(mage_right[idx], 0.1) for idx in sequence])
toggle_up_animation = Animation([AnimationFrame(mage_up[idx], 0.1) for idx in sequence])
toggle_left_animation = Animation([AnimationFrame(mage_left[idx], 0.1) for idx in sequence])

# Death animation
sequence = [19,20,21,22,23]
right_death_animation = Animation([AnimationFrame(mage_right[idx], 0.1) for idx in sequence])
down_death_animation = Animation([AnimationFrame(mage_down[idx], 0.1) for idx in sequence])
left_death_animation = Animation([AnimationFrame(mage_left[idx], 0.1) for idx in sequence])
up_death_animation = Animation([AnimationFrame(mage_up[idx], 0.1) for idx in sequence])




class PrettyAgentRenderer(ObjRenderer):

    # Agent directions
    # 0: right, 1: down, 2: left, 3: up

    def __init__(self, x: int, y: int, batch: pyglet.graphics.Batch, group: pyglet.graphics.Group, scale_factor: int = 1):
        super().__init__()
        self.sprite = pyglet.sprite.Sprite(img=standing[7], x=x, y=y, batch=batch, group=group)
        self.sprite.update(scale=scale_factor)

        # Sprite indexing parameters
        self.sprite_row = 7
        self.sprites_per_row = 26
        self.sprite_index = 0

        # Agent current state
        self.curr_action = 0
        self.curr_pos = (x, y)
        self.curr_dir = 0
        self.curr_carrying = None

    def update_agent_position(self, agent_pos: tuple[int, int]):
        """
        Update the agent position.
        """
        self.sprite.update(x=agent_pos[0], y=agent_pos[1])

        self.curr_pos = (agent_pos[0], agent_pos[1])

    def update_agent_animation(self, agent_action: int, agent_dir: int | None = None, carrying: WorldObj | None = None):
        """
        Update the animation of the agent.
        """

        if agent_action == Actions.forward:
            if agent_dir == 0: # right
                self.sprite.image = walking_right_animation
            elif agent_dir == 1: # down
                self.sprite.image = walking_down_animation
            elif agent_dir == 2: # left
                self.sprite.image = walking_left_animation
            elif agent_dir == 3: # up
                self.sprite.image = walking_up_animation

        elif agent_action == Actions.left: # rotate counter-clockwise
            if agent_dir == 0: # right
                self.sprite.image = turning_down_to_right_animation
            elif agent_dir == 1: # down
                self.sprite.image = turning_left_to_down_animation
            elif agent_dir == 2: # left
                self.sprite.image = turning_up_to_left_animation
            elif agent_dir == 3: # up
                self.sprite.image = turning_right_to_up_animation

        elif agent_action == Actions.right: # rotate clockwise
            if agent_dir == 0: # right
                self.sprite.image = turning_up_to_right_animation
            elif agent_dir == 1: # down
                self.sprite.image = turning_right_to_down_animation
            elif agent_dir == 2: # left
                self.sprite.image = turning_down_to_left_animation
            elif agent_dir == 3: # up
                self.sprite.image = turning_left_to_up_animation

        elif agent_action == Actions.toggle or agent_action == Actions.pickup or agent_action == Actions.drop:
            if agent_dir == 0:
                self.sprite.image = toggle_right_animation
            elif agent_dir == 1:
                self.sprite.image = toggle_down_animation
            elif agent_dir == 2:
                self.sprite.image = toggle_left_animation
            elif agent_dir == 3:
                self.sprite.image = toggle_up_animation

        elif agent_action == Actions.done:
            if agent_dir == 0:
                self.sprite.image = right_death_animation
            elif agent_dir == 1:
                self.sprite.image = down_death_animation
            elif agent_dir == 2:
                self.sprite.image = left_death_animation
            elif agent_dir == 3:
                self.sprite.image = up_death_animation

        else:
            if agent_dir == 0: # facing right
                self.sprite.image = standing_right
            elif agent_dir == 1: # facing down
                self.sprite.image = standing_down
            elif agent_dir == 2: # facing left
                self.sprite.image = standing_left
            elif agent_dir == 3: # facing up
                self.sprite.image = standing_up

rows = 11
columns = 3

lava_sprite_sheet = pyglet.image.load('sprites/oryx_16bit_lava.png')
lava_grid = ImageGrid(lava_sprite_sheet, rows=rows, columns=columns)

lava_animation = Animation([AnimationFrame(lava_grid[21], 0.1), AnimationFrame(lava_grid[18], 0.1)])


        
class PrettyLavaRenderer(ObjRenderer):
    
    def __init__(self, obj: WorldObj, x: int, y: int, batch: pyglet.graphics.Batch, group: pyglet.graphics.Group, scale_factor: int = 1):
        super().__init__()
        self.sprite = pyglet.sprite.Sprite(img=lava_animation, x=x, y=y, batch=batch, group=group)
        self.sprite.update(scale=scale_factor)

    def update(self, x: int, y: int):
        """
        Update state.
        """
        # update the sprite location
        self.sprite.update(x=x, y=y)

class PrettyKeyRenderer(ObjRenderer):
    pass

class PrettyBallRenderer(ObjRenderer):
    pass

class PrettyBoxRenderer(ObjRenderer):
    pass


bones_sprite_sheet = pyglet.image.load('sprites/oryx_16bit_bones.png')
bones_grid = ImageGrid(bones_sprite_sheet, rows=1, columns=7)

small_bones_1 = bones_grid[0]
small_bones_2 = bones_grid[1]
small_bones_3 = bones_grid[2]
small_bones_4 = bones_grid[3]

large_bones_1 = bones_grid[4]
large_bones_2 = bones_grid[5]
large_bones_3 = bones_grid[6]

class PrettyBonesRenderer(PrettyObjRenderer):
      pass