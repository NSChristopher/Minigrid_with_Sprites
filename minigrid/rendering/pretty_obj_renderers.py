import os
import numpy as np
from minigrid.core.constants import COLORS
from minigrid.core.world_object import WorldObj
from minigrid.utils.rendering import find_nearest_neighbor
from abc import ABC
from minigrid.core.actions import Actions

import pyglet
from pyglet.image import ImageGrid
from pyglet.image.animation import Animation, AnimationFrame


class PrettyObjRenderer(ABC):

    def __init__(self, obj: WorldObj | None):
        super().__init__()
        self.obj = obj
        self.sprite = None

    def update(self, x: int, y: int):
        pass

    def delete_sprite(self):
        """
        Delete the sprite.
        """
        if self.sprite is not None:
            self.sprite.delete()
        self.sprite = None

sprites_dir = os.path.join(os.path.dirname(__file__), '../../sprites')


environments_sprite_sheet = pyglet.image.load(os.path.join(sprites_dir, 'oryx_16bit_walls.png'))
floor_shadows_sprite_sheet = pyglet.image.load(os.path.join(sprites_dir, 'oryx_16bit_floor_shadows.png'))
environments_grid = ImageGrid(environments_sprite_sheet, rows=23, columns=27, row_padding=0, column_padding=0)

# all sprites from row 19 starting from bottom left corner
access_row = 18
env_tiles = [environments_grid[(27 * access_row) + i] for i in range(27)]

wall_tiles = env_tiles[:3] + env_tiles[9:]
floor_tiles = env_tiles[3:9]

# wall tile mapping
WALL_TYPES = {
    (0,0,0, 0,2,0, 0,0,0): (0,1),  # Single standalone wall
    (0,0,0, 0,2,2, 0,0,0): (4,4),  # Left cap wall (horizontal)
    (0,0,0, 2,2,2, 0,0,0): (5,20), # Horizontal wall segment
    (0,0,0, 2,2,0, 0,0,0): (6,6),  # Right cap wall (horizontal)
    (0,0,0, 0,2,0, 0,2,0): (7,7),  # Top cap wall (vertical)
    (0,2,0, 0,2,0, 0,2,0): (8,19), # Vertical wall segment
    (0,2,0, 0,2,0, 0,0,0): (9,9),  # Bottom cap wall (vertical)
    (0,0,0, 0,2,2, 0,2,0): (10,10),# Top-left corner wall
    (0,0,0, 2,2,0, 0,2,0): (11,11),# Top-right corner wall
    (0,2,0, 0,2,2, 0,0,0): (12,12),# Bottom-left corner wall
    (0,2,0, 2,2,0, 0,0,0): (13,13),# Bottom-right corner wall
    (0,2,0, 2,2,2, 0,2,0): (14,14),# Cross intersection wall
    (0,0,0, 2,2,2, 0,2,0): (15,15),# T-junction wall (top-open)
    (0,2,0, 2,2,0, 0,2,0): (16,16),# T-junction wall (right-open)
    (0,2,0, 0,2,2, 0,2,0): (17,17),# T-junction wall (left-open)
    (0,2,0, 2,2,2, 0,0,0): (18,18),# T-junction wall (bottom-open)
    (0,2,0, 0,2,0, 2,2,2): (8,19), # Vertical wall segment
    (2,0,0, 2,2,2, 2,0,0): (5,20), # Horizontal wall segment
    (2,2,2, 0,2,0, 0,2,0): (8,19), # Vertical wall segment
    (0,0,2, 2,2,2, 0,0,2): (5,20), # Horizontal wall segment
    (0,0,0, 0,2,0, 2,2,2): (7,7), # top cap wall (vertical)
    (2,0,0, 2,2,0, 2,0,0): (6,6), # right cap wall (horizontal)
    (2,2,2, 0,2,0, 0,0,0): (9,9), # bottom cap wall (vertical)
    (0,0,2, 0,2,2, 0,0,2): (4,4), # left cap wall (horizontal)
    (0,0,0, 2,2,2, 2,0,0): (5,20), # Horizontal wall segment
    (2,0,0, 2,2,2, 0,0,0): (5,20), # Horizontal wall segment
    (0,0,0, 2,2,2, 0,0,2): (5,20), # Horizontal wall segment
    (0,0,2, 2,2,2, 0,0,0): (5,20), # Horizontal wall segment
    (0,2,0, 0,2,0, 2,2,0): (8,19), # Vertical wall segment
    (0,2,0, 0,2,0, 0,2,2): (8,19), # Vertical wall segment
    (2,2,0, 0,2,0, 0,2,0): (8,19), # Vertical wall segment
    (0,2,2, 0,2,0, 0,2,0): (8,19), # Vertical wall segment
}

class PrettyWallRenderer(PrettyObjRenderer):
    
    def __init__(self, obj: WorldObj, x: int, y: int, batch: pyglet.graphics.Batch, group: pyglet.graphics.Group, scale_factor: float = 1):
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

        self.set_type(self.proximity_encoding)

    def set_type(self, proximity_encoding: np.ndarray):
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
        
        def __init__(self, obj: WorldObj | None, x: int, y: int, batch: pyglet.graphics.Batch, group: pyglet.graphics.Group, scale_factor: float = 1):
            super().__init__(obj)
            self.obj = obj
            self.sprite = pyglet.sprite.Sprite(img=floor_tiles[1], x=x, y=y, batch=batch, group=group)
            self.sprite.update(scale=scale_factor)

        def set_type(self, type: int):
            """
            Set the floor type.
            """
            self.sprite.image = floor_tiles[type]


door_sprite_sheet = pyglet.image.load(os.path.join(sprites_dir, 'oryx_16bit_doors.png'))
door_grid = ImageGrid(door_sprite_sheet, rows=1, columns=14)

greyscale_door_sprite_sheet = pyglet.image.load(os.path.join(sprites_dir, 'oryx_16bit_doors_greyscale.png'))
greyscale_door_grid = ImageGrid(greyscale_door_sprite_sheet, rows=1, columns=14)

unlocked_door = door_grid[0]
greyscale_unlocked_door = greyscale_door_grid[0]
opened_door = door_grid[1]
greyscale_opened_door = greyscale_door_grid[1]
locked_door = door_grid[2]
greyscale_locked_door = greyscale_door_grid[2]

class PrettyDoorRenderer(PrettyObjRenderer):
    
        def __init__(self, obj: WorldObj, x: int, y: int, batch: pyglet.graphics.Batch, group: pyglet.graphics.Group, scale_factor: float = 1):
            super().__init__(obj)

            # images
            self.unlocked_door = unlocked_door
            self.opened_door = opened_door
            self.locked_door = locked_door

            self.obj = obj
            self.sprite = pyglet.sprite.Sprite(img=unlocked_door, x=x, y=y, batch=batch, group=group)
            self.sprite.update(scale=scale_factor)

            self.type = None
            self.image = None

            # object color
            if obj.color is not None:
                self.unlocked_door = greyscale_unlocked_door
                self.opened_door = greyscale_opened_door
                self.locked_door = greyscale_locked_door
                self.sprite.color = COLORS[obj.color]



        def update(self, x: int, y: int):
            """
            Update state.
            """
            # update the sprite location
            self.sprite.update(x=x, y=y)

            if self.obj.is_open:
                self.sprite.image = self.opened_door
            elif self.obj.is_locked == False:
                self.sprite.image = self.unlocked_door
            else:
                self.sprite.image = self.locked_door

chests_sprite_sheet = pyglet.image.load(os.path.join(sprites_dir, 'oryx_16bit_chests.png'))
chests_grid = ImageGrid(chests_sprite_sheet, rows=1, columns=6)

open_chest_gold = chests_grid[5]

class PrettyGoalRenderer(PrettyObjRenderer):
        
        def __init__(self, obj: WorldObj, x: int, y: int, batch: pyglet.graphics.Batch, group: pyglet.graphics.Group, scale_factor: float = 1):
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
mage_sprite_sheet = pyglet.image.load(os.path.join(sprites_dir, 'mage_red.png'))
mage_grid = ImageGrid(mage_sprite_sheet, rows=rows, columns=columns)

# center all sprites
for img in mage_grid:
    img.anchor_x = img.width // 2
    img.anchor_y = img.height // 2

# walking sprites
frame_sequence = [1,2,3]
row=7
mage_down = [mage_grid[(columns * row) + i] for i in range(columns)]
walking_down_animation = Animation([AnimationFrame(mage_down[idx], 0.01) for idx in frame_sequence])
row=5
mage_right = [mage_grid[(columns * row) + i] for i in range(columns)]
walking_right_animation = Animation([AnimationFrame(mage_right[idx], 0.01) for idx in frame_sequence])
row=3
mage_up = [mage_grid[(columns * row) + i] for i in range(columns)]
walking_up_animation = Animation([AnimationFrame(mage_up[idx], 0.01) for idx in frame_sequence])
row=1
mage_left = [mage_grid[(columns * row) + i] for i in range(columns)]
walking_left_animation = Animation([AnimationFrame(mage_left[idx], 0.01) for idx in frame_sequence])

# idel sprites
standing_right = mage_right[1]
standing_left = mage_left[1]
standing_up = mage_up[1]
standing_down = mage_down[1]


# standing sprites (8 directions)
column = 0
standing = [mage_grid[(columns * i + 1) + column] for i in range(rows)]

# turning clockwise
turning_left_to_up_animation = Animation([AnimationFrame(standing[2], 0.01), AnimationFrame(standing[3], None)])
turning_up_to_right_animation = Animation([AnimationFrame(standing[4], 0.01), AnimationFrame(standing[5], None)])
turning_right_to_down_animation = Animation([AnimationFrame(standing[6], 0.01), AnimationFrame(standing[7], None)])
turning_down_to_left_animation = Animation([AnimationFrame(standing[0], 0.01), AnimationFrame(standing[1], None)])
# turning counter-clockwise
turning_left_to_down_animation = Animation([AnimationFrame(standing[0], 0.01), AnimationFrame(standing[7], None)])
turning_down_to_right_animation = Animation([AnimationFrame(standing[6], 0.01), AnimationFrame(standing[5], None)])
turning_right_to_up_animation = Animation([AnimationFrame(standing[4], 0.01), AnimationFrame(standing[3], None)])
turning_up_to_left_animation = Animation([AnimationFrame(standing[2], 0.01), AnimationFrame(standing[1], None)])

# Toggle animation
sequence = [15,16,17,15,1]
toggle_down_animation = Animation([AnimationFrame(mage_down[idx], 0.01) for idx in sequence])
toggle_right_animation = Animation([AnimationFrame(mage_right[idx], 0.01) for idx in sequence])
toggle_up_animation = Animation([AnimationFrame(mage_up[idx], 0.01) for idx in sequence])
toggle_left_animation = Animation([AnimationFrame(mage_left[idx], 0.01) for idx in sequence])

# Death animation
sequence = [19,20,21,22,23]
right_death_animation = Animation([AnimationFrame(mage_right[idx], 0.01) for idx in sequence])
down_death_animation = Animation([AnimationFrame(mage_down[idx], 0.01) for idx in sequence])
left_death_animation = Animation([AnimationFrame(mage_left[idx], 0.01) for idx in sequence])
up_death_animation = Animation([AnimationFrame(mage_up[idx], 0.01) for idx in sequence])




class PrettyAgentRenderer(PrettyObjRenderer):

    # Agent directions
    # 0: right, 1: down, 2: left, 3: up

    def __init__(self, x: int, y: int, batch: pyglet.graphics.Batch, group: pyglet.graphics.Group, scale_factor: float = 1):
        super().__init__(None)
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

lava_sprite_sheet = pyglet.image.load(os.path.join(sprites_dir, 'oryx_16bit_lava.png'))
lava_grid = ImageGrid(lava_sprite_sheet, rows=rows, columns=columns)

lava_animation = Animation([AnimationFrame(lava_grid[21], 0.01), AnimationFrame(lava_grid[18], 0.01)])

LAVA_EDGE_TYPES = {
    (9,9,9, 9,0,0, 9,0,0): 30, # top left corner edge
    (9,9,9, 0,0,0, 0,0,0): 31, # top edge
    (9,9,9, 0,0,9, 0,0,9): 32, # top right corner edge
    (9,0,0, 9,0,0, 9,0,0): 27, # left edge
    (9,9,9, 9,0,9, 9,9,9): 28, # edge on all sides
    (0,0,9, 0,0,9, 9,9,9): 29, # right edge
    (9,0,0, 9,0,0, 9,9,9): 24, # bottom left corner edge
    (0,0,0, 0,0,0, 9,9,9): 25, # bottom edge
    (0,0,9, 0,0,9, 9,9,9): 26, # bottom right corner edge
    (9,9,9, 9,0,9, 9,0,9): 22, # left right and top edges
    (9,9,9, 9,0,9, 0,0,0): 22, # left right and top edges   
    (9,9,9, 0,0,9, 9,9,9): 23, # top right and bottom edges
    (0,9,9, 0,0,9, 0,9,9): 23, # top right and bottom edges
    (9,0,9, 9,0,9, 9,9,9): 19, # left right and bottom edges
    (9,9,9, 9,0,0, 9,9,9): 20, # top left and bottom edges
    (9,9,9, 9,0,0,9,0,9): 15, # top left and bottom right corner
    (9,9,9, 0,0,9, 9,0,9): 16, # top right and bottom left corner
    (9,0,9, 9,0,0, 9,9,9): 12, # left bottom and top right corner
    (9,0,9, 0,0,9, 9,9,9): 13, # right bottom and top left corner
    (9,0,0, 0,0,0, 0,0,0): 9, # top left corner
    (0,0,9, 0,0,0, 0,0,0): 10, # top right corner
    (9,9,9, 0,0,0, 9,9,9): 11, # top and bottom edges
    (0,0,0, 9,0,0, 0,0,0): 6, # bottom left corner
    (0,0,0, 0,0,9, 0,0,0): 7, # bottom right corner
    (9,0,9, 9,0,9, 9,0,9): 8, # left and right edges
    (0,0,0, 0,0,0, 9,0,9): 3, # bottom left and right corners
    (9,0,9 ,0,0,0, 0,0,0): 4, # top left and right corners
    (9,0,9, 0,0,0, 9,0,9): 5, # all corners
    (9,0,0, 0,0,0, 9,0,0): 0, # top left and bottom left corners
    (0,0,9, 0,0,0, 0,0,9): 1, # top right and bottom right corners
    (0,0,0, 0,0,0, 0,0,0): 2, # no edges
}


        
class PrettyLavaRenderer(PrettyObjRenderer):
    
    def __init__(self, obj: WorldObj, x: int, y: int, batch: pyglet.graphics.Batch, group: pyglet.graphics.Group, scale_factor: float = 1):
        super().__init__(obj=None)
        self.sprite = pyglet.sprite.Sprite(img=lava_animation, x=x, y=y, batch=batch, group=group)
        self.edges_sprite = pyglet.sprite.Sprite(img=lava_grid[2], x=x, y=y, batch=batch, group=pyglet.graphics.Group(2))
        self.sprite.update(scale=scale_factor)
        self.edges_sprite.update(scale=scale_factor)


        self.lava_grid = lava_grid
        self.proximity_encoding = None

    def set_proximity_encoding(self, encoding: np.ndarray):
        """
        Set the proximity encoding of the lava object
        """
        self.proximity_encoding = encoding

        self.set_type(self.proximity_encoding)

    def set_type(self, proximity_encoding: np.ndarray):
        """
        Set the lava type based on the proximity encoding.
        """
        # set anything thats not lava to 9 and lava to 0
        mapped_proximity = np.where(proximity_encoding == 9, 0, 9)
        prox_flat = tuple(int(val) for val in mapped_proximity.flatten())

        self.prox_flat = prox_flat
        image = None
        if prox_flat in LAVA_EDGE_TYPES:
            image = lava_grid[LAVA_EDGE_TYPES[prox_flat]]
        else:
            best_match = find_nearest_neighbor(prox_flat, list(LAVA_EDGE_TYPES.keys()))
            image = lava_grid[LAVA_EDGE_TYPES[best_match]]

        self.edges_sprite.image = image

    def update(self, x: int, y: int):
        """
        Update state.
        """
        # update the sprite location
        self.sprite.update(x=x, y=y)

greyscale_keys_sprite_sheet = pyglet.image.load(os.path.join(sprites_dir, 'oryx_16bit_keys_greyscale.png'))
greyscale_keys_grid = ImageGrid(greyscale_keys_sprite_sheet, rows=1, columns=6)

key = greyscale_keys_grid[4]

class PrettyKeyRenderer(PrettyObjRenderer):
    
    def __init__(self, obj: WorldObj, x: int, y: int, batch: pyglet.graphics.Batch, group: pyglet.graphics.Group, scale_factor: float = 1):
        super().__init__(obj)
        self.sprite = pyglet.sprite.Sprite(img=key, x=x, y=y, batch=batch, group=group)
        self.sprite.update(scale=scale_factor)

        if obj.color is not None:
            self.sprite.color = COLORS[obj.color]

    def update(self, x: int, y: int):
        """
        Update state.
        """
        # update the sprite location
        self.sprite.update(x=x, y=y)

greysacle_ball = pyglet.image.load(os.path.join(sprites_dir, 'oryx_16bit_ball_greyscale.png'))

class PrettyBallRenderer(PrettyObjRenderer):
    
    def __init__(self, obj: WorldObj, x: int, y: int, batch: pyglet.graphics.Batch, group: pyglet.graphics.Group, scale_factor: float = 1):
        super().__init__(obj)
        self.sprite = pyglet.sprite.Sprite(img=greysacle_ball, x=x, y=y, batch=batch, group=group)
        self.sprite.update(scale=scale_factor)

        if obj.color is not None:
            self.sprite.color = COLORS[obj.color]

    def update(self, x: int, y: int):
        """
        Update state.
        """
        # update the sprite location
        self.sprite.update(x=x, y=y)

box_sprite_sheet = pyglet.image.load(os.path.join(sprites_dir, 'oryx_16bit_box_greyscale.png'))
box_grid = ImageGrid(box_sprite_sheet, rows=1, columns=2)

box_closed = box_grid[0]
box_opened = box_grid[1]


class PrettyBoxRenderer(PrettyObjRenderer):
    
    def __init__(self, obj: WorldObj, x: int, y: int, batch: pyglet.graphics.Batch, group: pyglet.graphics.Group, scale_factor: float = 1):
        super().__init__(obj)
        self.sprite = pyglet.sprite.Sprite(img=box_closed, x=x, y=y, batch=batch, group=group)
        self.sprite.update(scale=scale_factor)

        if obj.color is not None:
            self.sprite.color = COLORS[obj.color]

    def update(self, x: int, y: int):
        """
        Update state.
        """
        # update the sprite location
        self.sprite.update(x=x, y=y)


bones_sprite_sheet = pyglet.image.load(os.path.join(sprites_dir, 'oryx_16bit_bones.png'))
bones_grid = ImageGrid(bones_sprite_sheet, rows=1, columns=7)

small_bones_1 = bones_grid[0]
small_bones_2 = bones_grid[1]
small_bones_3 = bones_grid[2]
small_bones_4 = bones_grid[3]

large_bones_1 = bones_grid[4]
large_bones_2 = bones_grid[5]
large_bones_3 = bones_grid[6]

class PrettyBonesRenderer(PrettyObjRenderer):
      
    def __init__(self, x: int, y: int, batch: pyglet.graphics.Batch, group: pyglet.graphics.Group, scale_factor: float = 1):
        super().__init__(None)
        self.sprite = pyglet.sprite.Sprite(img=small_bones_1, x=x, y=y, batch=batch, group=group)
        self.sprite.update(scale=scale_factor)
        
    def set_type(self, type: int):

        rand = np.random.rand(1)
        rand2 = np.random.rand(1)

        if rand2 > 0.40:
            if type == 1:
                if rand < 0.25:
                    self.sprite.image = small_bones_1
                elif rand < 0.5:
                    self.sprite.image = small_bones_2
                elif rand < 0.75:
                    self.sprite.image = small_bones_3
                else:
                    self.sprite.image = small_bones_4
            elif type == 2:
                if rand < 0.33:
                    self.sprite.image = large_bones_1
                elif rand < 0.66:
                    self.sprite.image = large_bones_2
                else:
                    self.sprite.image = large_bones_3
        else:
            del self.sprite

floor_shadows_sprite_sheet = pyglet.image.load(os.path.join(sprites_dir, 'oryx_16bit_floor_shadows.png'))
floor_shadows_grid = ImageGrid(floor_shadows_sprite_sheet, rows=1, columns=8)

shadow = floor_shadows_grid[2]

class PrettyShadowRenderer(PrettyObjRenderer):

    def __init__(self, x: int, y: int, batch: pyglet.graphics.Batch, group: pyglet.graphics.Group, scale_factor: float = 1):
        super().__init__(obj=None)
        self.sprite = pyglet.sprite.Sprite(img=shadow, x=x, y=y, batch=batch, group=group)
        self.sprite.update(scale=scale_factor)

    def update(self, x: int, y: int):
        """
        Update state.
        """
        # update the sprite location
        self.sprite.update(x=x, y=y)

web_sprite_sheet = pyglet.image.load(os.path.join(sprites_dir, 'oryx_16bit_webs.png'))
web_grid = ImageGrid(web_sprite_sheet, rows=1, columns=6)

top_left_web = web_grid[0]
top_right_web = web_grid[1]
bottom_right_web = web_grid[2]
bottom_left_web = web_grid[3]

class PrettyWebRenderer(PrettyObjRenderer):

    def __init__(self, x: int, y: int, batch: pyglet.graphics.Batch, group: pyglet.graphics.Group, scale_factor: float = 1):
        super().__init__(obj=None)
        self.sprite = pyglet.sprite.Sprite(img=top_left_web, x=x, y=y, batch=batch, group=group)
        self.sprite.update(scale=scale_factor)

    def set_type(self, type: int):
        """
        Set the proximity encoding of the web object, set the wall type 
        and set the sprite location.
        """

        if type == 0:
            self.sprite.image = top_left_web
        elif type == 1:
            self.sprite.image = top_right_web
        elif type == 2:
            self.sprite.image = bottom_right_web
        elif type == 3:
            self.sprite.image = bottom_left_web

        
