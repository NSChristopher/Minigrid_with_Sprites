from __future__ import annotations

from typing import Any

import numpy as np

from minigrid.core.constants import DIR_TO_VEC

from minigrid.renderers.obj_renderer import (
    ObjRenderer,
    GoalRenderer,
    FloorRenderer,
    LavaRenderer,
    WallRenderer,
    DoorRenderer,
    KeyRenderer,
    BallRenderer,
    BoxRenderer,
)

# override the WallRenderer class with the PrettyWallRenderer class
from minigrid.renderers.pretty_obj_renderers import PrettyWallRenderer as WallRenderer


from typing import TYPE_CHECKING, Tuple

from minigrid.core.constants import (
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
)

if TYPE_CHECKING:
    from minigrid.minigrid_env import MiniGridEnv

Point = Tuple[int, int]


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type: str, color: str):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos: Point | None = None

        # Current position of the object
        self.cur_pos: Point | None = None

        # object renderer
        self.renderer: ObjRenderer

    def can_overlap(self) -> bool:
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self) -> bool:
        """Can the agent pick this up?"""
        return False

    def can_contain(self) -> bool:
        """Can this contain another object?"""
        return False

    def see_behind(self) -> bool:
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env: MiniGridEnv, pos: tuple[int, int]) -> bool:
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self) -> tuple[int, int, int]:
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

    @staticmethod
    def decode(type_idx: int, color_idx: int, state: int) -> WorldObj | None:
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == "empty" or obj_type == "unseen" or obj_type == "agent":
            return None

        # State, 0: open, 1: closed, 2: locked
        is_open = state == 0
        is_locked = state == 2

        if obj_type == "wall":
            v = Wall(color)
        elif obj_type == "floor":
            v = Floor(color)
        elif obj_type == "ball":
            v = Ball(color)
        elif obj_type == "key":
            v = Key(color)
        elif obj_type == "box":
            v = Box(color)
        elif obj_type == "door":
            v = Door(color, is_open, is_locked)
        elif obj_type == "goal":
            v = Goal()
        elif obj_type == "lava":
            v = Lava()
        elif obj_type == "monster":
            v = Monster()
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v

    def render(self, img: np.ndarray):
        """Draw this object with the given renderer"""
        self.renderer.render(img)


class Goal(WorldObj):
    def __init__(self):
        super().__init__("goal", "green")
        self.renderer = GoalRenderer(obj = self)

    def can_overlap(self):
        return True


class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, color: str = "blue"):
        super().__init__("floor", color)
        self.renderer = FloorRenderer(obj = self)

    def can_overlap(self):
        return True


class Lava(WorldObj):
    def __init__(self):
        super().__init__("lava", "orange")
        self.renderer = LavaRenderer(obj = self)

    def can_overlap(self):
        return True


class Wall(WorldObj):
    def __init__(self, color: str = "grey"):
        super().__init__("wall", color)
        self.renderer = WallRenderer(obj = self)

    def see_behind(self):
        return False

class Door(WorldObj):
    def __init__(self, color: str, is_open: bool = False, is_locked: bool = False):
        super().__init__("door", color)
        self.renderer = DoorRenderer(obj = self)
        self.is_open = is_open
        self.is_locked = is_locked

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        if self.is_open:
            state = 0
        elif self.is_locked:
            state = 2
        # if door is closed and unlocked
        elif not self.is_open:
            state = 1
        else:
            raise ValueError(
                f"There is no possible state encoding for the state:\n -Door Open: {self.is_open}\n -Door Closed: {not self.is_open}\n -Door Locked: {self.is_locked}"
            )

        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)


class Key(WorldObj):
    def __init__(self, color: str = "blue"):
        super().__init__("key", color)
        self.renderer = KeyRenderer(obj = self)

    def can_pickup(self):
        return True


class Ball(WorldObj):
    def __init__(self, color="blue"):
        super().__init__("ball", color)
        self.renderer = BallRenderer(obj = self)

    def can_pickup(self):
        return True


class Box(WorldObj):
    def __init__(self, color, contains: WorldObj | None = None):
        super().__init__("box", color)
        self.renderer = BoxRenderer(obj = self)
        self.contains = contains

    def can_pickup(self):
        return True

    def toggle(self, env, pos):
        # Replace the box by its contents
        env.grid.set(pos[0], pos[1], self.contains)
        return True

class Monster(WorldObj):
    def __init__(self, color="red", direction=0):
        super().__init__("monster", color)
        self.renderer = GoalRenderer(obj = self)
        self.direction = direction
        self.position = None
        self.path = []
        self.view_size = 7

    def can_overlap(self):
        return False

    def get_view_exts(self):
        """
        Get the extents of the square set of tiles visible to the monster
        """
        x, y = self.position

        # Facing right
        if self.direction == 0:
            topX = x
            topY = y - self.view_size // 2
        # Facing down
        elif self.direction == 1:
            topX = x - self.view_size // 2
            topY = y
        # Facing left
        elif self.direction == 2:
            topX = x - self.view_size + 1
            topY = y - self.view_size // 2
        # Facing up
        elif self.direction == 3:
            topX = x - self.view_size // 2
            topY = y - self.view_size + 1
        else:
            assert False, "Invalid monster direction"

        botX = topX + self.view_size
        botY = topY + self.view_size

        return topX, topY, botX, botY

    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        monster's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the monster's view size.
        """
        mx, my = self.position
        dx, dy = DIR_TO_VEC[self.direction]
        rx, ry = -dy, dx  # Right vector perpendicular to the direction vector

        # Compute the absolute coordinates of the top-left view corner
        sz = self.view_size
        hs = self.view_size // 2
        tx = mx + (dx * (sz - 1)) - (rx * hs)
        ty = my + (dy * (sz - 1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the monster's own coordinate system
        vx = rx * lx + ry * ly
        vy = -(dx * lx + dy * ly)

        return vx, vy

    def gen_obs_grid(self, grid):
        """
        Generate the sub-grid observed by the monster.
        This method also outputs a visibility mask telling us which grid
        cells the monster can actually see.
        """
        topX, topY, botX, botY = self.get_view_exts()

        # Slice out the monster's field of view
        grid = grid.slice(topX, topY, self.view_size, self.view_size)

        # Rotate the grid to match the monster's direction
        for i in range(self.direction + 1):
            grid = grid.rotate_left()

        # Process occluders and visibility, similar to the agent
        vis_mask = grid.process_vis(
            agent_pos=(self.view_size // 2, self.view_size - 1)
        )

        return grid, vis_mask

    def can_see(self, grid, target_pos):
        grid, vis_mask = self.gen_obs_grid(grid)
        tx, ty = target_pos

        # Convert target position to relative coordinates in the monster's view
        relative_coords = self.relative_coords(tx, ty)
        if relative_coords is None:
            return False

        vx, vy = relative_coords

        return vis_mask[vx, vy]

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the monster's field of view, and returns the corresponding coordinates
        """
        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.view_size or vy >= self.view_size:
            return None

        return vx, vy
    
    def move_towards(self, grid, target_pos):
        """
        Move towards the target position from the start position.
        If a wall or another object is encountered, or the move goes out of bounds, the monster changes direction.
        """
        sx, sy = self.position
        tx, ty = target_pos
        dx, dy = tx - sx, ty - sy

        # Determine the direction to move
        if abs(dx) > abs(dy):
            new_x = sx + (1 if dx > 0 else -1)
            new_y = sy
        else:
            new_x = sx
            new_y = sy + (1 if dy > 0 else -1)

        # Ensure the new position is within bounds and is empty
        if not (0 <= new_x < grid.width and 0 <= new_y < grid.height):
            self.point_random_direction()
        else:
            next_cell = grid.get(new_x, new_y)
            if next_cell is None:
                # Clear current position
                grid.set(sx, sy, None)
                # Move to the new position
                self.position = (new_x, new_y)
                grid.set(new_x, new_y, self)
            else:
                self.point_random_direction()

        return self.position

    def patrol_forward(self, grid):
        """
        Move the monster one space in its current direction if the path is clear.
        If a wall or another object is encountered, or the move goes out of bounds, the monster changes direction.
        """
        sx, sy = self.position
        dx, dy = DIR_TO_VEC[self.direction]
        new_x = sx + dx
        new_y = sy + dy

        # Ensure the new position is within bounds and is empty
        if not (0 <= new_x < grid.width and 0 <= new_y < grid.height):
            self.point_random_direction()
        else:
            next_cell = grid.get(new_x, new_y)
            if next_cell is None:
                # Clear current position
                grid.set(sx, sy, None)
                # Move to the new position
                self.position = (new_x, new_y)
                grid.set(new_x, new_y, self)
            else:
                self.point_random_direction()

        # random chance to change direction
        if np.random.rand() < 0.25:
            self.point_random_direction()

        return self.position

    
    def point_random_direction(self):
        """
        Choose a random direction for the monster to move in
        """
        self.direction = np.random.randint(0, 4)