from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np


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

    def render(self, r: np.ndarray) -> np.ndarray:
        """Draw this object with the given renderer"""
        raise NotImplementedError


class Goal(WorldObj):
    def __init__(self):
        super().__init__("goal", "green")

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, color: str = "blue"):
        super().__init__("floor", color)

    def can_overlap(self):
        return True

    def render(self, img):
        # Give the floor a pale color
        color = COLORS[self.color] / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)



class Lava(WorldObj):
    def __init__(self):
        super().__init__("lava", "red")

    def can_overlap(self):
        return True

    def render(self, img):
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))


class Wall(WorldObj):
    def __init__(self, color: str = "grey"):
        super().__init__("wall", color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])



class Door(WorldObj):
    def __init__(self, color: str, is_open: bool = False, is_locked: bool = False):
        super().__init__("door", color)
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

    def render(self, img):
        c = COLORS[self.color]

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
            return

        # Door frame and door
        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)


class Key(WorldObj):
    def __init__(self, color: str = "blue"):
        super().__init__("key", color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


class Ball(WorldObj):
    def __init__(self, color="blue"):
        super().__init__("ball", color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


class Box(WorldObj):
    def __init__(self, color, contains: WorldObj | None = None):
        super().__init__("box", color)
        self.contains = contains

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos):
        # Replace the box by its contents
        env.grid.set(pos[0], pos[1], self.contains)
        return True


class Monster(WorldObj):
    def __init__(self, color="red", direction=0):
        super().__init__("monster", color)
        self.direction = direction
        self.position = None
        self.path = []
        self.view_size = 7

    def can_overlap(self):
        return False

    def render(self, img):
        # Draw the monster as a red square
        c = COLORS[self.color]
        fill_coords(img, point_in_rect(0.3, 0.7, 0.3, 0.7), c)

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

