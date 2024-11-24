from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from minigrid.core.constants import OBJECT_TO_IDX, TILE_PIXELS
from minigrid.core.world_object import Wall, WorldObj
from minigrid.utils.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)


class Grid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache: dict[tuple[Any, ...], Any] = {}

    def __init__(self, width: int, height: int):
        assert width >= 3
        assert height >= 3

        self.width: int = width
        self.height: int = height

        self.grid: np.ndarray = np.full((width, height), None, dtype=object)

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, WorldObj):
            for e in self.grid:
                if e is key:
                    return True
        elif isinstance(key, tuple):
            for e in self.grid:
                if e is None:
                    continue
                if (e.color, e.type) == key:
                    return True
                if key[0] is None and key[1] == e.type:
                    return True
        return False

    def __eq__(self, other: Grid) -> bool:
        grid1 = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other: Grid) -> bool:
        return not self == other

    def copy(self) -> Grid:
        from copy import deepcopy

        new_grid = Grid(self.width, self.height)
        new_grid.grid = deepcopy(self.grid)
        return new_grid

    def set(self, i: int, j: int, v: WorldObj | None):
        assert (
            0 <= i < self.width
        ), f"column index {i} outside of grid of width {self.width}"
        assert (
            0 <= j < self.height
        ), f"row index {j} outside of grid of height {self.height}"
        self.grid[i, j] = v

    def get(self, i: int, j: int) -> WorldObj | None:
        assert 0 <= i < self.width
        assert 0 <= j < self.height
        return self.grid[i, j]

    def horz_wall(
        self,
        x: int,
        y: int,
        length: int | None = None,
        obj_type: Callable[[], WorldObj] = Wall,
    ):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type())

    def vert_wall(
        self,
        x: int,
        y: int,
        length: int | None = None,
        obj_type: Callable[[], WorldObj] = Wall,
    ):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type())

    def wall_rect(self, x: int, y: int, w: int, h: int):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y + h - 1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x + w - 1, y, h)

    def rotate_left(self) -> Grid:
        """
        Rotate the grid to the left (counter-clockwise)
        """
        new_grid = Grid(self.height, self.width)
        new_grid.grid = np.rot90(self.grid, k=1)

        return new_grid

    def slice(self, topX: int, topY: int, width: int, height: int) -> Grid:
        """
        Get a subset of the grid
        """

        new_grid = Grid(width, height)
        for j in range(height):
            for i in range(width):
                x, y = topX + i, topY + j
                if 0 <= x < self.width and 0 <= y < self.height:
                    new_grid.set(i, j, self.get(x, y))
                else:
                    # Fill out-of-bounds cells with `None` or a default object
                    new_grid.set(i, j, None)

        return new_grid

    def encode(self, vis_mask: np.ndarray | None = None) -> np.ndarray:
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, 3), dtype="uint8")

        for i in range(self.width):
            for j in range(self.height):
                assert vis_mask is not None
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = OBJECT_TO_IDX["empty"]
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0

                    else:
                        array[i, j, :] = v.encode()

        return array

    @staticmethod
    def decode(array: np.ndarray) -> tuple[Grid, np.ndarray]:
        """
        Decode an array grid encoding back into a grid
        """

        width, height, channels = array.shape
        assert channels == 3

        vis_mask = np.ones(shape=(width, height), dtype=bool)

        new_grid = Grid(width, height)

        for i in range(width):
            for j in range(height):
                type_idx, color_idx, state = array[i, j]
                v = WorldObj.decode(type_idx, color_idx, state)
                new_grid.set(i, j, v)
                vis_mask[i, j] = type_idx != OBJECT_TO_IDX["unseen"]


        return new_grid, vis_mask

    def process_vis(self, agent_pos: tuple[int, int]) -> np.ndarray:

        """
        Compute the visibility mask of the grid
        """
        mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        mask[agent_pos[0], agent_pos[1]] = True

        for j in reversed(range(0, self.height)):
            for i in range(0, self.width - 1):
                if not mask[i, j]:
                    continue

                cell = self.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i + 1, j] = True
                if j > 0:
                    mask[i + 1, j - 1] = True
                    mask[i, j - 1] = True

            for i in reversed(range(1, self.width)):
                if not mask[i, j]:
                    continue

                cell = self.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i - 1, j] = True
                if j > 0:
                    mask[i - 1, j - 1] = True
                    mask[i, j - 1] = True

        for j in range(0, self.height):
            for i in range(0, self.width):
                if not mask[i, j]:
                    self.set(i, j, None)

        return mask
