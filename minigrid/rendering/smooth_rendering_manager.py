from typing import Any, Callable
import math
import numpy as np
import pygame
from abc import ABC, abstractmethod

from minigrid.core.constants import TILE_PIXELS
from minigrid.core.world_object import WorldObj
from minigrid.rendering.rendering_manager import BaseRenderingManager

from minigrid.utils.rendering import (
    downsample,
    highlight_img,
)

from minigrid.rendering.obj_renderers import (
    ObjRenderer,
    AgentRenderer,
)

from minigrid.rendering.pretty_obj_renderers import (
    PrettyFloorRenderer,
    PrettyAgentRenderer,
    PrettyWallRenderer,
    PrettyGoalRenderer,
    PrettyLavaRenderer,
    PrettyKeyRenderer,
)

class SmoothRenderingManager(BaseRenderingManager):

    def __init__(self, env):
        super().__init__(env)
        self.agent_renderer = PrettyAgentRenderer()
        self.renderer_map = {
            'goal': PrettyGoalRenderer,
            'floor': PrettyFloorRenderer,
            'lava': PrettyLavaRenderer,
            'wall': PrettyWallRenderer,
            'door': DoorRenderer,
            'key': PrettyKeyRenderer,
            'ball': BallRenderer,
            'box': BoxRenderer
        }

        # Assume 10 FPS for rendering
        self.render_fps = 10

        # Number of frames per animation cycle
        self.frames_per_action = 5  # Example: walking takes 5 frames

        # Frame duration in terms of simulation steps
        self.step_duration = 1 / self.render_fps  # Time per frame (0.1 seconds at 10 FPS)

    def pre_render_setup(self):
        """
        Perform any setup needed before rendering
        """

        # loop through all objects in the grid
        for j in range(0, self.env.height):
            for i in range(0, self.env.width):
                cell = self.env.grid.get(i, j)

                if cell.renderer is None:
                    proximity_grid = []

                    # Check neighbors of the cell in 3x3 pattern
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            
                            neighbor_x = i + dx
                            neighbor_y = j + dy

                            if 0 <= neighbor_x < self.env.width and 0 <= neighbor_y < self.env.height:
                                neighbor_cell = self.env.grid.get(neighbor_x, neighbor_y)
                            else:
                                neighbor_cell = None

                            proximity_grid.append(neighbor_cell)

                    # Assign renderers and set render states for static objects





    def render(self):
        """
        """

        frame = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        for current_frame in range(self.frames_per_action):

            

            # Render static objects (i.e. the background: walls, floor, etc.)
            render_static_objects()

            # Render dynamic objects (i.e. boxes, balls, keys, etc.)
            render_dynamic_objects(current_frame)

            # Render the agent
            render_agent()


        

    def get_highlight_mask(self):
        """
        returns a mask of which cells to highlight
        """
        
        _, vis_mask = self.env.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.env.dir_vec
        r_vec = self.env.right_vec
        top_left = (
            self.env.agent_pos
            + f_vec * (self.env.agent_view_size - 1)
            - r_vec * (self.env.agent_view_size // 2)
        )

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.env.width, self.env.height), dtype=bool)

        # For each cell in the visibility mask
        for vis_j in range(0, self.env.agent_view_size):
            for vis_i in range(0, self.env.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.env.width:
                    continue
                if abs_j < 0 or abs_j >= self.env.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        return highlight_mask


    # def render_tile(
    #     self,
    #     obj: WorldObj | None,
    #     highlight: bool = False,
    #     tile_size: int = 16,
    # ) -> np.ndarray:
    #     """
    #     Render a tile and cache the result
    #     """
    #     # Hash map lookup key for the cache
    #     key: 'tuple[Any, ...]' = (highlight, tile_size)
    #     key = obj.encode() + obj.renderer.rendering_encoding() + key if obj and obj.renderer else key

    #     if key in self.tile_cache:
    #         return self.tile_cache[key]

    #     img = np.zeros(
    #         shape=(tile_size, tile_size, 3), dtype=np.uint8
    #     )

    #     PrettyFloorRenderer().render(img, (0,))

    #     # Overlay the object
    #     if obj and obj.renderer:
    #         # render object
    #         obj.renderer.render(img, obj.encode())

    #     # # Highlight the cell with a yellow orange glow
    #     # if highlight:
    #     #     img = np.clip(img * 1.2 + np.array([30, 20, 0]), 0, 255).astype(np.uint8)
    #     # # Darken the cell
    #     # elif obj == None or (obj and obj.type not in ['lava']):
    #     #     img = img * 0.7

    #     # Cache the rendered tile
    #     self.tile_cache[key] = img

    #     return img

    # def render_grid(
    #     self,
    #     tile_size: int,
    #     agent_pos: 'tuple[int, int]',
    #     agent_dir: int | None = None,
    #     highlight_mask: np.ndarray | None = None,
    # ) -> np.ndarray:
    #     """
    #     Render this grid at a given scale
    #     :param r: target renderer object
    #     :param tile_size: tile size in pixels
    #     """

    #     if highlight_mask is None:
    #         highlight_mask = np.zeros(shape=(self.env.width, self.env.height), dtype=bool)

    #     # Compute the total grid size
    #     width_px = self.env.width * tile_size
    #     height_px = self.env.height * tile_size

    #     img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

    #     # Dynamic objects
    #     dynamic_objects = []

    #     # Render the grid
    #     for j in range(0, self.env.height):
    #         for i in range(0, self.env.width):
    #             obj = self.env.grid.get(i, j)

    #             #assign renderers and set render states for static objects
    #             if obj:
    #                 if obj.renderer is None:
    #                     obj.renderer = self.renderer_map[obj.type]()
    #                 # set the render state of the wall
    #                 if obj.type == 'wall':
    #                     obj.renderer.set_render_state(self.get_proximity_grid('wall', (i, j), self.env.grid))

    #             assert highlight_mask is not None
    #             highlight = bool(highlight_mask[i, j])
    #             key: 'tuple[Any, ...]' = (highlight, tile_size)
    #             key = obj.encode() + obj.renderer.rendering_encoding() + key if obj and obj.renderer else key

    #             if key in self.tile_cache:
    #                 return self.tile_cache[key]

    #             tile_img = np.zeros(
    #                 shape=(tile_size, tile_size, 3), dtype=np.uint8
    #             )

    #             # Overlay the floor if there is no object
    #             if obj == None:
    #                 PrettyFloorRenderer().render(tile_img, (0,))
                
    #             # Overlay the object
    #             elif obj.renderer:
    #                 # check if the object has transparency
    #                 if obj.renderer.is_transparent:
    #                     PrettyFloorRenderer().render(tile_img, (0,))
    #                 # check if the object is dynamic and render it later
    #                 if obj.renderer.is_dynamic == False:
    #                     obj.renderer.render(tile_img, obj.encode())
    #                 else:
    #                     dynamic_objects.append((obj, (i, j)))

    #             # Highlight the cell with a yellow orange glow
    #             if highlight:
    #                 tile_img = np.clip(tile_img * 1.2 + np.array([30, 20, 0]), 0, 255).astype(np.uint8)
    #             # Darken the cell
    #             elif obj == None or (obj and obj.type not in ['lava']):
    #                 tile_img = tile_img * 0.7

    #             # Cache the rendered tile
    #             self.tile_cache[key] = tile_img

    #             ymin = j * tile_size
    #             ymax = (j + 1) * tile_size
    #             xmin = i * tile_size
    #             xmax = (i + 1) * tile_size
    #             img[ymin:ymax, xmin:xmax, :] = tile_img

    #     # Overlay the agent


    #     return img
    






    def get_proximity_grid(self, type, cur_pos, grid, proximity=1) -> tuple:
        """
        Get a proximity grid of a type of object around a position in the grid.
        This works for any proximity value, creating a grid of size (2*proximity + 1) x (2*proximity + 1).
        """

        x, y = cur_pos  # cur_pos is (i, j)

        # Create a proximity grid of size (2*proximity + 1) x (2*proximity + 1)
        proximity_grid_size = 2 * proximity + 1
        proximity_grid = np.zeros((proximity_grid_size, proximity_grid_size), dtype=int)

        # Iterate over the proximity grid (outer loop on rows j, inner loop on columns i)
        for j in range(-proximity, proximity + 1):  # Outer loop (rows)
            for i in range(-proximity, proximity + 1):  # Inner loop (columns)
                # Check bounds of the grid
                if 0 <= x + i < grid.width and 0 <= y + j < grid.height:
                    cell = grid.get(x + i, y + j)
                    if cell is not None and cell.type == type:
                        proximity_grid[j + proximity, i + proximity] = 1  # Maintain row/column order
                    else:
                        proximity_grid[j + proximity, i + proximity] = 0
                else:
                    proximity_grid[j + proximity, i + proximity] = 0

        return tuple(proximity_grid.flatten().tolist())