from typing import Any, Callable
import math
import numpy as np
import pygame
from abc import ABC, abstractmethod

from minigrid.core.constants import TILE_PIXELS
from minigrid.core.world_object import WorldObj

from minigrid.utils.rendering import (
    downsample,
    highlight_img,
)

from minigrid.renderers.obj_renderer import (
    AgentRenderer,
)


class BaseRenderingManager(ABC):

    # Static cache of pre-renderer tiles
    tile_cache: 'dict[tuple[Any, ...], Any]' = {}

    def __init__(self, env):
        self.env = env
        self.action = 0

    @classmethod
    def render_tile(
        cls,
        obj: WorldObj | None,
        agent_dir: int | None = None,
        highlight: bool = False,
        tile_size: int = TILE_PIXELS,
        subdivs: int = 3,
    ) -> np.ndarray:
        """
        Render a tile and cache the result
        """

        raise NotImplementedError

    def render_grid(
        self,
        tile_size: int,
        agent_pos: 'tuple[int, int]',
        agent_dir: int | None = None,
        highlight_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """
            
        raise NotImplementedError

       

    def get_pov_render(self, tile_size):
        """
        Render an agent's POV observation for visualization
        """
        grid, vis_mask = self.env.gen_obs_grid()

        # Render the whole grid
        img = self.render_grid(
            tile_size,
            agent_pos=(self.env.agent_view_size // 2, self.env.agent_view_size - 1),
            agent_dir=3,
            highlight_mask=vis_mask,
        )

        return img

    def get_full_render(self, highlight, tile_size):
        """
        Render a non-paratial observation for visualization
        """
        # Compute which cells are visible to the agent
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

        # Render the whole grid
        img = self.render_grid(
            tile_size,
            self.env.agent_pos,
            self.env.agent_dir,
            highlight_mask=highlight_mask if highlight else None,
        )

        return img
    
    def get_frame(
        self,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        """Returns an RGB image corresponding to the whole environment or the agent's point of view.

        Args:

            highlight (bool): If true, the agent's field of view or point of view is highlighted with a lighter gray color.
            tile_size (int): How many pixels will form a tile from the NxM grid.
            agent_pov (bool): If true, the rendered frame will only contain the point of view of the agent.

        Returns:

            frame (np.ndarray): A frame of type numpy.ndarray with shape (x, y, 3) representing RGB values for the x-by-y pixel image.

        """

        if agent_pov:
            return self.get_pov_render(tile_size)
        else:
            return self.get_full_render(highlight, tile_size)

    def render(self, action=None):

        if action:
            self.action = action

        img = self.get_frame(self.env.highlight, self.env.tile_size, self.env.agent_pov)

        if self.env.render_mode == "human":
            img = np.transpose(img, axes=(1, 0, 2))
            if self.env.render_size is None:
                self.env.render_size = img.shape[:2]
            if self.env.window is None:
                pygame.init()
                pygame.display.init()
                self.env.window = pygame.display.set_mode(
                    (self.env.screen_size, self.env.screen_size)
                )
                pygame.display.set_caption("minigrid")
            if self.env.clock is None:
                self.env.clock = pygame.time.Clock()
            surf = pygame.surfarray.make_surface(img)

            # Create background with mission description
            offset = surf.get_size()[0] * 0.1
            # offset = 32 if self.agent_pov else 64
            bg = pygame.Surface(
                (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
            )
            bg.convert()
            bg.fill((255, 255, 255))
            bg.blit(surf, (offset / 2, 0))

            bg = pygame.transform.smoothscale(bg, (self.env.screen_size, self.env.screen_size))

            font_size = 22
            text = self.env.mission
            font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
            text_rect = font.get_rect(text, size=font_size)
            text_rect.center = bg.get_rect().center
            text_rect.y = bg.get_height() - font_size * 1.5
            font.render_to(bg, text_rect, text, size=font_size)

            self.env.window.blit(bg, (0, 0))
            pygame.event.pump()
            self.env.clock.tick(self.env.metadata["render_fps"])
            pygame.display.flip()

        elif self.env.render_mode == "rgb_array":
            return img


class RenderingManager(BaseRenderingManager):

    def __init__(self, env):
        super().__init__(env)

    @classmethod
    def render_tile(
        cls,
        obj: WorldObj | None,
        agent_dir: int | None = None,
        highlight: bool = False,
        tile_size: int = TILE_PIXELS,
        subdivs: int = 3,
    ) -> np.ndarray:
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key: 'tuple[Any, ...]' = (agent_dir, highlight, tile_size)
        key = obj.renderer.rendering_encoding() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
        )

        # Overlay the agent
        if agent_dir is not None:
            AgentRenderer().render(img, agent_dir)

        # Overlay the object
        if obj is not None:
            # render object
            obj.render(img)

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render_grid(
        self,
        tile_size: int,
        agent_pos: 'tuple[int, int]',
        agent_dir: int | None = None,
        highlight_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.env.width, self.env.height), dtype=bool)

        # Compute the total grid size
        width_px = self.env.width * tile_size
        height_px = self.env.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.env.height):
            for i in range(0, self.env.width):
                cell = self.env.grid.get(i, j)

                agent_here = np.array_equal(agent_pos, (i, j))
                assert highlight_mask is not None
                tile_img = self.render_tile(
                    cell,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=bool(highlight_mask[i, j]),
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img
    
from minigrid.renderers.pretty_obj_renderers import (
    PrettyFloorRenderer,
    PrettyAgentRenderer,
)

class PrettyRenderingManager(BaseRenderingManager):

    def __init__(self, env):
        super().__init__(env)
        self.agent_renderer = PrettyAgentRenderer()


    def render_tile(
        self,
        obj: WorldObj | None,
        agent_here: bool,
        agent_pos: 'tuple[int, int]',
        agent_dir: int | None = None,
        highlight: bool = False,
        tile_size: int = 16,
        subdivs: int = 1,
    ) -> np.ndarray:
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key: 'tuple[Any, ...]' = (highlight, tile_size)
        key = self.agent_renderer.rendering_encoding() + key if agent_here else key
        key = obj.renderer.rendering_encoding() + key if obj else key

        if key in self.tile_cache:
            return self.tile_cache[key]

        img = np.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
        )

        PrettyFloorRenderer().render(img)

        # Overlay the object
        if obj is not None:
            # render object
            obj.render(img)

        # Overlay the agent
        if agent_here and agent_dir is not None:
            self.agent_renderer.render(img)

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        self.tile_cache[key] = img

        return img

    def render_grid(
        self,
        tile_size: int,
        agent_pos: 'tuple[int, int]',
        agent_dir: int | None = None,
        highlight_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.env.width, self.env.height), dtype=bool)

        # Compute the total grid size
        width_px = self.env.width * tile_size
        height_px = self.env.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Updates agents sprite index
        self.agent_renderer.update_render_state(self.action, agent_dir)

        # Render the grid
        for j in range(0, self.env.height):
            for i in range(0, self.env.width):
                cell = self.env.grid.get(i, j)

                if cell is not None and cell.type == 'wall':
                    proximity_grid = self.get_proximity_grid('wall', (i, j), self.env.grid)
                    cell.renderer.set_render_state(proximity_grid)

                agent_here = np.array_equal(agent_pos, (i, j))
                assert highlight_mask is not None
                tile_img = self.render_tile(
                    cell,
                    agent_here=agent_here,
                    agent_pos=agent_pos,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=bool(highlight_mask[i, j]),
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img
    
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