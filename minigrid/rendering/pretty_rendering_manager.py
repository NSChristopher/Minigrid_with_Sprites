import time
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX
import pyglet
from pyglet import shapes
from pyglet.gl import glEnable, GL_BLEND, glBlendFunc, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA

from minigrid.rendering.rendering_manager import BaseRenderingManager


from minigrid.rendering.pretty_obj_renderers import (
    PrettyDoorRenderer,
    PrettyFloorRenderer,
    PrettyAgentRenderer,
    PrettyShadowRenderer,
    PrettyWallRenderer,
    PrettyGoalRenderer,
    PrettyLavaRenderer,
    PrettyBonesRenderer,
    PrettyKeyRenderer,
    PrettyBallRenderer,
    PrettyBoxRenderer,
    PrettyWebRenderer,
)

class PrettyRenderingManager(BaseRenderingManager):

    def __init__(self, env):
        super().__init__(env)
        # Map of object type to renderer classes
        self.static_renderer_map = {
            'floor': PrettyFloorRenderer,
            'wall': PrettyWallRenderer,
            'lava': PrettyLavaRenderer,
        }
        self.dynamic_renderer_map = {
            'door': PrettyDoorRenderer,
            'goal': PrettyGoalRenderer,
            'key': PrettyKeyRenderer,
            'ball': PrettyBallRenderer,
            'box': PrettyBoxRenderer,
        }

        # memory for renderers with objects
        self.object_renderers = {}

        # memory for renderers without objects
        self.floor_renderers = {}
        self.debri_renderers = {}
        self.shadow_renderers = {}
        self.spider_web_renderers = {}
        self.highlight_renderers = {}
        self.ambiance_renderers = {}

        # mask of debri and damage positions
        self.debri_positions = None
        self.damage_positions = None

        # pyglet rendering groups
        self.static_layer = pyglet.graphics.Group(order=1)
        self.decorative_layer = pyglet.graphics.Group(order=2)
        self.dynamic_layer = pyglet.graphics.Group(order=3)
        self.highlight_layer = pyglet.graphics.Group(order=4)
        self.status_bar_layer = pyglet.graphics.Group(order=99)
        self.ambient_layer = pyglet.graphics.Group(order=10)

        self.agent_renderer = None

        # Assume 10 FPS for rendering
        self.render_fps = self.env.metadata['render_fps']

        # Time per frame in seconds
        self.time_per_frame = 1.0 / self.render_fps

        # Number of frames per animation cycle
        self.frames_per_action = 5  # Example: walking takes 5 frames

        self.window = None
        self.visible_window = (self.env.render_mode == 'human')
        self.scale_factor = 1
        self.tile_size = 24
        self.tile_pixels = int(self.tile_size * self.scale_factor)
        self.pixel_height = self.env.height * self.tile_pixels
        self.pixel_width = self.env.width * self.tile_pixels
        self.batch = None

        self.last_step = -2

        # frames from most recent render
        self.frames = []


    def render(self):
        """
        Render the current grid state
        """
        # Setup the rendering window
        if self.window is None:
            self.setup_pyglet()

        # Setup renderers
        self.last_step += 1 # sync with the env step count
        if self.env.step_count != self.last_step:
            self.setup_new_batches()
            self.setup_obj_renderers()
            self.setup_decorative_renderers()
            self.setup_status_bar()
        self.last_step = self.env.step_count


        # Update the renderers
        self.update_obj_renderers()

        # Update the status bar
        self.update_status_bar()

        # Update agent animation
        assert self.agent_renderer is not None
        self.agent_renderer.update_agent_animation(
            agent_dir=self.env.agent_dir, 
            agent_action=self.env.action, 
            carrying=self.env.carrying
        )

        # Get agent's position in OpenGL coordinates
        assert self.agent_renderer is not None
        old_px_x, old_px_y = self.agent_renderer.curr_pos
        new_px_x, new_px_y = self._calculate_opengl_coordinates(self.env.agent_pos[0], 
                                                                self.env.agent_pos[1], 
                                                                center=True)
        self.frames = []
        for i in range(5):
            
            # Interpolate the agent's position
            t = (i + 1) / self.frames_per_action
            interp_x = old_px_x + t*(new_px_x - old_px_x)
            interp_y = old_px_y + t*(new_px_y - old_px_y)

            # Update the agent's position
            self.agent_renderer.update_agent_position(agent_pos=(interp_x, interp_y))

            # Clear the window
            assert self.window is not None
            self.window.clear()

            # Draw the batch
            assert self.batch is not None
            self.batch.draw()
            pyglet.clock.tick()
            self.window.flip()

            if self.env.render_mode == 'rgb_array':
                # capture the frame
                self.frames.append(self._get_frame())
            else:
                # sleep for a bit
                time.sleep(self.time_per_frame)


    def get_frame(self) -> list:
        """
        Return the frames from the most recent render call
        """
        return self.frames

    def setup_pyglet(self):
        """
        Perform any setup needed to start rendering with pyglet
        """
        # Create the window
        self.window = pyglet.window.Window(
            width=self.pixel_width,
            height=self.pixel_height,
            visible=self.visible_window,
        )

    def setup_new_batches(self):
        """
        Create a new batch to render the objects
        """

        self.batch = pyglet.graphics.Batch()
        self.highlight_batch = pyglet.graphics.Batch()


    def setup_obj_renderers(self):
        """
        Set up object renderers
        """

        # Clear floor renderers
        self.floor_renderers = {}

        # Create the agent renderer
        # opengl coordinates
        x, y = self._calculate_opengl_coordinates(self.env.agent_pos[0], self.env.agent_pos[1], center=True)
        self.agent_renderer = PrettyAgentRenderer(x=x, y=y, batch=self.batch, scale_factor=self.scale_factor, group=self.dynamic_layer)

        # loop through all objects in the grid
        for j in range(self.env.height):
            for i in range(self.env.width):

                obj = self.env.grid.get(i, j)

                # opengl coordinates
                x, y = self._calculate_opengl_coordinates(i, j)


                if obj and obj.type != 'wall' or obj is None:
                    floor_renderer = PrettyFloorRenderer(obj=None, x=x, y=y, batch=self.batch, scale_factor=self.scale_factor, group=self.static_layer)
                    self.floor_renderers[(i, j)] = floor_renderer

                if obj:
                    if obj.renderer is None:

                        if obj.type in self.static_renderer_map:
                            # assign the object a renderer and set the group to the static layer
                            obj.renderer = self.static_renderer_map[obj.type](obj, x=x, y=y, batch=self.batch, scale_factor=self.scale_factor, group=self.static_layer)

                            # set the proximity encoding of the renderer if the renderer has the method
                            if hasattr(obj.renderer, 'set_proximity_encoding'):
                                proximity_encoding = self.env.grid.get_proximity_encoding(i, j)
                                obj.renderer.set_proximity_encoding(proximity_encoding)

                            self.object_renderers[(i, j)] = obj.renderer

                        elif obj.type in self.dynamic_renderer_map:
                            obj.renderer = self.dynamic_renderer_map[obj.type](obj, x=x, y=y, batch=self.batch, scale_factor=self.scale_factor, group=self.dynamic_layer)

                            self.object_renderers[(i, j)] = obj.renderer

    def setup_decorative_renderers(self):
        """
        Set up decorative renderers for debri damage and shadows
        """

        encoded_grid = self.env.grid.encode()
        self.debri_positions = np.zeros((self.env.width, self.env.height), dtype=int)
        self.damage_positions = np.zeros((self.env.width, self.env.height), dtype=int)
        self.debri_renderers = {}
        self.shadow_renderers = {}

        # objects:
        # 0: unseen, 1: empty, 2: wall, 3: floor, 4: door, 5: key, 6: ball, 7: box, 8: goal, 9: lava, 10: agent

        # filter out all objects except empty from the grid and create a mask
        empty_mask = encoded_grid[:, :, 0] == OBJECT_TO_IDX['empty']

        # create a noise matrix for debri positions
        self.debri_positions = self._create_noise_matrix(self.debri_positions, mask=empty_mask, thresholds=[0.58, 0.61])

        # create a noise matrix for damage positions
        self.damage_positions = self._create_noise_matrix(self.damage_positions, mask=empty_mask, thresholds=[0.57, 0.61])

        for j in range(self.env.height):
            for i in range(self.env.width):

                if self.debri_positions[i, j] >= 1:
                    x, y = self._calculate_opengl_coordinates(i, j)
                    self.debri_renderers[(i, j)] = PrettyBonesRenderer(x=x, y=y, batch=self.batch, scale_factor=self.scale_factor, group=self.decorative_layer)
                    self.debri_renderers[(i, j)].set_type(type=self.debri_positions[i, j])
                if self.damage_positions[i, j] >= 1:
                    x, y = self._calculate_opengl_coordinates(i, j)
                    if self.floor_renderers[(i, j)]:
                        self.floor_renderers[(i, j)].set_type(type=2)

                proximity_encoding = self.env.grid.get_proximity_encoding(i, j)

                if proximity_encoding[0,1] == OBJECT_TO_IDX['wall'] and proximity_encoding[1,1] != OBJECT_TO_IDX['wall']:
                    x, y = self._calculate_opengl_coordinates(i, j)
                    self.shadow_renderers[(i, j)] = PrettyShadowRenderer(x=x, y=y, batch=self.batch, scale_factor=self.scale_factor, group=self.decorative_layer)

    def update_obj_renderers(self):
        """
        Update the object renderers to reflect the current grid state
        """
        # get the highlight mask
        self.highlight_renderers = {}
        highlight_mask = self._get_highlight_mask()
        self.highlight_batch = pyglet.graphics.Batch()
        # loop through all objects in the grid
        for j in range(self.env.height):
            for i in range(self.env.width):

                obj = self.env.grid.get(i, j)

                # opengl coordinates
                x, y = self._calculate_opengl_coordinates(i, j)
                
                # remove the renderer if the object is None
                if obj is None:
                    if (i, j) in self.object_renderers:
                        self.object_renderers[(i, j)].delete_sprite()
                        del self.object_renderers[(i, j)]
                # update the renderer if the object is not None
                else:
                    # if the object has no renderer, create one and remove the old one
                    if obj.renderer is None or obj.renderer.sprite is None:
                        if obj.type in self.static_renderer_map:
                            # assign the object a renderer
                            obj.renderer = self.static_renderer_map[obj.type](obj, x=x, y=y, batch=self.batch, scale_factor=self.scale_factor, group=self.static_layer)
                            if (i, j) in self.object_renderers:
                                self.object_renderers[(i, j)].delete_sprite()
                                del self.object_renderers[(i, j)]
                            self.object_renderers[(i, j)] = obj.renderer

                        elif obj.type in self.dynamic_renderer_map:
                            obj.renderer = self.dynamic_renderer_map[obj.type](obj, x=x, y=y, batch=self.batch, scale_factor=self.scale_factor, group=self.dynamic_layer)
                            if (i, j) in self.object_renderers:
                                self.object_renderers[(i, j)].delete_sprite()
                                del self.object_renderers[(i, j)]
                            self.object_renderers[(i, j)] = obj.renderer
                    # if the object has a renderer, update it
                    else:
                        obj.renderer.update(x=x, y=y)

                # if cell is part of the highlight mask, highlight it
                if highlight_mask[i, j]:
                    # Create a Rectangle in the highlight_batch
                    rect = shapes.Rectangle(
                        x=x, 
                        y=y,
                        width=self.tile_pixels, 
                        height=self.tile_pixels,
                        # yellowish red like a flame
                        color=(255, 165, 100),
                        batch=self.batch,
                        group=self.highlight_layer
                    ) # 0..255 for transparency
                    rect.opacity = 25

                    self.highlight_renderers[(i, j)] = rect



    def setup_status_bar(self):
        """
        Set up the status bar
        """

        self.status_label = pyglet.text.Label(
            text="Step: 0, Action: None",
            x=10,
            y=self.pixel_height - self.tile_pixels * .9,
            anchor_x='left',
            anchor_y='bottom',
            font_name='Press Start 2P',  # or a monospaced/pixel-like font
            font_size=16,
            color=(255, 255, 255, 255),
            batch=self.batch,
            group=self.status_bar_layer,
        )

    def update_status_bar(self):
        """
        Update the label text to reflect the current step and reward.
        """
        # If you have a mapping from action int -> action name, you can get a readable string.
        # For a quick demo, we just show the action as an integer.
        step = self.env.step_count
        reward = round(self.env.current_reward, 2)
        self.status_label.text = f"Step: {step}, Reward: {reward}"

    def close(self):
        """
        Close the rendering window
        """
        if self.window:
            self.window.close()
            self.window = None

    def _create_noise_matrix(self, positions_matrix, mask=None, thresholds=[0.58, 0.61]):
        """
        Create a noise matrix with defined thresholds and values between 0 and 2
        """
        # create noise grid with values between 0 and 1
        noise = np.random.rand(self.env.width, self.env.height)

        # smooth the noise
        smoothed = noise.copy()
        for _ in range(10):
            up = np.roll(smoothed, -1, axis=0)
            down = np.roll(smoothed, 1, axis=0)
            left = np.roll(smoothed, -1, axis=1)
            right = np.roll(smoothed, 1, axis=1)
            smoothed = (up + down + left + right + smoothed) / 4.99

        # Threshold into {0, 1, 2}
        positions_matrix[smoothed < thresholds[0]] = 0
        positions_matrix[smoothed >= thresholds[0]] = 1
        positions_matrix[smoothed >= thresholds[1]] = 2

        # mask to only keep the empty cells
        if mask is not None:
            positions_matrix[~mask] = 0

        return positions_matrix
    
    def _get_frame(self):
        """
        Capture the current color buffer as a NumPy RGB array.
        """
        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        image_data = buffer.get_image_data()
        width = image_data.width
        height = image_data.height

        # "RGB", 3 bytes per pixel
        raw_bytes = image_data.get_data('RGB', width * 3)
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        arr = arr.reshape((height, width, 3))
        arr = np.flipud(arr)
        return arr

    def _calculate_opengl_coordinates(self, i, j, center=False):
        """
        Calculate the opengl coordinates of the bottom-left corner of the cell (i, j),
        or the center of the cell if center=True
        """
        x = i * self.tile_pixels
        y = self.pixel_height - (j + 1) * self.tile_pixels

        if center:
            x += self.tile_pixels // 2
            y += self.tile_pixels // 2

        return x, y

    def _get_highlight_mask(self):
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