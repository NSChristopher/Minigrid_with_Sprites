from minigrid.rendering.obj_renderers import ObjRenderer
import pyglet

class DynamicObjRenderer(ObjRenderer):
    def __init__(self):
        super().__init__()
        