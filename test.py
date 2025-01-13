import pyglet
from pyglet import shapes

# Create a 1×1 white image in-memory for a sprite
# We'll tint it red via sprite.color
solid_white = pyglet.image.SolidColorImagePattern(color=(255, 255, 255, 255)).create_image(1, 1)

window = pyglet.window.Window(400, 300, "Sprites and Shapes Test")
batch = pyglet.graphics.Batch()

# 1) Create a sprite at (50, 50) tinted red, size = 64×64
sprite = pyglet.sprite.Sprite(solid_white, x=50, y=50, batch=batch)
sprite.scale_x = 64
sprite.scale_y = 64
sprite.color = (255, 0, 0)  # Red

# 2) Create a semi-transparent rectangle that overlaps the sprite
rect = shapes.Rectangle(x=80, y=70, width=100, height=80, color=(0, 255, 0), batch=batch)
rect.opacity = 128  # 50% alpha


@window.event
def on_draw():
    window.clear()
    batch.draw()


pyglet.app.run()
