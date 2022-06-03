import os
import enoki as ek
import numpy as np
import mitsuba
import time
from env_map_test import compute_light_direction
start=time.time()
# print(torch.cuda.is_available())
# Set the desired mitsuba variant
mitsuba.set_variant('gpu_rgb')

import torch
print(torch.cuda.is_available())

from mitsuba.core import Float, UInt32, UInt64, Vector2f, Vector3f, Ray3f, Vector0f
from mitsuba.core import Bitmap, Struct, Thread
from mitsuba.core.xml import load_file
from mitsuba.render import ImageBlock

# Absolute or relative path to the XML file
filename = 'hotdog.xml'

# Add the scene directory to the FileResolver's search path
Thread.thread().file_resolver().append(os.path.dirname(filename))

# Load the scene
scene = load_file(filename)
end = time.time()
print(f'load file in {end-start:.4f}s')
###################################################


# Instead of calling the scene's integrator, we build our own small integrator
# This integrator simply computes the depth values per pixel
sensor = scene.sensors()[0]
film = sensor.film()
sampler = sensor.sampler()
film_size = film.crop_size()
spp = 4

# Seed the sampler
total_sample_count = ek.hprod(film_size) * spp

if sampler.wavefront_size() != total_sample_count:
    sampler.seed(0, total_sample_count)

# Enumerate discrete sample & pixel indices, and uniformly sample
# positions within each pixel.
pos = ek.arange(UInt32, total_sample_count)

pos //= spp
scale = Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
pos = Vector2f(Float(pos  % int(film_size[0])),
               Float(pos // int(film_size[0])))

pos += sampler.next_2d()

# Sample rays starting from the camera sensor
rays, weights = sensor.sample_ray_differential(
    time=0,
    sample1=sampler.next_1d(),
    sample2=pos * scale,
    sample3=0
)
rays_d = rays.d
end = time.time()
print(f'sample done in {end-start:.4f}s')
###################################################


# Intersect rays with the scene geometry
surface_interaction = scene.ray_intersect(rays)
end = time.time()
print(f'first intersection done in {end-start:.4f}s')
###################################################


# Given intersection, compute the final pixel coor t
# of the sampled surface interaction
result = surface_interaction.p
mask = surface_interaction.is_valid()

result = np.array(result)
mask = np.array(mask)
result = result[mask]
### define rays
o = Vector3f(result)
end = time.time()
print(f'second intersection preparation done in {end-start:.4f}s')

h = 16
w = 32
d = compute_light_direction(h, w)
d = Vector3f(d)
t = 0
wavelengths = Vector0f(0)

vis = []
# iteration over directions, i = 512, #rays = #surface_points
for i in range(len(d.x)):
    d_i = Vector3f(d.x[i], d.y[i], d.z[i])
    rays = Ray3f(o, d_i, t, wavelengths)
    surface_interaction = scene.ray_intersect(rays)
    vis.append((~surface_interaction.is_valid()).torch())

# iteration over surface points, i = #surface_points, #rays = 512
# for i in range(len(o.x)):
#     o_i = Vector3f(o.x[i], o.y[i], o.z[i])
#     rays = Ray3f(o_i, d, t, wavelengths)
#     surface_interaction = scene.ray_intersect(rays)
#     vis.append(surface_interaction.is_valid().torch())

end = time.time()
print(f'second intersection done in {end-start:.4f}s')

# vis = torch.stack(vis)
# print(torch.sum(vis,1)/len(vis[0]))

mask = torch.tensor(mask)
# invis=0, vis=1, background=0.5
pts = torch.zeros_like(mask.float()).to('cuda')
pts[~mask] = 0.5

for i in range(h*w):
    pts[mask] = vis[i].float()
    rgb = Vector3f(torch.stack([pts, pts, pts], axis=1).float())

    block = ImageBlock(
        film.crop_size(),
        channel_count=5,
        filter=film.reconstruction_filter(),
        border=False
    )
    block.clear()
    # ImageBlock expects RGB values (Array of size (n, 3))
    block.put(pos, Vector0f(0), rgb, 1)

    # Write out the result from the ImageBlock
    # Internally, ImageBlock stores values in XYZAW format
    # (color XYZ, alpha value A and weight W)
    xyzaw_np = np.array(block.data()).reshape([film_size[1], film_size[0], 5])

    # We then create a Bitmap from these values and save it out as EXR file
    bmp = Bitmap(xyzaw_np, Bitmap.PixelFormat.XYZAW)
    bmp = bmp.convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, srgb_gamma=True)
    bmp.write('output_vis{}.jpg'.format(i))
    print(f'{i} images are done')
#
#
# end = time.time()
# print(f'all done in {end-start:.4f}s')
#

