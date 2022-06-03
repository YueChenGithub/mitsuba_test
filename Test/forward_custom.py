import os
import enoki as ek
import numpy as np
import mitsuba
import time



start=time.time()
# Set the desired mitsuba variant
mitsuba.set_variant('gpu_rgb')
import torch
print(torch.cuda.is_available())
from mitsuba.core import Float, UInt32, UInt64, Vector2f, Vector3f
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
spp = 1

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


end = time.time()
print(f'sample done in {end-start:.4f}s')
###################################################


# Intersect rays with the scene geometry
surface_interaction = scene.ray_intersect(rays)

end = time.time()
print(f'compute intersection done in {end-start:.4f}s')
###################################################


# Given intersection, compute the final pixel values as the depth t
# of the sampled surface interaction
result = surface_interaction.t

# Set to zero if no intersection was found
result[~surface_interaction.is_valid()] = 0

block = ImageBlock(
    film.crop_size(),
    channel_count=5,
    filter=film.reconstruction_filter(),
    border=False
)
block.clear()
# ImageBlock expects RGB values (Array of size (n, 3))
block.put(pos, rays.wavelengths, Vector3f(result, result, result), 1)

# Write out the result from the ImageBlock
# Internally, ImageBlock stores values in XYZAW format
# (color XYZ, alpha value A and weight W)
xyzaw_np = np.array(block.data()).reshape([film_size[1], film_size[0], 5])

# We then create a Bitmap from these values and save it out as EXR file
bmp = Bitmap(xyzaw_np, Bitmap.PixelFormat.XYZAW)
bmp = bmp.convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, srgb_gamma=True)
bmp.write('output2.jpg')


end = time.time()
print(f'all done in {end-start:.4f}s')

