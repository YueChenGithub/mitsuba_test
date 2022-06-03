import mitsuba
mitsuba.set_variant('gpu_rgb')


import numpy as np
from mitsuba.core import Float, UInt32, UInt64, Vector2f, Vector3f, Ray3f, Vector0f
import torch
print(torch.cuda.is_available())

# vec_2d = [1,1]
# mit_2d = Vector2f(vec_2d)
# print(mit_2d)
# print(mit_2d.numpy())
# print(mit_2d.torch())
# print(Vector2f(mit_2d.numpy()))
# print(Vector2f(mit_2d.torch()))

# array_2d = [[1,2,3], [1,1,1]]
# mit_2d = Vector2f(array_2d)
# print(mit_2d)
# print(mit_2d.numpy())
# print(mit_2d.torch())
# print(Vector2f(mit_2d.numpy()))
# print(Vector2f(mit_2d.torch()))

# vec_3d = [1,1,1]
# mit_3d = Vector3f(vec_3d)
# print(mit_3d)
# print(mit_3d.numpy())
# print(mit_3d.torch())
# print(Vector3f(mit_3d.numpy()))
# print(Vector3f(mit_3d.torch()))


# array_3d = [[1,1],[2,2],[3,3]]
# mit_3d = Vector3f(array_3d)
# print(mit_3d)
# print(mit_3d.numpy())
# print(mit_3d.torch())
# print(Vector3f(mit_3d.numpy()))
# print(Vector3f(mit_3d.torch()))

d = torch.tensor([[1.,2,3],[1,2,3]]).to("cuda")
d_mit = Vector3f(d)
print(d_mit)

print(len(d_mit.x))

# list = [[x],[y],[z]]
# numpy/tensor = [[xyz1],[xyz2],[xyz3],...]  # type = float

