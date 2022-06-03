import numpy as np
def compute_light_direction(envmap_h, envmap_w):
    """
    return: normalized direction of h*w light sources
    """
    lat_step_size = np.pi / (envmap_h + 2)
    lng_step_size = 2 * np.pi / (envmap_w + 2)
    # Try to exclude the problematic polar points
    lats = np.linspace(
        np.pi / 2 - lat_step_size, -np.pi / 2 + lat_step_size, envmap_h)
    lngs = np.linspace(
        np.pi - lng_step_size, -np.pi + lng_step_size, envmap_w)
    lngs, lats = np.meshgrid(lngs, lats)
    lngs = lngs.reshape(-1)
    lats = lats.reshape(-1)

    r = 1  # unit sphere
    # Compute x, y and z
    z = r * np.sin(lats)
    x = r * np.cos(lats) * np.cos(lngs)
    y = r * np.cos(lats) * np.sin(lngs)

    # Assemble and return
    # xyz = np.stack((x, y, z), axis=1)
    xyz = np.stack((x, z, y), axis=1)

    return xyz

# print(compute_light_direction(10, 20))