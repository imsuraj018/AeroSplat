from __future__ import annotations

import numpy as np


def save_initial_splat_ply(path: str, points: np.ndarray, colors: np.ndarray, radius: float = 0.01) -> None:
    """Write a viewer-friendly 3DGS-compatible DC-only PLY from COLMAP points.
    This is an initialization/fallback export; actual photometric optimization is delegated to the official 3DGS repo.
    """
    points = np.asarray(points, dtype=np.float32)
    colors = np.clip(np.asarray(colors, dtype=np.float32),0,1)
    n = len(points)
    sh_c0 = 0.28209479177387814
    data = np.empty(n, dtype=[('x','f4'),('y','f4'),('z','f4'),('nx','f4'),('ny','f4'),('nz','f4'),('f_dc_0','f4'),('f_dc_1','f4'),('f_dc_2','f4'),('opacity','f4'),('scale_0','f4'),('scale_1','f4'),('scale_2','f4'),('rot_0','f4'),('rot_1','f4'),('rot_2','f4'),('rot_3','f4')])
    data['x'],data['y'],data['z']=points.T
    data['nx']=data['ny']=data['nz']=0
    data['f_dc_0'],data['f_dc_1'],data['f_dc_2']=((colors-0.5)/sh_c0).T
    data['opacity']=0.0
    log_r=np.log(max(radius,1e-6)); data['scale_0']=data['scale_1']=data['scale_2']=log_r
    data['rot_0']=1.0; data['rot_1']=data['rot_2']=data['rot_3']=0.0
    header=f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\nproperty float opacity\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\nproperty float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\nend_header\n"
    with open(path,'wb') as f:
        f.write(header.encode('ascii')); f.write(data.tobytes())
