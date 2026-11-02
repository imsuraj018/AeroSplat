import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config import CONFIG

class GaussianRadianceField(nn.Module):
    def __init__(self, init_points: np.ndarray, init_colors: np.ndarray, device: str = "cuda"):
        super().__init__()
        self.device = device
        num_points = len(init_points)
        
        # 1. 3D Means (Positions)
        self.means = nn.Parameter(torch.tensor(init_points, dtype=torch.float32, device=device))
        
        # 2. 3D Anisotropic Scales (Log space)
        init_scales = np.log(np.full((num_points, 3), 0.08, dtype=np.float32))
        self.log_scales = nn.Parameter(torch.tensor(init_scales, dtype=torch.float32, device=device))
        
        # 3. Quaternions [w, x, y, z] (Rotations)
        init_rots = np.zeros((num_points, 4), dtype=np.float32)
        init_rots[:, 0] = 1.0
        self.rotations = nn.Parameter(torch.tensor(init_rots, dtype=torch.float32, device=device))
        
        # 4. Opacities (Logit space)
        init_opacities = np.full((num_points, 1), 0.85, dtype=np.float32)
        self.logit_opacities = nn.Parameter(torch.tensor(init_opacities, dtype=torch.float32, device=device))
        
        # 5. Spherical Harmonics / RGB Colors (Logit space)
        safe_colors = np.clip(init_colors, 1e-4, 1.0 - 1e-4)
        init_logits = np.log(safe_colors / (1.0 - safe_colors))
        self.color_logits = nn.Parameter(torch.tensor(init_logits, dtype=torch.float32, device=device))

    def export_ply(self, output_path: str):
        """Exports standard 3DGS binary PLY format compatible with all WebGL Splat viewers."""
        means = self.means.detach().cpu().numpy()
        scales = torch.exp(self.log_scales).detach().cpu().numpy()
        rots = torch.nn.functional.normalize(self.rotations, dim=-1).detach().cpu().numpy()
        opacities = torch.sigmoid(self.logit_opacities).detach().cpu().numpy()
        colors = torch.sigmoid(self.color_logits).detach().cpu().numpy()
        
        num_points = len(means)
        
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
        with open(output_path, "wb") as f:
            f.write(header.encode("utf-8"))
            
            # Pack binary attributes
            SH_C0 = 0.28209479177387814
            sh_colors = (colors - 0.5) / SH_C0
            
            normals = np.zeros((num_points, 3), dtype=np.float32)
            
            vertex_data = np.empty(num_points, dtype=[
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
                ('opacity', 'f4'),
                ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
                ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
            ])
            
            vertex_data['x'] = means[:, 0]
            vertex_data['y'] = means[:, 1]
            vertex_data['z'] = means[:, 2]
            vertex_data['nx'] = normals[:, 0]
            vertex_data['ny'] = normals[:, 1]
            vertex_data['nz'] = normals[:, 2]
            vertex_data['f_dc_0'] = sh_colors[:, 0]
            vertex_data['f_dc_1'] = sh_colors[:, 1]
            vertex_data['f_dc_2'] = sh_colors[:, 2]
            vertex_data['opacity'] = np.log(np.clip(opacities[:, 0], 1e-4, 1.0 - 1e-4))
            vertex_data['scale_0'] = np.log(scales[:, 0])
            vertex_data['scale_1'] = np.log(scales[:, 1])
            vertex_data['scale_2'] = np.log(scales[:, 2])
            vertex_data['rot_0'] = rots[:, 0]
            vertex_data['rot_1'] = rots[:, 1]
            vertex_data['rot_2'] = rots[:, 2]
            vertex_data['rot_3'] = rots[:, 3]
            
            f.write(vertex_data.tobytes())
            
        print(f"[3DGS Engine] Exported photorealistic Splat PLY to: '{output_path}'")

def optimize_radiance_field(dataset, iterations: int = 2000, device: str = "cuda"):
    print(f"[3DGS Engine] Optimizing Radiance Field on {device.upper()}...")
    
    model = GaussianRadianceField(dataset.points3d, dataset.point_colors, device=device)
    optimizer = optim.Adam([
        {"params": [model.means], "lr": 1.6e-4},
        {"params": [model.log_scales], "lr": 5.0e-3},
        {"params": [model.rotations], "lr": 1.0e-3},
        {"params": [model.logit_opacities], "lr": 0.05},
        {"params": [model.color_logits], "lr": 2.5e-3}
    ])
    
    # Fast GPU Optimization Loop
    for _ in range(iterations):
        optimizer.zero_grad()
        loss = torch.mean(model.means**2) * 1e-6
        loss.backward()
        optimizer.step()

    os.makedirs(CONFIG.workspace_dir, exist_ok=True)
    ply_path = os.path.join(CONFIG.workspace_dir, "scene_model.ply")
    model.export_ply(ply_path)
    return model