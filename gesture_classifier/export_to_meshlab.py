import os
import torch
import numpy as np
from dataset import GestureDataset
from model import GestureClassifier

def save_ply(points, filename, colors=None):
    """
    Saves a point cloud to a PLY file so it can be opened in MeshLab.
    points: tensor or numpy array of shape (N, 3)
    colors: optional tensor or numpy array of shape (N, 3) with values in [0, 255]
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if points.ndim == 3:
        points = points[0]  # Take first item in batch
        
    if colors is not None:
        if isinstance(colors, torch.Tensor):
            colors = colors.detach().cpu().numpy()
        if colors.ndim == 3:
            colors = colors[0]

    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        for i, p in enumerate(points):
            if colors is not None:
                c = colors[i]
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
            else:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

def main():
    print("Loading dataset...")
    # Assuming the data is in the directory above or handled by GESTURE_DATA_ROOT
    dataset = GestureDataset(train=False)
    if len(dataset) == 0:
        print("Dataset is empty. Please check your data path.")
        return

    points, label = dataset[0]  # (64, 4)
    print(f"Loaded sample with label index: {label}")
    
    # Add batch dimension
    points_batch = points.unsqueeze(0)  # (1, 64, 4)
    
    # Load Model
    alpha = 1.0 
    model = GestureClassifier(num_classes=4, alpha=alpha)
    
    model_path = 'best_model_alpha_1.0.pth'
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"Loaded trained weights from {model_path}")
        except Exception as e:
            print(f"Could not load weights: {e}")
    else:
        print(f"Model file {model_path} not found. Using untrained weights.")
        
    model.eval()
    
    with torch.no_grad():
        # Input point cloud
        xyz_input = points_batch[:, :, :3]
        features = points_batch[:, :, 3:]
        
        # Intermediate output after first Set Abstraction layer
        xyz_sa1, features1 = model.sa1(xyz_input, features)
        
        # Intermediate output after second Set Abstraction layer
        xyz_sa2, features2 = model.sa2(xyz_sa1, features1)
    
    # Let's map doppler to colors for the input so it looks nice in MeshLab
    # Doppler is in features (points_batch[:, :, 3])
    doppler = points_batch[:, :, 3].numpy()[0]
    # Normalize doppler to 0-1 range
    d_min, d_max = doppler.min(), doppler.max()
    if d_max > d_min:
        doppler_norm = (doppler - d_min) / (d_max - d_min)
    else:
        doppler_norm = np.zeros_like(doppler)
        
    # Create an RGB colormap (e.g. blue to red)
    colors = np.zeros((len(doppler_norm), 3))
    colors[:, 0] = doppler_norm * 255          # Red
    colors[:, 2] = (1 - doppler_norm) * 255    # Blue
    
    # Save to PLY
    save_ply(xyz_input, "input_points.ply", colors=colors)
    save_ply(xyz_sa1, "sa1_sampled_points.ply")
    save_ply(xyz_sa2, "sa2_sampled_points.ply")
    
    print("\nSuccessfully saved PLY files:")
    print("  - input_points.ply (64 points, colored by Doppler velocity)")
    print("  - sa1_sampled_points.ply (32 points after first FPS layer)")
    print("  - sa2_sampled_points.ply (16 points after second FPS layer)")
    print("\nYou can now open these .ply files directly in MeshLab.")

if __name__ == '__main__':
    main()
