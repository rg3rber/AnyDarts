import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from scipy.stats import gaussian_kde
import matplotlib.colors as colors

def generate_heatmap(label_dir, output_path, image_width=640, image_height=640, class_index=4, 
                     bandwidth=0.05, resolution=100, colormap='hot_r', alpha=0.7, overlay=False):
    """
    Generates a heatmap visualization of bounding box centers for a specific class from YOLO-formatted labels.

    Args:
        label_dir (str): Path to the directory containing the YOLO-formatted label text files.
        output_path (str): Path to save the generated heatmap image.
        image_width (int): Width of the image used for normalization. Defaults to 640.
        image_height (int): Height of the image used for normalization. Defaults to 640.
        class_index (int): The class index to visualize. Defaults to 4.
        bandwidth (float): Controls the smoothness of the heatmap.
        resolution (int): Resolution of the heatmap grid.
        colormap (str): Matplotlib colormap to use for the heatmap.
        alpha (float): Transparency level of the heatmap.
        overlay (bool): Whether to overlay a scatter plot of the actual points.
    """
    x_coords = []
    y_coords = []

    # Process each label file in the directory
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            label_file_path = os.path.join(label_dir, filename)
            try:
                with open(label_file_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue  # Skip empty lines
                        try:
                            class_id = int(parts[0])
                            if class_id == class_index:
                                # YOLO format: class_id, center_x_norm, center_y_norm, width_norm, height_norm
                                center_x_norm = float(parts[1])
                                center_y_norm = float(parts[2])
                                # Convert normalized coordinates to pixel coordinates
                                center_x = center_x_norm * image_width
                                center_y = center_y_norm * image_height
                                x_coords.append(center_x)
                                y_coords.append(center_y)
                        except ValueError as e:
                            print(f"Error processing line in {label_file_path}: {line.strip()}. Skipping. Error: {e}")
            except Exception as e:
                print(f"An error occurred while processing {label_file_path}: {e}")
                continue

    if not x_coords:
        print(f"No objects of class {class_index} found in the provided labels.")
        return

    print(f"Found {len(x_coords)} objects of class {class_index}")

    # Create the figure
    plt.figure(figsize=(10, 10))
    
    # Create a meshgrid for the heatmap
    x_min, x_max = 0, image_width
    y_min, y_max = 0, image_height
    
    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    # Calculate kernel density estimation
    try:
        values = np.vstack([x_coords, y_coords])
        kernel = gaussian_kde(values, bw_method=bandwidth)
        Z = np.reshape(kernel(positions), X.shape)
        
        # Plot the heatmap
        plt.imshow(Z, 
                   extent=[x_min, x_max, y_max, y_min],  # Invert y-axis for image coordinates
                   cmap=colormap,
                   alpha=alpha,
                   interpolation='gaussian')
        
        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label('Density')
        
        # Overlay scatter plot of the actual points if requested
        if overlay:
            plt.scatter(x_coords, y_coords, c='black', s=5, alpha=0.3)
        
        plt.title(f'Heatmap of Class {class_index} Bounding Box Centers')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.xlim(x_min, x_max)
        plt.ylim(y_max, y_min)  # Invert y-axis for image coordinates
        
        # Save the heatmap
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved to {output_path}")
        
    except np.linalg.LinAlgError as e:
        print(f"Error generating kernel density estimate: {e}")
        print("Try using more data points or adjusting the bandwidth parameter.")
        
        # Fallback to a 2D histogram if KDE fails
        print("Falling back to histogram-based heatmap...")
        plt.hist2d(x_coords, y_coords, bins=(50, 50), cmap=colormap, norm=colors.LogNorm())
        plt.colorbar(label='Count')
        plt.title(f'Histogram of Class {class_index} Bounding Box Centers')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.gca().invert_yaxis()  # Invert y-axis for image coordinates
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Histogram-based heatmap saved to {output_path}")

def main():
    """
    Main function to parse command line arguments and run the heatmap generation.
    """
    parser = argparse.ArgumentParser(description="Generate a heatmap of bounding box centers for a specific class from YOLO labels.")
    parser.add_argument("label_dir", help="Path to the directory containing the YOLO label files.")
    parser.add_argument("output_path", help="Path to save the output heatmap image (e.g., heatmap.png).")
    parser.add_argument("--image_width", type=int, default=640, help="Width of the image (default: 640).")
    parser.add_argument("--image_height", type=int, default=640, help="Height of the image (default: 640).")
    parser.add_argument("--class_index", type=int, default=4, help="Class index to visualize (default: 4).")
    parser.add_argument("--bandwidth", type=float, default=0.05, help="Bandwidth for kernel density estimation (default: 0.05).")
    parser.add_argument("--resolution", type=int, default=100, help="Resolution of the heatmap grid (default: 100).")
    parser.add_argument("--colormap", type=str, default='hot_r', help="Matplotlib colormap (default: 'hot_r').")
    parser.add_argument("--alpha", type=float, default=0.7, help="Transparency of the heatmap (default: 0.7).")
    parser.add_argument("--overlay", action='store_true', help="Overlay points on the heatmap.")

    args = parser.parse_args()

    # Validate arguments
    if not os.path.isdir(args.label_dir):
        print(f"Error: Label directory '{args.label_dir}' does not exist.")
        return
    
    if not args.output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        print("Error: Output path must be a valid image file name (.png, .jpg, .jpeg).")
        return

    generate_heatmap(
        args.label_dir, 
        args.output_path, 
        args.image_width, 
        args.image_height, 
        args.class_index,
        args.bandwidth,
        args.resolution,
        args.colormap,
        args.alpha,
        args.overlay
    )

if __name__ == "__main__":
    main()