import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def generate_heatmap(label_dir, output_path, image_width=640, image_height=640, class_index=4, dot_size=10):
    """
    Generates a heatmap visualization of bounding box centers for a specific class from YOLO-formatted labels.

    Args:
        label_dir (str): Path to the directory containing the YOLO-formatted label text files.
        output_path (str): Path to save the generated heatmap image.
        image_width (int, optional): Width of the image used for normalization. Defaults to 640.
        image_height (int, optional): Height of the image used for normalization. Defaults to 640.
        class_index (int, optional): The class index to visualize. Defaults to 4.
        dot_size (int, optional): Size of the dots in the heatmap.
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
                            print(f"Error processing line in {label_file_path}: {line.strip()}. Skipping.  Error: {e}")
            except FileNotFoundError:
                print(f"Error: File not found at {label_file_path}")
                continue  # Go to the next file
            except Exception as e:
                print(f"An unexpected error occurred while processing {label_file_path}: {e}")
                continue

    if not x_coords:
        print(f"No objects of class {class_index} found in the provided labels.")
        return

    # Create the heatmap
    plt.figure(figsize=(8, 8))
    plt.plot(x_coords, y_coords, 'o', markersize=dot_size, color='red', alpha=0.5)  # Increased alpha
    plt.title(f'Heatmap of Class {class_index} Bounding Box Centers')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.xlim(0, image_width)  # Set appropriate limits
    plt.ylim(image_height, 0)  # Invert y-axis for image coordinates
    plt.grid(False) # Remove grid lines.

    # Save the heatmap
    plt.savefig(output_path)
    plt.show()  # Display the heatmap

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
    parser.add_argument("--dot_size", type=int, default=10, help="Size of the dots in the heatmap (default: 10).")

    args = parser.parse_args()

    # Validate arguments
    if not os.path.isdir(args.label_dir):
        print(f"Error: Label directory '{args.label_dir}' does not exist.")
        return
    if not args.output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        print("Error: Output path must be a valid image file name (.png, .jpg, .jpeg).")
        return

    generate_heatmap(args.label_dir, args.output_path, args.image_width, args.image_height, args.class_index, args.dot_size)

if __name__ == "__main__":
    main()
