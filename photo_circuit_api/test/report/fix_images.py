import os
from PIL import Image, ImageOps

# Directory containing the images
input_directory = 'images'
output_directory = 'output_images'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)


def process_image(image_path, output_path):
  with Image.open(image_path) as img:
    # Add white background
    img = ImageOps.expand(img, border=(max(0, 50 - img.size[0]) // 2, max(0, 50 - img.size[1]) // 2), fill='white')
    
    # Crop to 50x50 if necessary
    img = ImageOps.fit(img, (50, 50), method=0, bleed=0.0, centering=(0.5, 0.5))
    
    # Save the modified image
    img.save(output_path)


# Process each image in the directory
for filename in os.listdir(input_directory):
  if filename.endswith('.png'):
    input_path = os.path.join(input_directory, filename)
    output_path = os.path.join(output_directory, filename)
    process_image(input_path, output_path)

print("Processing complete.")
