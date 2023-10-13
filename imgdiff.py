from PIL import Image

# Open the two input images
img1 = Image.open("DRRNoutput0_image.png")
img2 = Image.open("DRRNoutput40_image.png")

# Get the dimensions of the images
width, height = img1.size

# Create a new image to hold the difference
diff = Image.new("RGB", (width, height), (0, 0, 0))

# Loop over each pixel in the images and calculate the difference
for x in range(width):
    for y in range(height):
        # Get the RGB values for the two pixels
        pixel1 = img1.getpixel((x, y))
        pixel2 = img2.getpixel((x, y))

        # Calculate the difference between the RGB values
        diff_r = abs(pixel1[0] - pixel2[0])
        diff_g = abs(pixel1[1] - pixel2[1])
        diff_b = abs(pixel1[2] - pixel2[2])

        # Set the pixel in the difference image to the difference value
        diff.putpixel((x, y), (diff_r, diff_g, diff_b))

# Save the difference image
diff.save("diff.png")
