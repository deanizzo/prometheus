from PIL import Image
import glob

# Load all PNGs in order
frames = [Image.open(f) for f in sorted(glob.glob("outputs/mma/LTEs/contours/*.png"))]

# Save as GIF
frames[0].save(
    "outputs/mma/animations/contour_evolution.gif",
    save_all=True,
    append_images=frames[1:],
    duration=200,       # ms per frame
    loop=0,             # 0 = infinite loop
    optimize=True
)