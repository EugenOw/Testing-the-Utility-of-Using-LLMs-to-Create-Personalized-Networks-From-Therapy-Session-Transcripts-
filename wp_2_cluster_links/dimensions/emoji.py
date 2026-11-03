from PIL import Image, ImageDraw
import os

def generate_colored_circle(color_hex, size=32, output_dir="dimension_icons", name="circle"):
    os.makedirs(output_dir, exist_ok=True)
    image = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    padding = 4
    draw.ellipse(
        [padding, padding, size - padding, size - padding],
        fill=color_hex,
        outline="black"
    )
    path = os.path.join(output_dir, f"{name}.png")
    image.save(path)
    print(f"✅ Saved: {path}")
    return path

# Example: Generate icons for each dimension
dimension_colors = {
    "Cognition": "#0000FF",
    "Sense of Self": "#006400",
    "(Overt) Behaviour": "#FF0000",
    "Affect": "#800080",
    "Moderators": "#FFA500",
    "Attention": "#FFFF00",
    "Biophysiological": "#FFD1DC",
    "Motivation": "#FF6347",
    "Sociocultural": "#00FF00"
}

for dim, color in dimension_colors.items():
    generate_colored_circle(color, name=dim.replace(" ", "_").replace("(", "").replace(")", ""))
