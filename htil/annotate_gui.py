import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import json
import glob

# === CONFIG ===
ENSEMBLE_ROOT_DIR = "results/sahi_ensemble_outputs"
SAHI_OUTPUT_DIR = "results/sahi_outputs"


def get_latest_sahi_output_folder():
    """Get the latest timestamped SAHI output folder for saving corrections"""
    if not os.path.exists(SAHI_OUTPUT_DIR):
        raise Exception(f"SAHI output directory not found: {SAHI_OUTPUT_DIR}")

    # Get all timestamped folders
    subfolders = [
        os.path.join(SAHI_OUTPUT_DIR, d)
        for d in os.listdir(SAHI_OUTPUT_DIR)
        if os.path.isdir(os.path.join(SAHI_OUTPUT_DIR, d))
        and d.replace("_", "").replace("-", "").isdigit()
    ]

    if not subfolders:
        raise Exception("No timestamped folders found in SAHI outputs.")

    # Get the latest folder by modification time
    latest_folder = sorted(subfolders, key=os.path.getmtime)[-1]
    corrections_dir = os.path.join(latest_folder, "corrections")
    os.makedirs(corrections_dir, exist_ok=True)

    return corrections_dir


def get_latest_ensemble_folder():
    subfolders = [
        os.path.join(ENSEMBLE_ROOT_DIR, d) for d in os.listdir(ENSEMBLE_ROOT_DIR)
    ]
    subfolders = [d for d in subfolders if os.path.isdir(d)]
    if not subfolders:
        raise Exception("No subfolders found in ensemble outputs.")
    return sorted(subfolders, key=os.path.getmtime)[-1]


class CorrectionTool:
    def __init__(self, master):
        self.master = master
        self.master.title("Room Correction Tool")
        self.master.geometry("1200x800")  # Set initial window size

        self.canvas = tk.Canvas(master, cursor="tcross", bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Get the corrections directory in the latest SAHI output folder
        try:
            self.corrections_dir = get_latest_sahi_output_folder()
            print(f"[INFO] Corrections will be saved to: {self.corrections_dir}")
        except Exception as e:
            print(f"[ERROR] {e}")
            # Fallback to old location if SAHI outputs not found
            self.corrections_dir = "corrections/json"
            os.makedirs(self.corrections_dir, exist_ok=True)
            print(
                f"[WARN] Using fallback corrections directory: {self.corrections_dir}"
            )

        # Load images from latest ensemble folder
        latest_ensemble_dir = get_latest_ensemble_folder()
        self.ensemble_dir = os.path.join(latest_ensemble_dir, "ensemble_results")
        self.image_paths = sorted(glob.glob(os.path.join(self.ensemble_dir, "*.jpg")))

        self.image_index = 0
        self.rectangles = []
        self.tk_image = None
        self.image_size = (0, 0)
        self.scale_factor = 1.0

        # Bind events first
        self.canvas.bind("<Button-1>", self.on_click_start)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.on_right_click)

        self.master.bind("<Right>", self.next_image)
        self.master.bind("<Left>", self.prev_image)
        self.master.bind("<s>", self.save_annotations)

        # Load image after window is set up
        self.master.after(100, self.load_image)  # Delay to ensure canvas is ready

    def load_image(self):
        if not self.image_paths:
            print("No images found in ensemble_results.")
            return

        img_path = self.image_paths[self.image_index]
        self.current_image_name = os.path.basename(img_path)

        pil_image = Image.open(img_path).convert("RGB")
        self.image_size = pil_image.size

        # Scale image to fit canvas if it's too large
        self.master.update_idletasks()  # Ensure canvas dimensions are updated
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 100 and canvas_height > 100:  # Canvas has reasonable size
            scale_x = (canvas_width - 20) / self.image_size[0]  # Leave some margin
            scale_y = (canvas_height - 20) / self.image_size[1]
            scale = min(scale_x, scale_y, 1.0)  # Don't upscale

            if scale < 1.0:
                new_width = int(self.image_size[0] * scale)
                new_height = int(self.image_size[1] * scale)
                pil_image = pil_image.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS
                )
                self.scale_factor = scale
            else:
                self.scale_factor = 1.0
        else:
            # Fallback: scale down large images
            if self.image_size[0] > 1200 or self.image_size[1] > 800:
                scale = min(1200 / self.image_size[0], 800 / self.image_size[1])
                new_width = int(self.image_size[0] * scale)
                new_height = int(self.image_size[1] * scale)
                pil_image = pil_image.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS
                )
                self.scale_factor = scale
            else:
                self.scale_factor = 1.0

        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.rectangles.clear()

        print(
            f"[INFO] Viewing {self.current_image_name} | Size: {self.image_size} | Scale: {self.scale_factor:.2f}"
        )

        # Load existing corrections if available
        json_path = os.path.join(
            self.corrections_dir, f"{os.path.splitext(self.current_image_name)[0]}.json"
        )
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                for box in data.get("final_detections", []):
                    x1, y1, x2, y2 = map(int, box["bbox"])
                    # Scale coordinates if image was resized
                    if hasattr(self, "scale_factor"):
                        x1, y1, x2, y2 = (
                            int(x1 * self.scale_factor),
                            int(y1 * self.scale_factor),
                            int(x2 * self.scale_factor),
                            int(y2 * self.scale_factor),
                        )
                    rect_id = self.canvas.create_rectangle(
                        x1, y1, x2, y2, outline="red", width=2
                    )
                    self.rectangles.append(((x1, y1, x2, y2), rect_id))

    def on_click_start(self, event):
        self.start_x, self.start_y = event.x, event.y
        self.temp_rectangle = self.canvas.create_rectangle(
            self.start_x,
            self.start_y,
            event.x,
            event.y,
            outline="green",
            width=2,
            dash=(4, 2),
        )

    def on_drag(self, event):
        if self.temp_rectangle:
            self.canvas.coords(
                self.temp_rectangle, self.start_x, self.start_y, event.x, event.y
            )

    def on_release(self, event):
        if self.temp_rectangle:
            x1, y1, x2, y2 = map(int, self.canvas.coords(self.temp_rectangle))
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                rect_id = self.canvas.create_rectangle(
                    x1, y1, x2, y2, outline="red", width=2
                )
                self.rectangles.append(((x1, y1, x2, y2), rect_id))
            self.canvas.delete(self.temp_rectangle)
            self.temp_rectangle = None

    def on_right_click(self, event):
        to_remove = None
        for bbox, rect_id in self.rectangles:
            x1, y1, x2, y2 = bbox
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                to_remove = rect_id
                break
        if to_remove:
            self.canvas.delete(to_remove)
            self.rectangles = [(b, r) for b, r in self.rectangles if r != to_remove]

    def save_annotations(self, event=None):
        json_path = os.path.join(
            self.corrections_dir, f"{os.path.splitext(self.current_image_name)[0]}.json"
        )
        final_detections = []
        for (x1, y1, x2, y2), _ in self.rectangles:
            # Convert back to original image coordinates
            if hasattr(self, "scale_factor") and self.scale_factor != 1.0:
                orig_x1 = int(x1 / self.scale_factor)
                orig_y1 = int(y1 / self.scale_factor)
                orig_x2 = int(x2 / self.scale_factor)
                orig_y2 = int(y2 / self.scale_factor)
            else:
                orig_x1, orig_y1, orig_x2, orig_y2 = int(x1), int(y1), int(x2), int(y2)

            final_detections.append(
                {"bbox": [orig_x1, orig_y1, orig_x2, orig_y2], "label": "room"}
            )

        data = {
            "image": self.current_image_name,
            "size": {"width": self.image_size[0], "height": self.image_size[1]},
            "final_detections": final_detections,
        }
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[✓] Saved corrections to {json_path}")

        # Create data.yaml in the corrected_dataset directory if it doesn't exist
        self.create_data_yaml()

    def create_data_yaml(self):
        """Create data.yaml file in the corrected_dataset directory"""
        # Get the parent directory of corrections (the timestamped folder)
        parent_dir = os.path.dirname(self.corrections_dir)
        corrected_dataset_dir = os.path.join(parent_dir, "corrected_dataset")

        # Only create if corrected_dataset directory exists (created by convert script)
        if os.path.exists(corrected_dataset_dir):
            data_yaml_path = os.path.join(corrected_dataset_dir, "data.yaml")

            # Only create if it doesn't already exist
            if not os.path.exists(data_yaml_path):
                yaml_content = """# corrected_dataset/data.yaml

train: corrected_dataset/images
val: corrected_dataset/images  # use same for small corrections

nc: 1
names: ['room']
"""
                with open(data_yaml_path, "w") as f:
                    f.write(yaml_content)
                print(f"[✓] Created data.yaml at {data_yaml_path}")

    def next_image(self, event=None):
        self.save_annotations()
        if self.image_index + 1 < len(self.image_paths):
            self.image_index += 1
            self.load_image()

    def prev_image(self, event=None):
        self.save_annotations()
        if self.image_index > 0:
            self.image_index -= 1
            self.load_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = CorrectionTool(root)
    root.mainloop()
