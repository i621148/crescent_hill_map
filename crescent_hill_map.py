import os
import re
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from PIL.ExifTags import GPSTAGS
import easyocr
import tkintermapview

# Initialize EasyOCR reader
ocr_reader = easyocr.Reader(['en'], gpu=False)

# Center coordinates for Crescent Hill Cemetery
DEFAULT_CENTER_LAT = 38.4255
DEFAULT_CENTER_LON = -94.35775


def extract_gps_from_image(image_path):
    """Extracts latitude and longitude floats from EXIF data in image file."""
    try:
        image = Image.open(image_path)
        exif = image.getexif()
        if not exif:
            return None

        gps_ifd = exif.get_ifd(0x8825)
        if not gps_ifd:
            return None

        gps_info = {GPSTAGS.get(key, key): val for key, val in gps_ifd.items()}

        if "GPSLatitude" not in gps_info or "GPSLongitude" not in gps_info:
            return None

        def convert_to_degrees(value):
            d, m, s = value
            return float(d) + (float(m) / 60.0) + (float(s) / 3600.0)

        lat = convert_to_degrees(gps_info["GPSLatitude"])
        if gps_info.get("GPSLatitudeRef") == "S":
            lat = -lat

        lon = convert_to_degrees(gps_info["GPSLongitude"])
        if gps_info.get("GPSLongitudeRef") == "W":
            lon = -lon

        return lat, lon
    except Exception as e:
        print(f"EXIF parsing error: {e}")
        return None


def preprocess_headstone_image(image_path):
    """Applies grayscale, CLAHE contrast enhancement, and sharpening for weathered stone."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    height, width = img.shape[:2]
    if max(height, width) > 1800:
        scale = 1800 / float(max(height, width))
        img = cv2.resize(img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return sharpened


def extract_text_details_from_image(image_path):
    """Reads headstone image using OpenCV preprocessing and improved name extraction heuristics."""
    name = "Unknown Name"
    dates = "Unknown Dates"

    STOP_WORDS = {
        "IN", "LOVING", "MEMORY", "OF", "REST", "PEACE", "HERE", "LIES",
        "BELOVED", "MOTHER", "FATHER", "SON", "DAUGHTER", "HUSBAND", "WIFE",
        "BROTHER", "SISTER", "BORN", "DIED", "AT", "MARRIED", "FOREVER",
        "TOGETHER", "OUR", "BABY", "CHILD", "SACRED", "TO", "THE"
    }

    try:
        processed_img = preprocess_headstone_image(image_path)
        if processed_img is not None:
            results = ocr_reader.readtext(processed_img, detail=0)
        else:
            results = ocr_reader.readtext(image_path, detail=0)

        if not results:
            return name, dates

        full_text = " ".join(results)

        years = re.findall(r"\b(1\d{3}|20\d{2})\b", full_text)
        if len(years) >= 2:
            dates = f"{years[0]}–{years[1]}"
        elif len(years) == 1:
            dates = f"b. {years[0]}"

        candidates = []
        for line in results:
            clean_line = re.sub(r"[^A-Za-z\s]", "", line).strip()
            words = [w.upper() for w in clean_line.split() if len(w) > 1]
            filtered_words = [w.title() for w in words if w not in STOP_WORDS]

            if filtered_words:
                candidate_str = " ".join(filtered_words)
                if 3 <= len(candidate_str) <= 30:
                    candidates.append(candidate_str)

        if candidates:
            upper_candidates = [c for c in candidates if c.isupper()]
            name = upper_candidates[0] if upper_candidates else candidates[0]

    except Exception as err:
        print(f"OCR Parsing error: {err}")

    return name, dates


def make_marker_label_draggable(marker, pin):
    """
    Renders a custom white background box for marker label text and enables
    independent drag-and-drop capability for the label text without moving the pin.
    """
    original_draw = marker.draw

    def draw_with_draggable_box(*args, **kwargs):
        original_draw(*args, **kwargs)
        if not hasattr(marker, "canvas_text") or not marker.canvas_text:
            return

        canvas = marker.map_widget.canvas
        dx, dy = pin.get("label_offset", (0, 0))

        # Shift canvas text by user drag offset
        if dx != 0 or dy != 0:
            canvas.move(marker.canvas_text, dx, dy)

        bbox = canvas.bbox(marker.canvas_text)
        if bbox:
            if hasattr(marker, "canvas_bg_box") and marker.canvas_bg_box:
                canvas.delete(marker.canvas_bg_box)

            pad_x = 6
            pad_y = 3
            marker.canvas_bg_box = canvas.create_rectangle(
                bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y,
                fill="white", outline="#333333", width=1, tags=("draggable_label",)
            )

            # Ensure text is on top of white box
            canvas.tag_raise(marker.canvas_text, marker.canvas_bg_box)

            # Bind drag events to label elements
            def start_drag(event):
                marker.drag_start_x = event.x
                marker.drag_start_y = event.y

            def do_drag(event):
                if hasattr(marker, "drag_start_x"):
                    move_x = event.x - marker.drag_start_x
                    move_y = event.y - marker.drag_start_y

                    cur_dx, cur_dy = pin.get("label_offset", (0, 0))
                    pin["label_offset"] = (cur_dx + move_x, cur_dy + move_y)

                    marker.drag_start_x = event.x
                    marker.drag_start_y = event.y

                    canvas.move(marker.canvas_text, move_x, move_y)
                    canvas.move(marker.canvas_bg_box, move_x, move_y)

            for tag_item in (marker.canvas_text, marker.canvas_bg_box):
                canvas.tag_bind(tag_item, "<ButtonPress-1>", start_drag)
                canvas.tag_bind(tag_item, "<B1-Motion>", do_drag)

    marker.draw = draw_with_draggable_box


class CemeteryMapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Crescent Hill Cemetery Gravestone Finder")
        self.root.geometry("1300x850")

        self.pins = []
        self.pending_photo_queue = []
        self.detail_popup = None

        self._build_gui()

    def _build_gui(self):
        toolbar = ttk.Frame(self.root, padding=5)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        btn_upload_photo = ttk.Button(toolbar, text="📷 Bulk Upload Photos", command=self.process_bulk_photo_upload)
        btn_upload_photo.pack(side=tk.LEFT, padx=3)

        btn_delete_selected = ttk.Button(toolbar, text="🗑️ Delete Selected Record", command=self.delete_selected_from_list)
        btn_delete_selected.pack(side=tk.LEFT, padx=3)

        self.lbl_status = ttk.Label(toolbar, text="Upload photos to place labeled dots. Labels are drag-positionable.", font=("Arial", 10, "italic"))
        self.lbl_status.pack(side=tk.LEFT, padx=10)

        main_frame = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_frame.pack(fill=tk.BOTH, expand=True)

        sidebar = ttk.Frame(main_frame, width=280, padding=10)
        main_frame.add(sidebar, weight=0)

        lbl_title = ttk.Label(sidebar, text="Uploaded Headstones", font=("Arial", 11, "bold"))
        lbl_title.pack(anchor=tk.W, pady=(0, 5))

        list_container = ttk.Frame(sidebar)
        list_container.pack(fill=tk.BOTH, expand=True)

        self.record_listbox = tk.Listbox(list_container, font=("Arial", 10), selectmode=tk.SINGLE)
        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.record_listbox.yview)
        self.record_listbox.config(yscrollcommand=scrollbar.set)

        self.record_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.record_listbox.bind("<<ListboxSelect>>", self.on_listbox_select)
        self.record_listbox.bind("<Delete>", lambda event: self.delete_selected_from_list())

        map_container = ttk.Frame(main_frame)
        main_frame.add(map_container, weight=1)

        self.map_widget = tkintermapview.TkinterMapView(map_container, corner_radius=0)
        self.map_widget.pack(fill=tk.BOTH, expand=True)

        self.map_widget.set_tile_server("https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", max_zoom=22)
        self.map_widget.set_position(DEFAULT_CENTER_LAT, DEFAULT_CENTER_LON)
        self.map_widget.set_zoom(18)

        self.map_widget.add_left_click_map_command(self.on_map_click)

    def redraw_pins(self):
        """Redraws red markers on map with well-spaced default positions and user-draggable labels."""
        self.map_widget.delete_all_marker()

        coords_group = {}
        for idx, pin in enumerate(self.pins, start=1):
            pin["id"] = idx
            if pin.get("gps"):
                lat, lon = pin["gps"]
                key = (round(lat, 5), round(lon, 5))
                if key not in coords_group:
                    coords_group[key] = []
                coords_group[key].append(pin)

        for key, pin_list in coords_group.items():
            for cluster_idx, pin in enumerate(pin_list):
                lat, lon = pin["gps"]

                # Default offset spacing (28px vertical gap per stacked item) if not dragged
                if "label_offset" not in pin:
                    pin["label_offset"] = (0, -28 * cluster_idx)

                marker_text = f" #{pin['id']} {pin['name']} "

                marker = self.map_widget.set_marker(
                    lat, lon, 
                    text=marker_text,
                    text_color="black",
                    marker_color_circle="#C23B22",
                    marker_color_outside="#8E2A18",
                    font=("Arial", 10, "bold"),
                    command=lambda p=pin: self.show_pin_details(p)
                )

                make_marker_label_draggable(marker, pin)
                pin["marker_obj"] = marker

        self.refresh_listbox()

    def refresh_listbox(self):
        self.record_listbox.delete(0, tk.END)
        for pin in self.pins:
            self.record_listbox.insert(tk.END, f"#{pin['id']} - {pin['name']}")

    def process_bulk_photo_upload(self):
        photo_paths = filedialog.askopenfilenames(
            title="Select Gravestone Photos (Bulk)",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if not photo_paths:
            return

        for photo_path in photo_paths:
            extracted_name, extracted_dates = extract_text_details_from_image(photo_path)

            photo_info = {
                "name": extracted_name,
                "dates": extracted_dates,
                "photo_path": photo_path
            }

            gps_coords = extract_gps_from_image(photo_path)
            if gps_coords:
                lat, lon = gps_coords
                photo_info["gps"] = (lat, lon)
                self.add_pin(photo_info)
            else:
                self.pending_photo_queue.append(photo_info)

        if self.pending_photo_queue:
            self.lbl_status.config(text=f"No GPS in photo. CLICK ON MAP to place Pin #{len(self.pins) + 1}.")
        else:
            self.lbl_status.config(text=f"Processed {len(photo_paths)} photo(s).")

    def add_pin(self, photo_info):
        pin_id = len(self.pins) + 1
        pin_data = {
            "id": pin_id,
            "gps": photo_info.get("gps"),
            "photo": photo_info["photo_path"],
            "name": photo_info["name"],
            "dates": photo_info["dates"]
        }
        self.pins.append(pin_data)
        self.redraw_pins()

    def delete_pin(self, pin_to_delete):
        """Removes a pin record, clears its marker, and re-indexes all pins."""
        if pin_to_delete in self.pins:
            self.pins.remove(pin_to_delete)
            self.redraw_pins()
            if self.detail_popup and self.detail_popup.winfo_exists():
                self.detail_popup.destroy()
            self.lbl_status.config(text="Record removed.")

    def delete_selected_from_list(self):
        selection = self.record_listbox.curselection()
        if selection:
            idx = selection[0]
            if idx < len(self.pins):
                pin = self.pins[idx]
                if messagebox.askyesno("Confirm Delete", f"Delete record #{pin['id']} ({pin['name']})?"):
                    self.delete_pin(pin)

    def on_map_click(self, coords):
        click_lat, click_lon = coords

        if self.pending_photo_queue:
            photo_info = self.pending_photo_queue.pop(0)
            photo_info["gps"] = (click_lat, click_lon)
            self.add_pin(photo_info)

            if self.pending_photo_queue:
                self.lbl_status.config(text=f"Placed Pin #{len(self.pins)}. CLICK MAP for Pin #{len(self.pins) + 1}.")
            else:
                self.lbl_status.config(text=f"Placed Pin #{len(self.pins)}. All photos mapped.")

    def on_listbox_select(self, event):
        selection = self.record_listbox.curselection()
        if selection:
            idx = selection[0]
            if idx < len(self.pins):
                pin = self.pins[idx]
                if pin.get("gps"):
                    self.map_widget.set_position(pin["gps"][0], pin["gps"][1])
                    self.map_widget.set_zoom(20)
                self.show_pin_details(pin)

    def show_pin_details(self, pin):
        if self.detail_popup is None or not self.detail_popup.winfo_exists():
            self.detail_popup = tk.Toplevel(self.root)
            self.detail_popup.geometry("520x720")

        for widget in self.detail_popup.winfo_children():
            widget.destroy()

        self.detail_popup.title(f"Record #{pin['id']}")
        ttk.Label(self.detail_popup, text=f"Pin #{pin['id']}", font=("Arial", 12, "bold")).pack(pady=(10, 2))

        form_frame = ttk.Frame(self.detail_popup, padding=10)
        form_frame.pack(fill=tk.X)

        ttk.Label(form_frame, text="Name:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=4)
        entry_name = ttk.Entry(form_frame, font=("Arial", 10), width=32)
        entry_name.insert(0, pin['name'])
        entry_name.grid(row=0, column=1, sticky=tk.E, pady=4)

        ttk.Label(form_frame, text="Dates:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky=tk.W, pady=4)
        entry_dates = ttk.Entry(form_frame, font=("Arial", 10), width=32)
        entry_dates.insert(0, pin['dates'])
        entry_dates.grid(row=1, column=1, sticky=tk.E, pady=4)

        if pin.get("gps"):
            lat, lon = pin["gps"]
            ttk.Label(form_frame, text="GPS Coordinates:", font=("Arial", 9, "bold")).grid(row=2, column=0, sticky=tk.W, pady=4)
            ttk.Label(form_frame, text=f"{lat:.6f}, {lon:.6f}", font=("Arial", 9)).grid(row=2, column=1, sticky=tk.W, pady=4)

        btn_container = ttk.Frame(self.detail_popup)
        btn_container.pack(pady=5)

        def save_changes():
            pin['name'] = entry_name.get().strip() or "Unknown Name"
            pin['dates'] = entry_dates.get().strip() or "Unknown Dates"
            self.redraw_pins()
            messagebox.showinfo("Saved", "Headstone record updated successfully.", parent=self.detail_popup)

        btn_save = ttk.Button(btn_container, text="💾 Save Changes", command=save_changes)
        btn_save.pack(side=tk.LEFT, padx=5)

        btn_delete = ttk.Button(btn_container, text="🗑️ Delete Record", command=lambda: self.delete_pin(pin))
        btn_delete.pack(side=tk.LEFT, padx=5)

        if os.path.exists(pin['photo']):
            zoom_frame = ttk.Frame(self.detail_popup)
            zoom_frame.pack(pady=(10, 0))

            base_img = Image.open(pin['photo'])
            zoom_level = [1.0]

            img_canvas = tk.Canvas(self.detail_popup, width=460, height=360, bg="#222222")
            img_canvas.pack(pady=5, fill=tk.BOTH, expand=True)

            def render_image():
                w, h = base_img.size
                scale = zoom_level[0]
                new_w = max(10, int(w * scale))
                new_h = max(10, int(h * scale))

                resample_method = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
                resized = base_img.resize((new_w, new_h), resample_method)
                tk_img = ImageTk.PhotoImage(resized)

                img_canvas.delete("all")
                img_canvas.create_image(230, 180, image=tk_img, anchor=tk.CENTER)
                img_canvas.image = tk_img

            def zoom_in():
                zoom_level[0] = min(zoom_level[0] * 1.25, 5.0)
                render_image()

            def zoom_out():
                zoom_level[0] = max(zoom_level[0] / 1.25, 0.1)
                render_image()

            def reset_zoom():
                w, h = base_img.size
                scale_w = 440 / float(w)
                scale_h = 340 / float(h)
                zoom_level[0] = min(scale_w, scale_h)
                render_image()

            btn_zoom_in = ttk.Button(zoom_frame, text="➕ Zoom In", command=zoom_in)
            btn_zoom_in.pack(side=tk.LEFT, padx=3)

            btn_zoom_out = ttk.Button(zoom_frame, text="➖ Zoom Out", command=zoom_out)
            btn_zoom_out.pack(side=tk.LEFT, padx=3)

            btn_zoom_reset = ttk.Button(zoom_frame, text="🔄 Reset Zoom", command=reset_zoom)
            btn_zoom_reset.pack(side=tk.LEFT, padx=3)

            def on_mouse_wheel(event):
                if event.delta > 0 or event.num == 4:
                    zoom_in()
                else:
                    zoom_out()

            img_canvas.bind("<MouseWheel>", on_mouse_wheel)
            img_canvas.bind("<Button-4>", on_mouse_wheel)
            img_canvas.bind("<Button-5>", on_mouse_wheel)

            reset_zoom()


if __name__ == "__main__":
    root = tk.Tk()
    app = CemeteryMapApp(root)
    root.mainloop()