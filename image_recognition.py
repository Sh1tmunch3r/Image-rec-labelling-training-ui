import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import os
import importlib.util
import random
import mss
import numpy as np
import tkinter as tk
import json
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from tkinter import filedialog
from tkinter.messagebox import showinfo, showerror

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

RECOGNIZER_FOLDER = "recognizers"
PROJECTS_FOLDER = "projects"
BOX_COLORS = [
    "#FF5733", "#33FF57", "#3357FF", "#F3FF33", "#FF33E3",
    "#33FFF4", "#FFA533", "#8D33FF", "#33FF8D", "#FF3380"
]

def load_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()

class RecognizerManager:
    def __init__(self, folder):
        self.folder = folder
        self.recognizers = self.load_recognizers()

    def load_recognizers(self):
        modules = {}
        for fname in os.listdir(self.folder):
            if fname.endswith(".py") and not fname.startswith("_"):
                path = os.path.join(self.folder, fname)
                spec = importlib.util.spec_from_file_location(fname[:-3], path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "Recognizer"):
                    modules[fname[:-3]] = module.Recognizer()
        return modules

    def get_names(self):
        return list(self.recognizers.keys())

    def recognize(self, name, image_np):
        return self.recognizers[name].recognize(image_np)

class ResultListBox(ctk.CTkFrame):
    def __init__(self, master, callback_highlight, callback_delete=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.scrollbar = ctk.CTkScrollbar(self)
        self.scrollbar.pack(side="right", fill="y")
        self.listbox = tk.Listbox(self, activestyle="none", height=7, font=("Segoe UI", 12))
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.configure(command=self.listbox.yview)
        self.callback_highlight = callback_highlight
        self.callback_delete = callback_delete
        self.listbox.bind("<<ListboxSelect>>", self.on_select)

    def update_results(self, results):
        self.listbox.delete(0, tk.END)
        for idx, r in enumerate(results):
            label = r.get("label", "Unknown")
            score = r.get("score", None)
            s = f"{label}"
            if score is not None and isinstance(score, float):
                s += f" ({score*100:.1f}%)"
            self.listbox.insert(tk.END, s)

    def on_select(self, e):
        sel = self.listbox.curselection()
        if sel:
            self.callback_highlight(sel[0])
        else:
            self.callback_highlight(None)

    def clear_selection(self):
        self.listbox.selection_clear(0, tk.END)

class AnnotationDataset(Dataset):
    def __init__(self, project_path, class_to_id):
        self.img_dir = os.path.join(project_path, "images")
        self.ann_dir = os.path.join(project_path, "annotations")
        self.imgs = [f for f in os.listdir(self.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.class_to_id = class_to_id

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        ann_name = os.path.splitext(img_name)[0] + '.json'
        ann_path = os.path.join(self.ann_dir, ann_name)
        if not os.path.exists(ann_path):
            return None, None

        img = Image.open(img_path).convert("RGB")
        with open(ann_path, 'r') as f:
            data = json.load(f)

        boxes = []
        labels = []
        for ann in data.get('annotations', []):
            box = ann.get('box')
            label = ann.get('label')
            if box and label in self.class_to_id:
                boxes.append(box)
                labels.append(self.class_to_id[label])

        if not boxes:
            return None, None

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        img = F.to_tensor(img)
        return img, target

class ImageRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Image Recognizer Pro")
        self.geometry("1100x700")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        if not os.path.exists(RECOGNIZER_FOLDER):
            os.makedirs(RECOGNIZER_FOLDER)
        if not os.path.exists(PROJECTS_FOLDER):
            os.makedirs(PROJECTS_FOLDER)

        self.recognizer_manager = RecognizerManager(RECOGNIZER_FOLDER)
        self.current_project = None
        self.classes = []
        self.current_image_path = None
        self.original_image = None
        self.annotations = []
        self.drawing = False
        self.rect = None
        self.highlighted_box_idx = None
        self.displayed_w = 0
        self.displayed_h = 0
        self.img_left = 0
        self.img_top = 0

        # Project bar
        self.project_frame = ctk.CTkFrame(self, height=50)
        self.project_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(self.project_frame, text="Project:").pack(side="left", padx=5)
        self.project_label = ctk.CTkLabel(self.project_frame, text="None")
        self.project_label.pack(side="left", padx=5)
        self.new_project_button = ctk.CTkButton(self.project_frame, text="New Project", command=self.new_project)
        self.new_project_button.pack(side="left", padx=5)
        self.open_project_button = ctk.CTkButton(self.project_frame, text="Open Project", command=self.open_project)
        self.open_project_button.pack(side="left", padx=5)

        # Tabview
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=5)

        self.setup_recognize_tab()
        self.setup_label_tab()
        self.setup_train_tab()

        self.reload_recognizers()

    def setup_recognize_tab(self):
        tab = self.tabview.add("Recognize")
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)

        # Left panel
        left_panel = ctk.CTkFrame(tab, width=230)
        left_panel.grid(row=0, column=0, sticky="ns", padx=10, pady=10)
        left_panel.grid_propagate(False)

        ctk.CTkLabel(left_panel, text="Recognizers", font=ctk.CTkFont(size=17, weight="bold")).pack(pady=(18, 8))
        names = self.recognizer_manager.get_names()
        self.selected_recognizer = ctk.StringVar(value=names[0] if names else "")
        self.recognizer_menu = ctk.CTkOptionMenu(left_panel, variable=self.selected_recognizer, values=names)
        self.recognizer_menu.pack(pady=4)

        self.rec_capture_button = ctk.CTkButton(left_panel, text="Capture & Recognize", command=self.rec_capture_and_recognize_thread)
        self.rec_capture_button.pack(pady=18)

        self.rec_status_label = ctk.CTkLabel(left_panel, text="Ready", text_color="gray")
        self.rec_status_label.pack(pady=(0, 15))

        ctk.CTkLabel(left_panel, text="Results", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 4))
        self.rec_result_panel = ResultListBox(left_panel, self.rec_highlight_box)
        self.rec_result_panel.pack(fill="x", padx=10, pady=(0, 12))

        self.rec_save_button = ctk.CTkButton(left_panel, text="Save Result", command=self.rec_save_image)
        self.rec_save_button.pack(pady=6)

        self.rec_copy_button = ctk.CTkButton(left_panel, text="Copy Labels", command=self.rec_copy_labels)
        self.rec_copy_button.pack(pady=3)

        # Image frame
        image_frame = ctk.CTkFrame(tab)
        image_frame.grid(row=0, column=1, sticky="nsew")
        image_frame.rowconfigure(0, weight=1)
        image_frame.columnconfigure(0, weight=1)

        self.rec_canvas = tk.Canvas(image_frame, bg="black", highlightthickness=0)
        self.rec_canvas.grid(row=0, column=0, sticky="nsew")
        self.rec_canvas.bind("<Configure>", self.rec_on_canvas_resize)

        self.rec_last_image = None
        self.rec_last_results = []
        self.rec_image_on_canvas = None

    def setup_label_tab(self):
        tab = self.tabview.add("Label")
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)

        # Left panel
        left_panel = ctk.CTkFrame(tab, width=230)
        left_panel.grid(row=0, column=0, sticky="ns", padx=10, pady=10)
        left_panel.grid_propagate(False)

        ctk.CTkLabel(left_panel, text="Classes", font=ctk.CTkFont(size=17, weight="bold")).pack(pady=(18, 8))
        self.add_class_button = ctk.CTkButton(left_panel, text="Add Class", command=self.add_class)
        self.add_class_button.pack(pady=4)

        self.classes_listbox = tk.Listbox(left_panel, height=5)
        self.classes_listbox.pack(fill="x", pady=4)

        ctk.CTkLabel(left_panel, text="Image", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 4))
        self.lab_capture_button = ctk.CTkButton(left_panel, text="Capture Image", command=self.lab_capture_image_thread)
        self.lab_capture_button.pack(pady=4)
        self.lab_load_button = ctk.CTkButton(left_panel, text="Load Image", command=self.lab_load_image)
        self.lab_load_button.pack(pady=4)

        self.lab_status_label = ctk.CTkLabel(left_panel, text="Ready", text_color="gray")
        self.lab_status_label.pack(pady=4)

        ctk.CTkLabel(left_panel, text="Annotations", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 4))
        self.lab_result_panel = ResultListBox(left_panel, self.lab_highlight_box)
        self.lab_result_panel.pack(fill="x", padx=10, pady=(0, 12))

        self.lab_delete_button = ctk.CTkButton(left_panel, text="Delete Selected", command=self.lab_delete_selected)
        self.lab_delete_button.pack(pady=4)

        self.lab_save_button = ctk.CTkButton(left_panel, text="Save Annotations", command=self.lab_save_annotations)
        self.lab_save_button.pack(pady=4)

        # Image frame
        image_frame = ctk.CTkFrame(tab)
        image_frame.grid(row=0, column=1, sticky="nsew")
        image_frame.rowconfigure(0, weight=1)
        image_frame.columnconfigure(0, weight=1)

        self.lab_canvas = tk.Canvas(image_frame, bg="black", highlightthickness=0)
        self.lab_canvas.grid(row=0, column=0, sticky="nsew")
        self.lab_canvas.bind("<Configure>", self.lab_on_canvas_resize)
        self.lab_canvas.bind("<Button-1>", self.lab_on_mouse_down)
        self.lab_canvas.bind("<B1-Motion>", self.lab_on_mouse_move)
        self.lab_canvas.bind("<ButtonRelease-1>", self.lab_on_mouse_up)

        self.lab_image_on_canvas = None

    def setup_train_tab(self):
        tab = self.tabview.add("Train")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        frame = ctk.CTkFrame(tab)
        frame.grid(row=0, column=0, sticky="nsew", padx=100, pady=100)

        ctk.CTkLabel(frame, text="Training Settings", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)

        ctk.CTkLabel(frame, text="Epochs").pack(pady=5)
        self.epochs_entry = ctk.CTkEntry(frame, placeholder_text="10")
        self.epochs_entry.pack(pady=5)

        self.train_button = ctk.CTkButton(frame, text="Train Model", command=self.train_model_thread)
        self.train_button.pack(pady=10)

        self.train_status_label = ctk.CTkLabel(frame, text="Ready", text_color="gray")
        self.train_status_label.pack(pady=10)

    def new_project(self):
        dialog = ctk.CTkInputDialog(title="New Project", text="Enter project name:")
        name = dialog.get_input()
        if name:
            path = os.path.join(PROJECTS_FOLDER, name)
            if os.path.exists(path):
                showerror("Error", "Project already exists.")
                return
            os.makedirs(path)
            os.makedirs(os.path.join(path, "images"))
            os.makedirs(os.path.join(path, "annotations"))
            with open(os.path.join(path, "classes.txt"), "w") as f:
                pass
            self.load_project(path)

    def open_project(self):
        projects = [d for d in os.listdir(PROJECTS_FOLDER) if os.path.isdir(os.path.join(PROJECTS_FOLDER, d))]
        if not projects:
            showinfo("Info", "No projects found.")
            return
        dialog = ctk.CTkInputDialog(title="Open Project", text="Select project:")
        # For simplicity, use entry, but better optionmenu, but inputdialog is entry.
        # List them.
        selected = ctk.StringVar(value=projects[0])
        option = ctk.CTkOptionMenu(self, values=projects, variable=selected)
        option.pack()  # temp
        # Wait, better custom dialog, but to simplify, assume user types name.
        name = dialog.get_input()
        if name in projects:
            path = os.path.join(PROJECTS_FOLDER, name)
            self.load_project(path)
        else:
            showerror("Error", "Project not found.")

    def load_project(self, path):
        self.current_project = path
        self.project_label.configure(text=os.path.basename(path))
        with open(os.path.join(path, "classes.txt"), "r") as f:
            self.classes = [line.strip() for line in f if line.strip()]
        self.update_classes_list()
        # Clear current image
        self.current_image_path = None
        self.original_image = None
        self.annotations = []
        self.lab_canvas.delete("all")
        self.lab_result_panel.update_results([])

    def update_classes_list(self):
        self.classes_listbox.delete(0, tk.END)
        for c in self.classes:
            self.classes_listbox.insert(tk.END, c)

    def add_class(self):
        if not self.current_project:
            showerror("Error", "No project selected.")
            return
        dialog = ctk.CTkInputDialog(title="Add Class", text="Enter class name:")
        name = dialog.get_input()
        if name and name not in self.classes:
            self.classes.append(name)
            with open(os.path.join(self.current_project, "classes.txt"), "a") as f:
                f.write(name + "\n")
            self.update_classes_list()

    def lab_capture_image_thread(self):
        if not self.current_project:
            showerror("Error", "No project selected.")
            return
        t = threading.Thread(target=self.lab_capture_image)
        t.daemon = True
        t.start()

    def lab_capture_image(self):
        self.lab_status_label.configure(text="Capturing...", text_color="#4A90E2")
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        self.lab_load_image_to_canvas(img)
        self.lab_status_label.configure(text="Captured", text_color="green")

    def lab_load_image(self):
        if not self.current_project:
            showerror("Error", "No project selected.")
            return
        f = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if f:
            img = Image.open(f)
            self.lab_load_image_to_canvas(img)

    def lab_load_image_to_canvas(self, img):
        img_name = f"img_{len(os.listdir(os.path.join(self.current_project, 'images'))):04d}.png"
        img_path = os.path.join(self.current_project, "images", img_name)
        img.save(img_path)
        self.current_image_path = img_path
        self.original_image = img
        ann_path = os.path.join(self.current_project, "annotations", os.path.splitext(img_name)[0] + ".json")
        if os.path.exists(ann_path):
            with open(ann_path, "r") as f:
                data = json.load(f)
                self.annotations = data.get("annotations", [])
        else:
            self.annotations = []
        self.lab_display_image_with_boxes(None)
        self.lab_result_panel.update_results(self.annotations)

    def lab_on_canvas_resize(self, event):
        self.lab_display_image_with_boxes(self.highlighted_box_idx)

    def lab_display_image_with_boxes(self, highlight_idx):
        if self.original_image is None:
            return
        canvas_w, canvas_h = self.lab_canvas.winfo_width(), self.lab_canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            return
        aspect_ratio = self.original_image.width / self.original_image.height
        canvas_ratio = canvas_w / canvas_h
        if aspect_ratio > canvas_ratio:
            new_w = canvas_w
            new_h = int(canvas_w / aspect_ratio)
        else:
            new_h = canvas_h
            new_w = int(canvas_h * aspect_ratio)
        img = self.original_image.copy().resize((new_w, new_h), Image.LANCZOS)
        draw = ImageDraw.Draw(img, "RGBA")
        font = load_font(size=max(14, new_h//32))
        for idx, res in enumerate(self.annotations):
            box = res.get("box")
            label = res.get("label", "Unknown")
            if not box:
                continue
            color = BOX_COLORS[idx % len(BOX_COLORS)]
            if highlight_idx == idx:
                outline = (255,255,255,255)
                width = 5
            else:
                outline = color
                width = 3
            x1, y1, x2, y2 = [int(a * new_w / self.original_image.width) for a in box[:2]] + [int(a * new_w / self.original_image.width) for a in box[2:]]
            draw.rounded_rectangle([x1, y1, x2, y2], radius=8, outline=outline, width=width, fill=color+"40")
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.rectangle([x1, y1 - text_h - 8, x1 + text_w + 10, y1], fill=color+"C0")
            draw.text((x1+5, y1-text_h-4), label, fill="white", font=font)
        tk_img = ImageTk.PhotoImage(img)
        self.lab_canvas.delete("all")
        self.lab_image_on_canvas = tk_img
        self.img_left = (canvas_w - new_w) // 2
        self.img_top = (canvas_h - new_h) // 2
        self.displayed_w = new_w
        self.displayed_h = new_h
        self.lab_canvas.create_image(canvas_w//2, canvas_h//2, anchor="center", image=tk_img)
        # Redraw temp rect if drawing
        if self.drawing and self.rect:
            self.lab_canvas.coords(self.rect, self.start_x, self.start_y, self.cur_x, self.cur_y)

    def lab_on_mouse_down(self, event):
        if self.original_image is None:
            return
        self.drawing = True
        self.start_x = self.lab_canvas.canvasx(event.x)
        self.start_y = self.lab_canvas.canvasy(event.y)
        self.rect = self.lab_canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='white', width=2, dash=(4,4))

    def lab_on_mouse_move(self, event):
        if self.drawing:
            self.cur_x = self.lab_canvas.canvasx(event.x)
            self.cur_y = self.lab_canvas.canvasy(event.y)
            self.lab_canvas.coords(self.rect, self.start_x, self.start_y, self.cur_x, self.cur_y)

    def lab_on_mouse_up(self, event):
        if self.drawing:
            self.drawing = False
            self.cur_x = self.lab_canvas.canvasx(event.x)
            self.cur_y = self.lab_canvas.canvasy(event.y)
            self.lab_canvas.delete(self.rect)
            self.rect = None

            # Calculate box in original coords
            x1 = min(self.start_x, self.cur_x) - self.img_left
            y1 = min(self.start_y, self.cur_y) - self.img_top
            x2 = max(self.start_x, self.cur_x) - self.img_left
            y2 = max(self.start_y, self.cur_y) - self.img_top

            if x1 < 0 or y1 < 0 or x2 > self.displayed_w or y2 > self.displayed_h or x2 - x1 < 5 or y2 - y1 < 5:
                return  # invalid

            orig_x1 = x1 * self.original_image.width / self.displayed_w
            orig_y1 = y1 * self.original_image.height / self.displayed_h
            orig_x2 = x2 * self.original_image.width / self.displayed_w
            orig_y2 = y2 * self.original_image.height / self.displayed_h

            box = [orig_x1, orig_y1, orig_x2, orig_y2]

            # Get label
            if self.classes:
                selected = ctk.StringVar(value=self.classes[0])
                dialog = ctk.CTkToplevel(self)
                dialog.title("Select Label")
                ctk.CTkLabel(dialog, text="Select or enter new label:").pack(pady=5)
                option = ctk.CTkOptionMenu(dialog, variable=selected, values=self.classes)
                option.pack(pady=5)
                entry = ctk.CTkEntry(dialog, placeholder_text="New label")
                entry.pack(pady=5)
                def submit():
                    lbl = entry.get().strip()
                    if lbl:
                        if lbl not in self.classes:
                            self.classes.append(lbl)
                            with open(os.path.join(self.current_project, "classes.txt"), "a") as f:
                                f.write(lbl + "\n")
                            self.update_classes_list()
                    else:
                        lbl = selected.get()
                    if lbl:
                        self.annotations.append({"box": box, "label": lbl})
                        self.lab_result_panel.update_results(self.annotations)
                        self.lab_display_image_with_boxes(None)
                    dialog.destroy()
                btn = ctk.CTkButton(dialog, text="OK", command=submit)
                btn.pack(pady=5)
                dialog.grab_set()
            else:
                dialog = ctk.CTkInputDialog(title="Enter Label", text="Enter label:")
                lbl = dialog.get_input()
                if lbl:
                    if lbl not in self.classes:
                        self.classes.append(lbl)
                        with open(os.path.join(self.current_project, "classes.txt"), "a") as f:
                            f.write(lbl + "\n")
                        self.update_classes_list()
                    self.annotations.append({"box": box, "label": lbl})
                    self.lab_result_panel.update_results(self.annotations)
                    self.lab_display_image_with_boxes(None)

    def lab_highlight_box(self, idx):
        self.highlighted_box_idx = idx
        self.lab_display_image_with_boxes(idx)

    def lab_delete_selected(self):
        sel = self.lab_result_panel.listbox.curselection()
        if sel:
            del self.annotations[sel[0]]
            self.lab_result_panel.update_results(self.annotations)
            self.lab_display_image_with_boxes(None)

    def lab_save_annotations(self):
        if not self.current_image_path or not self.annotations:
            return
        ann_name = os.path.splitext(os.path.basename(self.current_image_path))[0] + ".json"
        ann_path = os.path.join(self.current_project, "annotations", ann_name)
        data = {
            "image_width": self.original_image.width,
            "image_height": self.original_image.height,
            "annotations": self.annotations
        }
        with open(ann_path, "w") as f:
            json.dump(data, f)
        self.lab_status_label.configure(text="Saved", text_color="green")

    def train_model_thread(self):
        if not self.current_project:
            showerror("Error", "No project selected.")
            return
        t = threading.Thread(target=self.train_model)
        t.daemon = True
        t.start()

    def train_model(self):
        self.train_status_label.configure(text="Training...", text_color="#4A90E2")
        try:
            with open(os.path.join(self.current_project, "classes.txt"), "r") as f:
                classes = [line.strip() for line in f if line.strip()]
            if not classes:
                raise ValueError("No classes defined.")
            class_to_id = {c: i+1 for i, c in enumerate(classes)}
            num_classes = len(classes) + 1

            dataset = AnnotationDataset(self.current_project, class_to_id)
            # Filter None
            data = [d for d in dataset if d[0] is not None]
            if not data:
                raise ValueError("No valid annotated images.")
            dataloader = DataLoader(data, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

            model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

            epochs = int(self.epochs_entry.get() or 10)

            for epoch in range(epochs):
                model.train()
                for images, targets in dataloader:
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
                self.train_status_label.configure(text=f"Epoch {epoch+1}/{epochs}", text_color="#E67E22")

            model_path = os.path.join(self.current_project, "model.pth")
            torch.save(model.state_dict(), model_path)

            project_name = os.path.basename(self.current_project)
            self.generate_recognizer(project_name, model_path, classes)
            self.reload_recognizers()

            self.train_status_label.configure(text="Trained!", text_color="green")
        except Exception as e:
            showerror("Training Error", str(e))
            self.train_status_label.configure(text="Error", text_color="red")

    def generate_recognizer(self, project_name, model_path, classes):
        code = f"""
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image

class Recognizer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = {len(classes) + 1}
        self.model = fasterrcnn_resnet50_fpn(weights=None, num_classes=self.num_classes)
        self.model.load_state_dict(torch.load(r'{model_path}', weights_only=True, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.classes = {classes}

    def recognize(self, image_np):
        img = Image.fromarray(image_np).convert('RGB')
        img_t = F.to_tensor(img).to(self.device)
        with torch.no_grad():
            outputs = self.model([img_t])[0]
        results = []
        for i in range(len(outputs['boxes'])):
            score = outputs['scores'][i].item()
            if score > 0.5:
                label_id = outputs['labels'][i].item()
                label = self.classes[label_id - 1] if 1 <= label_id <= len(self.classes) else 'Unknown'
                box = outputs['boxes'][i].cpu().numpy().tolist()
                results.append({{'box': box, 'label': label, 'score': score}})
        return results
"""
        rec_path = os.path.join(RECOGNIZER_FOLDER, f"{project_name}.py")
        with open(rec_path, "w") as f:
            f.write(code)

    def reload_recognizers(self):
        self.recognizer_manager = RecognizerManager(RECOGNIZER_FOLDER)
        names = self.recognizer_manager.get_names()
        self.recognizer_menu.configure(values=names)
        if names:
            self.selected_recognizer.set(names[0])

    # Recognize tab functions
    def rec_on_canvas_resize(self, event):
        if self.rec_last_image is not None:
            self.rec_display_image_with_boxes(self.rec_last_image, self.rec_last_results, self.highlighted_box_idx)

    def rec_capture_and_recognize_thread(self):
        names = self.recognizer_manager.get_names()
        if not names:
            showerror("Error", "No recognizers available. Train a model first.")
            return
        t = threading.Thread(target=self.rec_capture_and_recognize)
        t.daemon = True
        t.start()

    def rec_capture_and_recognize(self):
        self.rec_status_label.configure(text="Capturing...", text_color="#4A90E2")
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        img_np = np.array(img)
        self.rec_status_label.configure(text="Recognizing...", text_color="#E67E22")
        rec_name = self.selected_recognizer.get()
        results = self.recognizer_manager.recognize(rec_name, img_np)
        self.rec_last_image = img
        self.rec_last_results = results
        self.rec_display_image_with_boxes(img, results, None)
        self.rec_result_panel.update_results(results)
        self.rec_status_label.configure(text="Done", text_color="green")

    def rec_display_image_with_boxes(self, pil_image, results, highlight_idx):
        w, h = self.rec_canvas.winfo_width(), self.rec_canvas.winfo_height()
        if w < 10 or h < 10:
            return
        aspect_ratio = pil_image.width / pil_image.height
        canvas_ratio = w / h
        if aspect_ratio > canvas_ratio:
            new_w = w
            new_h = int(w / aspect_ratio)
        else:
            new_h = h
            new_w = int(h * aspect_ratio)
        img = pil_image.copy().resize((new_w, new_h), Image.LANCZOS)
        draw = ImageDraw.Draw(img, "RGBA")
        font = load_font(size=max(14, new_h//32))
        for idx, res in enumerate(results):
            box = res.get("box")
            label = res.get("label", "Unknown")
            score = res.get("score", None)
            if not box: continue
            color = BOX_COLORS[idx % len(BOX_COLORS)]
            if highlight_idx == idx:
                outline = (255,255,255,255)
                width = 5
            else:
                outline = color
                width = 3
            x1, y1, x2, y2 = [int(a * new_w / pil_image.width) for a in box]
            draw.rounded_rectangle([x1, y1, x2, y2], radius=8, outline=outline, width=width, fill=color+"40")
            lbl = label
            if score is not None:
                lbl += f" ({score*100:.1f}%)"
            bbox = draw.textbbox((0, 0), lbl, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.rectangle([x1, y1 - text_h - 8, x1 + text_w + 10, y1], fill=color+"C0", outline=None)
            draw.text((x1+5, y1-text_h-4), lbl, fill="white", font=font)
        tk_img = ImageTk.PhotoImage(img)
        self.rec_canvas.delete("all")
        self.rec_image_on_canvas = tk_img
        self.rec_canvas.create_image(w//2, h//2, anchor="center", image=tk_img)

    def rec_highlight_box(self, idx):
        self.highlighted_box_idx = idx
        if self.rec_last_image is not None:
            self.rec_display_image_with_boxes(self.rec_last_image, self.rec_last_results, idx)

    def rec_save_image(self):
        if self.rec_last_image is None:
            return
        f = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png")])
        if not f: return
        img = self.rec_last_image.copy()
        draw = ImageDraw.Draw(img, "RGBA")
        font = load_font(size=18)
        for idx, res in enumerate(self.rec_last_results):
            box = res.get("box")
            label = res.get("label", "Unknown")
            score = res.get("score", None)
            if not box: continue
            color = BOX_COLORS[idx % len(BOX_COLORS)]
            x1, y1, x2, y2 = box
            draw.rounded_rectangle([x1, y1, x2, y2], radius=8, outline=color, width=4, fill=color+"40")
            lbl = label
            if score is not None:
                lbl += f" ({score*100:.1f}%)"
            bbox = draw.textbbox((0,0), lbl, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.rectangle([x1, y1 - text_h - 8, x1 + text_w + 10, y1], fill=color+"C0", outline=None)
            draw.text((x1+5, y1-text_h-4), lbl, fill="white", font=font)
        img.save(f)
        self.rec_status_label.configure(text="Saved!", text_color="#27AE60")

    def rec_copy_labels(self):
        if not self.rec_last_results:
            return
        s = "\n".join([f"{r.get('label', 'Unknown')} ({r.get('score', 0)*100:.1f}%)" if "score" in r else r.get('label', 'Unknown') for r in self.rec_last_results])
        self.clipboard_clear()
        self.clipboard_append(s)
        self.rec_status_label.configure(text="Labels Copied!", text_color="#27AE60")

if __name__ == "__main__":
    app = ImageRecognitionApp()
    app.mainloop()