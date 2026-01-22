import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import sys
import os
from pathlib import Path
import threading
import time
import json

# Setup path to handle imports correctly
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from multi_cancer_ai.config import config
from multi_cancer_ai.src.active_trainer import ActiveTrainer

class CancerDiagApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window setup
        self.title("Multi-Cancer AI Diagnostic System (Extended)")
        self.geometry("1100x700")

        # Config
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        
        # Grid Configuration (1x2)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # State
        self.lang = "it" # Default Language
        self.active_trainer = ActiveTrainer()
        self.trainer_loaded = False
        
        # Data placeholders
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
        # Dynamic Class Loading
        classes_json_path = config.MODELS_DIR / "classes.json"
        if classes_json_path.exists():
            print(f"[INFO] Found dynamic classes configuration at {classes_json_path}")
            with open(classes_json_path, 'r') as f:
                data = json.load(f)
                self.classes = sorted(data["classes"])
                self.mapping = data["mapping"]
        else:
            self.classes = sorted(list(config.CLASS_MAPPING.keys()))
            self.mapping = config.CLASS_MAPPING
            
        # Modules placeholders
        self.np = None
        self.cv2 = None
        self.model = None
        self.model_type = None

        # Build UI
        self._create_sidebar()
        self._create_frames()
        
        # Start Loading Backend
        self.status_label = None # Will be ref'd in frames
        self.after(500, self._initial_backend_load)

    def _create_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        ctk.CTkLabel(self.sidebar_frame, text="🏥 AI Diagnostic", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.btn_home = ctk.CTkButton(self.sidebar_frame, text="Home & Status", command=lambda: self.show_frame("Home"))
        self.btn_home.grid(row=1, column=0, padx=20, pady=10)
        
        self.btn_diag = ctk.CTkButton(self.sidebar_frame, text="Diagnosis System", command=lambda: self.show_frame("Diagnosis"))
        self.btn_diag.grid(row=2, column=0, padx=20, pady=10)
        
        self.btn_train = ctk.CTkButton(self.sidebar_frame, text="Active Training", command=lambda: self.show_frame("Training"))
        self.btn_train.grid(row=3, column=0, padx=20, pady=10)

        self.btn_info = ctk.CTkButton(self.sidebar_frame, text="Dataset Info", command=self.show_dataset_info, fg_color="gray40", hover_color="gray50")
        self.btn_info.grid(row=4, column=0, padx=20, pady=10)

        # Bottom
        self.lbl_mode = ctk.CTkLabel(self.sidebar_frame, text="Tema:", anchor="w")
        self.lbl_mode.grid(row=5, column=0, padx=20, pady=(10, 0))
        ctk.CTkOptionMenu(self.sidebar_frame, values=["Dark", "Light"], command=self.change_appearance_mode).grid(row=6, column=0, padx=20, pady=(5, 10))
        
        # Language Toggle
        self.btn_lang = ctk.CTkButton(self.sidebar_frame, text="🇮🇹 IT / 🇺🇸 EN", command=self.toggle_language, fg_color="transparent", border_width=1)
        self.btn_lang.grid(row=7, column=0, padx=20, pady=(10, 20))

    def toggle_language(self):
        self.lang = "en" if self.lang == "it" else "it"
        self.update_ui_text()
        
    def update_ui_text(self):
        from multi_cancer_ai.src.localization import TRANSLATIONS
        t = TRANSLATIONS[self.lang]
        
        # Sidebar
        self.btn_home.configure(text=t["sidebar_home"])
        self.btn_diag.configure(text=t["sidebar_diagnose"])
        self.btn_train.configure(text=t["sidebar_train"])
        self.btn_info.configure(text=t["sidebar_info"])
        self.lbl_mode.configure(text=t["sidebar_mode"])
        
        # Propagate to frames if they have the method
        for frame in self.frames.values():
            if hasattr(frame, 'update_language'):
                frame.update_language(self.lang)

    def show_dataset_info(self):
        try:
            from multi_cancer_ai.tools import inspect_dataset
            from multi_cancer_ai.src.localization import TRANSLATIONS, MEDICAL_GLOSSARY
            
            t = TRANSLATIONS[self.lang]
            glossary = MEDICAL_GLOSSARY[self.lang]
            
            info_text = inspect_dataset.get_structure_string(glossary=glossary)
            
            # Create Pop-up
            top = ctk.CTkToplevel(self)
            top.title(t["info_title"])
            top.geometry("700x700")
            
            # Textbox
            textbox = ctk.CTkTextbox(top, font=ctk.CTkFont(family="Courier", size=12))
            textbox.pack(fill="both", expand=True, padx=10, pady=10)
            textbox.insert("0.0", info_text)
            textbox.configure(state="disabled") # Read-only
            
            top.grab_set() 
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load dataset info: {e}")

    def _create_frames(self):
        self.frames = {}
        
        # 1. Home Frame
        self.frames["Home"] = HomeFrame(self)
        self.frames["Home"].grid(row=0, column=1, sticky="nsew")
        
        # 2. Diagnosis Frame
        self.frames["Diagnosis"] = DiagnosisFrame(self)
        self.frames["Diagnosis"].grid(row=0, column=1, sticky="nsew")

        # 3. Training Frame
        self.frames["Training"] = TrainingFrame(self)
        self.frames["Training"].grid(row=0, column=1, sticky="nsew")

        self.show_frame("Home")

    def show_frame(self, name):
        frame = self.frames[name]
        frame.tkraise()
        # Visual feedback on buttons
        self.btn_home.configure(fg_color=("gray75", "gray25") if name != "Home" else "#1f6aa5")
        self.btn_diag.configure(fg_color=("gray75", "gray25") if name != "Diagnosis" else "#1f6aa5")
        self.btn_train.configure(fg_color=("gray75", "gray25") if name != "Training" else "#1f6aa5")

    def change_appearance_mode(self, mode):
        ctk.set_appearance_mode(mode)

    def _initial_backend_load(self):
        threading.Thread(target=self._load_backend_worker, daemon=True).start()

    def _load_backend_worker(self):
        # ... logic to load libs and model ...
        # (This logic is moved to DiagnosisFrame generally, but we keep core model state in App)
        import numpy as np
        import cv2
        self.np = np
        self.cv2 = cv2
        
        # Trigger model load in the Diagnosis Frame
        self.frames["Diagnosis"].load_model_backend()


# --- Sub-Frames ---

class HomeFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, corner_radius=0, fg_color="transparent")
        self.controller = master
        
        self.lbl_main = ctk.CTkLabel(self, text="Welcome", font=ctk.CTkFont(size=24, weight="bold"))
        self.lbl_main.pack(pady=(40, 20))
        
        # Status
        self.status_frame = ctk.CTkFrame(self)
        self.status_frame.pack(fill="x", padx=40, pady=20)
        
        self.lbl_status_title = ctk.CTkLabel(self.status_frame, text="System Status", font=ctk.CTkFont(size=16, weight="bold"))
        self.lbl_status_title.pack(pady=(10, 5))
        
        self.lbl_model_status = ctk.CTkLabel(self.status_frame, text="Checking model...", text_color="orange")
        self.lbl_model_status.pack(pady=5)
        
        self.lbl_model_name = ctk.CTkLabel(self.status_frame, text="")
        self.lbl_model_name.pack(pady=(0, 10))

        # Instructions
        self.lbl_instr = ctk.CTkLabel(self, text="", justify="left", font=ctk.CTkFont(size=14))
        self.lbl_instr.pack(pady=40, padx=40)
        
        # Initial Text Update
        self.update_language(self.controller.lang)

    def update_language(self, lang):
        from multi_cancer_ai.src.localization import TRANSLATIONS
        t = TRANSLATIONS[lang]
        
        self.lbl_main.configure(text=t["home_welcome"])
        self.lbl_status_title.configure(text=t["home_status"])
        self.lbl_instr.configure(text=t["home_instr"])

class DiagnosisFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, corner_radius=0, fg_color="transparent")
        self.app = master
        self.pack_propagate(False) 
        
        # Grid Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=4) 
        self.grid_rowconfigure(1, weight=1) 

        # Left: Image
        self.img_frame = ctk.CTkFrame(self)
        self.img_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.canvas = ctk.CTkLabel(self.img_frame, text="No Image")
        self.canvas.pack(expand=True)
        
        # Right: Results
        self.res_frame = ctk.CTkScrollableFrame(self, label_text="Results")
        self.res_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")
        
        # Bottom: Controls
        self.ctrl_frame = ctk.CTkFrame(self, height=50)
        self.ctrl_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        
        self.btn_load = ctk.CTkButton(self.ctrl_frame, text="Upload Image", command=self.load_image)
        self.btn_load.pack(side="left", padx=20, pady=10)
        
        self.status = ctk.CTkLabel(self.ctrl_frame, text="System Init...")
        self.status.pack(side="left", padx=20)
        
        # If result is wrong button (hidden initially)
        self.btn_wrong = ctk.CTkButton(self.ctrl_frame, text="Wrong?", fg_color="orange", hover_color="#d35400", command=self._on_wrong_diag)
        
        self.progress = ctk.CTkProgressBar(self.ctrl_frame, mode="indeterminate", width=150)

        # Variables
        self.current_img_path = None
        self.current_pil_img = None
        self.predictions = None
        
        # Init Language
        self.update_language(self.app.lang)

    def update_language(self, lang):
        from multi_cancer_ai.src.localization import TRANSLATIONS
        t = TRANSLATIONS[lang]
        
        self.btn_load.configure(text=t["diag_load_btn"])
        self.res_frame.configure(label_text=t["diag_result"].format("..."))
        self.btn_wrong.configure(text=t["diag_wrong_btn"])

    def _on_wrong_diag(self):
        # Switch to training tab and pass current image
        if self.current_img_path:
            self.app.frames["Training"].set_training_image(self.current_img_path)
            self.app.show_frame("Training")

        # Variables
        self.current_img_path = None
        self.current_pil_img = None
        self.predictions = None
    
    def load_model_backend(self):
        # Triggered by App's thread
        try:
            self.status.configure(text="Loading Model...")
            self.progress.pack(side="left", padx=10)
            self.progress.start()
            
            # Paths
            tflite_path = config.MODELS_DIR / "model_optimized.tflite"
            keras_path = config.MODELS_DIR / "best_model.h5"
            custom_path = config.MODELS_DIR / "best_model_custom.h5"
            tuned_path = config.MODELS_DIR / "best_model_tuned.h5"

            # 1. Custom Model First (User Corrections)
            if custom_path.exists():
                print(f"[INFO] Found custom trained model: {custom_path}")
                import tensorflow as tf
                self.app.model = tf.keras.models.load_model(str(custom_path))
                self.app.model_type = 'keras'
                self._on_model_loaded("Custom AI (User Edits)")
                return

            # 2. Tuned Model Second (Kaggle Imports)
            if tuned_path.exists():
                print(f"[INFO] Found fine-tuned model: {tuned_path}")
                import tensorflow as tf
                # Check compatibility with classes.json? (Assumed implicit)
                self.app.model = tf.keras.models.load_model(str(tuned_path))
                self.app.model_type = 'keras'
                self._on_model_loaded("AI (Kaggle Tuned)")
                return

            # 2. Try TFLite
            try:
                print("[INFO] Attempting to load TFLite model...")
                if not tflite_path.exists():
                    raise FileNotFoundError("TFLite model not found")

                # Try importing TFLite
                try:
                    import tflite_runtime.interpreter as tflite
                except ImportError:
                    import tensorflow.lite as tflite
                
                self.app.interpreter = tflite.Interpreter(model_path=str(tflite_path))
                self.app.interpreter.allocate_tensors()
                
                self.app.input_details = self.app.interpreter.get_input_details()
                self.app.output_details = self.app.interpreter.get_output_details()
                self.app.model_type = 'tflite'
                self._on_model_loaded("TFLite Model (Fast)")
                return

            except Exception as e_tflite:
                print(f"[WARN] TFLite Load Failed: {e_tflite}")

            # 3. Fallback to Base Keras
            print("[INFO] Falling back to Keras model...")
            import tensorflow as tf
            if not keras_path.exists():
                raise FileNotFoundError(f"Keras model not found at {keras_path}")

            self.app.model = tf.keras.models.load_model(str(keras_path))
            self.app.model_type = 'keras'
            self._on_model_loaded("Keras Model (Base)")

        except Exception as e:
            print(f"[ERROR] Init Failed: {e}")
            self.status.configure(text=f"Error: {str(e)}", text_color="red")
            self.progress.stop()

    def _on_model_loaded(self, msg):
        self.app.trainer_loaded = True
        # Schedule UI update on main thread
        self.after(0, lambda: self._update_status_ready(msg))

    def _update_status_ready(self, msg):
        self.status.configure(text=f"Ready: {msg}", text_color="green")
        self.progress.stop()
        self.progress.pack_forget()

    def load_image(self):
        try:
            path = filedialog.askopenfilename(title="Select Scan", filetypes=[("Images", "*.png *.jpg *.jpeg *.tif")])
            if path:
                self.current_img_path = path
                self._show_image(path)
                self._run_inference(path)
        except Exception as e:
            print(f"Error: {e}")

    def _show_image(self, path):
        img = Image.open(path)
        self.current_pil_img = img
        # Resize for display
        display_img = img.copy()
        display_img.thumbnail((400, 400), Image.Resampling.LANCZOS)
        tk_img = ctk.CTkImage(light_image=display_img, dark_image=display_img, size=display_img.size)
        self.canvas.configure(image=tk_img, text="")
        self.canvas.image = tk_img

    def _run_inference(self, path):
         # Run on thread
         threading.Thread(target=self._inference_worker, args=(path,), daemon=True).start()

    def _inference_worker(self, path):
        if not self.app.model_type:
            return

        try:
            start_time = time.time()
            # Load and Preprocess
            pil_image = Image.open(path)
            # Use High Quality Resampling for Inference too
            img_resized = pil_image.resize((config.IMG_WIDTH, config.IMG_HEIGHT), Image.Resampling.LANCZOS)
            img_array = self.app.np.array(img_resized)
            
            if img_array.shape[-1] == 4:
                img_array = img_array[..., :3]
            
            img_array = img_array.astype('float32') / 255.0
            img_batch = self.app.np.expand_dims(img_array, axis=0)

            # Inference
            predictions = None
            if self.app.model_type == 'tflite':
                self.app.interpreter.set_tensor(self.app.input_details[0]['index'], img_batch)
                self.app.interpreter.invoke()
                predictions = self.app.interpreter.get_tensor(self.app.output_details[0]['index'])[0]
            elif self.app.model_type == 'keras':
                predictions = self.app.model.predict(img_batch, verbose=0)[0]

            inf_time_ms = (time.time() - start_time) * 1000
            
            # Convert to float for UI safety
            predictions = predictions.astype(float)
            
            self.after(0, lambda: self._update_ui_results(predictions, inf_time_ms))
            
        except Exception as e:
            print(f"Inference Error: {e}")
            self.after(0, lambda: self.status.configure(text=f"Error: {e}"))

    def _update_ui_results(self, probs, time_ms):
        # Clear previous results
        for widget in self.res_frame.winfo_children():
            widget.destroy()

        top_k = 5
        top_indices = probs.argsort()[-top_k:][::-1]
        
        best_idx = top_indices[0]
        best_label_key = self.app.classes[best_idx]
        best_label_name = self.app.mapping.get(best_label_key, best_label_key)
        best_conf = probs[best_idx]

        # Colors
        # Colors
        color = "#e74c3c" if best_conf < 0.6 else "#f39c12" if best_conf < 0.85 else "#27ae60"

        # Localization
        from multi_cancer_ai.src.localization import TRANSLATIONS
        t = TRANSLATIONS[self.app.lang]

        # 1. Main Result
        # Extract "Result: {}" -> Split to get label
        res_label = t["diag_result"].split(":")[0] 
        lbl = ctk.CTkLabel(self.res_frame, text=f"{res_label}:", font=ctk.CTkFont(size=14))
        lbl.pack(anchor="w")
        
        lbl_res = ctk.CTkLabel(self.res_frame, text=f"{best_label_name}", font=ctk.CTkFont(size=22, weight="bold"), text_color=color)
        lbl_res.pack(anchor="w")
        
        ctk.CTkLabel(self.res_frame, text=t["diag_confidence"].format(best_conf*100)).pack(anchor="w")
        ctk.CTkLabel(self.res_frame, text=t["diag_time"].format(time_ms/1000)).pack(anchor="w")

        ctk.CTkFrame(self.res_frame, height=2, fg_color="gray").pack(fill="x", pady=10)

        # 2. Probability Bars
        for idx in top_indices:
            key = self.app.classes[idx]
            name = self.app.mapping.get(key, key)
            p = probs[idx]
            
            row = ctk.CTkFrame(self.res_frame, fg_color="transparent")
            row.pack(fill="x", pady=2)
            
            ctk.CTkLabel(row, text=name[:20], width=100, anchor="w", font=ctk.CTkFont(size=11)).pack(side="left")
            prog = ctk.CTkProgressBar(row, height=8)
            prog.pack(side="left", fill="x", expand=True, padx=5)
            prog.set(p)
            ctk.CTkLabel(row, text=f"{p:.1%}", width=40, font=ctk.CTkFont(size=11)).pack(side="left")

        # 3. Training Prompt
        ctk.CTkFrame(self.res_frame, height=2, fg_color="gray").pack(fill="x", pady=10)
        
        btn_wrong = ctk.CTkButton(self.res_frame, text=t["diag_wrong_btn"], fg_color="#c0392b", 
                                  command=lambda: self._goto_training(best_label_name))
        btn_wrong.pack(pady=5)
        
        self.status.configure(text=f"Analysis Done ({time_ms:.0f}ms)")
        
        # Store prediction for Training Tab
        self.predictions = probs
        
    def _goto_training(self, expert_label):
        # Switch to Training Tab
        self.app.show_frame("Training")
        # Update Training Tab UI
        train_frame = self.app.frames["Training"]
        train_frame.lbl_pred.configure(text=f"AI Prediction: {expert_label}")

class TrainingFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, corner_radius=0, fg_color="transparent")
        self.app = master
        self.controller = master # Alias for consistency
        
        self.lbl_title = ctk.CTkLabel(self, text="Active Training Mode", font=ctk.CTkFont(size=20, weight="bold"))
        self.lbl_title.pack(pady=20)
        
        self.lbl_instr = ctk.CTkLabel(self, text="Teach the AI...", font=ctk.CTkFont(size=12))
        self.lbl_instr.pack(pady=5)
        
        self.img_label = ctk.CTkLabel(self, text="[No Image Selected in Diagnosis]", width=300, height=300, fg_color="gray20", corner_radius=10)
        self.img_label.pack(pady=20)
        
        self.lbl_pred = ctk.CTkLabel(self, text="AI Prediction: N/A", font=ctk.CTkFont(weight="bold"))
        self.lbl_pred.pack(pady=5)
        
        self.correction_frame = ctk.CTkFrame(self)
        self.correction_frame.pack(pady=10, fill="x", padx=50)
        
        self.lbl_select = ctk.CTkLabel(self.correction_frame, text="Select Correct Class:")
        self.lbl_select.pack(pady=5)
        
        self.class_var = ctk.StringVar(value=self.app.classes[0])
        self.combo = ctk.CTkOptionMenu(self.correction_frame, values=self.app.classes, variable=self.class_var)
        self.combo.pack(pady=5)
        
        self.btn_train = ctk.CTkButton(self.correction_frame, text="Confirm & Train", fg_color="green", command=self.run_training)
        self.btn_train.pack(pady=20)
        
        self.status_lbl = ctk.CTkLabel(self, text="")
        self.status_lbl.pack(pady=5)
        
        # Init Language
        self.update_language(self.app.lang)

    def update_language(self, lang):
        from multi_cancer_ai.src.localization import TRANSLATIONS
        t = TRANSLATIONS[lang]
        
        self.lbl_title.configure(text=t["train_title"])
        self.lbl_instr.configure(text=t["train_instr"])
        self.lbl_select.configure(text=t["train_select_label"])
        self.btn_train.configure(text=t["train_confirm_btn"])

    def run_training(self):
        # 1. Get image from DiagnosisFrame
        diag_frame = self.app.frames["Diagnosis"]
        if not diag_frame.current_pil_img:
            self.status_lbl.configure(text="Error: No image loaded in Diagnosis tab.", text_color="red")
            return
            
        # 2. Get Label
        label_str = self.class_var.get()
        label_idx = self.app.classes.index(label_str)

        # Confirm Action
        if not messagebox.askyesno("Confirm Training", 
            f"Are you sure you want to teach the AI that this image is:\n\n'{label_str}'?\n\nThis will modify the model weights."):
            self.status_lbl.configure(text="Training cancelled.", text_color="gray")
            return
        
        # 3. Preprocess
        img = diag_frame.current_pil_img.resize((config.IMG_WIDTH, config.IMG_HEIGHT), Image.Resampling.LANCZOS)
        img_arr = self.app.np.array(img).astype('float32') / 255.0
        
        # 4. Train
        if not self.app.active_trainer.model:
            success, msg = self.app.active_trainer.load_model_for_training()
            if not success:
                self.status_lbl.configure(text=msg, text_color="red")
                return

        self.status_lbl.configure(text="Training in progress... (Please wait)", text_color="orange")
        self.app.update()
        
        # Run in thread ideally, but for single image fit() is fast enough to block briefly
        # Or better -> Update App state
        success, msg = self.app.active_trainer.train_on_single_image(img_arr, label_idx)
        
        if success:
            self.app.active_trainer.save_labeled_image(diag_frame.current_img_path, label_str)
            self.status_lbl.configure(text=f"Success! {msg}", text_color="green")
            # Force Keras model usage in App for next inference to see improvement
            self.app.model_type = 'keras'
            self.app.model = self.app.active_trainer.model 
        else:
            self.status_lbl.configure(text=f"Error: {msg}", text_color="red")

# Configuration for CustomTkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Legacy code removed

if __name__ == "__main__":
    app = CancerDiagApp()
    app.mainloop()
