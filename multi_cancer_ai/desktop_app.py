import os
import sys
from pathlib import Path
import time
import json
import threading

# -----------------------------------------------------------------------------
# CRITICAL: SET ENV VARS BEFORE ANY OTHER HEAVY IMPORT
# -----------------------------------------------------------------------------
# Disable TensorFlow Metal/GPU to prevent SegFaults with Tkinter on macOS
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # Optional: disable oneDNN if conflicting

# NOW import GUI libs
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Setup path to handle imports correctly
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from multi_cancer_ai.config import config
from multi_cancer_ai.src.active_trainer import ActiveTrainer
from multi_cancer_ai.src.evaluator import GradCAM

# -----------------------------------------------------------------------------
# Security / robustness defaults for local desktop usage
# -----------------------------------------------------------------------------
# Anche se l'app è “solo desktop”, una validazione minima dell'input evita crash,
# freeze (file enormi) o immagini corrotte.
_ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_MAX_IMAGE_BYTES = 25 * 1024 * 1024  # 25MB: limite conservativo per UI reattiva

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

        # Trigger model load in the Diagnosis Frame
        # self.frames["Diagnosis"].load_model_backend() -> Do NOT call this directly from thread if it touches UI
        
        # Schedule the loading via the DiagnosisFrame (but DiagnosisFrame needs to split UI/Logic)
        # Actually, let's just make DiagnosisFrame expose a method that STARTS the thread, or handles the logic
        
        # Correct pattern:
        # App calls DiagnosisFrame.start_backend_loading() on MAIN THREAD
        # DiagnosisFrame.start_backend_loading() updates UI -> Starts Thread -> Thread does work -> Callback to UI
        
        # So here in _load_backend_worker (which is ALREADY a thread), we should purely do the IMPORTS 
        # that are shared, and then perhaps schedule the next step?
        
        # But wait, imports can be heavy.
        
        # Let's pivot:
        # 1. _initial_backend_load (Main Thread) -> Calls DiagnosisFrame.start_loading()
        # 2. DiagnosisFrame.start_loading() (Main Thread) -> Updates UI -> Starts NEW Thread for Model
        
        # The current `_load_backend_worker` is just importing numpy/cv2.
        # Let's change this structure.
        pass

    def _initial_backend_load(self):
        # Delegate everything to the explicit loading flow
        self.frames["Diagnosis"].start_model_loading()


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
    
    def update_model_status(self, status_text: str, model_name: str = "", is_ready: bool = False):
        """
        Aggiorna lo stato del modello nel HomeFrame.
        
        Args:
            status_text: Testo dello stato (es. "Model loaded", "Checking model...")
            model_name: Nome del modello caricato (opzionale)
            is_ready: Se True, mostra in verde, altrimenti in arancione
        """
        self.lbl_model_status.configure(
            text=status_text,
            text_color="green" if is_ready else "orange"
        )
        
        if model_name:
            self.lbl_model_name.configure(text=f"Model: {model_name}")
        else:
            self.lbl_model_name.configure(text="")

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
        
        # Clear image button
        self.btn_clear_img_diag = ctk.CTkButton(
            self.ctrl_frame, 
            text="Clear Image", 
            command=self.clear_image_diagnosis, 
            fg_color="red", 
            hover_color="darkred",
            state="disabled"
        )
        self.btn_clear_img_diag.pack(side="left", padx=5, pady=10)
        
        # Reload model button (always available in Diagnosis tab)
        self.btn_reload_model_diag = ctk.CTkButton(
            self.ctrl_frame, 
            text="🔄 Reload Model", 
            command=self.reload_model_diagnosis, 
            fg_color="blue", 
            hover_color="darkblue"
        )
        self.btn_reload_model_diag.pack(side="left", padx=10, pady=10)
        
        self.status = ctk.CTkLabel(self.ctrl_frame, text="System Init...")
        self.status.pack(side="left", padx=20)
        
        # Heatmap Switch
        self.heatmap_var = ctk.BooleanVar(value=False)
        self.switch_heatmap = ctk.CTkSwitch(self.ctrl_frame, text="Show Heatmap", variable=self.heatmap_var, command=self._toggle_heatmap)
        self.switch_heatmap.pack(side="right", padx=20)

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
        self.btn_clear_img_diag.configure(text=t["diag_clear_btn"])
        self.btn_reload_model_diag.configure(text=t["diag_reload_model_btn"])
        self.res_frame.configure(label_text=t["diag_result"].format("..."))
        self.btn_wrong.configure(text=t["diag_wrong_btn"])
        self.switch_heatmap.configure(text=t.get("diag_heatmap", "Heatmap"))
    
    def clear_image_diagnosis(self):
        """Rimuove l'immagine corrente dal tab Diagnostica."""
        self.current_img_path = None
        self.current_pil_img = None
        self.predictions = None
        self.heatmap_var.set(False)
        
        # Reset canvas
        self.canvas.configure(image=None, text="No Image")
        self.canvas.image = None
        
        # Clear results
        for widget in self.res_frame.winfo_children():
            widget.destroy()
        
        # Disabilita pulsante clear
        self.btn_clear_img_diag.configure(state="disabled")
        
        self.status.configure(text="Image cleared")

    def _on_wrong_diag(self):
        # Switch to training tab and pass current image
        if self.current_img_path and self.current_pil_img:
            self.app.frames["Training"].set_training_image(self.current_img_path, self.current_pil_img)
            self.app.show_frame("Training")
        else:
            messagebox.showwarning("No Image", "Please load an image in the Diagnosis tab first.")

    def _toggle_heatmap(self):
        if not self.current_pil_img:
            return
            
        if self.heatmap_var.get():
            # Turn ON
            self._compute_and_show_heatmap()
        else:
            # Turn OFF - Restore original
            self._show_image_pil(self.current_pil_img)

    def _compute_and_show_heatmap(self):
        if self.app.model_type != 'keras':
            messagebox.showwarning("Feature Not Available", "Heatmap requires the full Keras model.\nCurrent mode: " + str(self.app.model_type))
            self.heatmap_var.set(False)
            return

        threading.Thread(target=self._heatmap_worker, daemon=True).start()

    def _heatmap_worker(self):
        try:
            # Re-prepare image
            img_resized = self.current_pil_img.resize((config.IMG_WIDTH, config.IMG_HEIGHT), Image.Resampling.LANCZOS)
            img_array = self.app.np.array(img_resized)
            if img_array.shape[-1] == 4: img_array = img_array[..., :3]
            img_array = img_array.astype('float32') / 255.0
            img_batch = self.app.np.expand_dims(img_array, axis=0)

            # Compute Heatmap
            grad_cam = GradCAM(self.app.model)
            heatmap = grad_cam.compute_heatmap(img_batch)
            
            # Overlay
            # We need the original image as array for overlay, but scaled
            # GradCAM returns 224x224 heatmap usually
            
            # We want to display high-res overlay if possible, but GradCAM output is low res.
            # Best is to overlay on the 224x224 and then maybe upscale for display? 
            # Or overlay on the original PIL image?
            
            # Let's use the low-res overlay for now as it's easier with cv2
            # Evaluator.overlay_heatmap takes (heatmap, original_image)
            # original_image should be uint8 0-255
            
            original_cv = (img_array * 255).astype('uint8')
            overlay = grad_cam.overlay_heatmap(heatmap, original_cv, alpha=0.4)
            
            # Convert back to PIL for TKinter
            overlay_pil = Image.fromarray(overlay)
            
            self.after(0, lambda: self._show_image_pil(overlay_pil))
            
        except Exception as e:
            print(f"Heatmap Error: {e}")
            self.after(0, lambda: messagebox.showerror("Error", f"Heatmap failed: {e}"))
            self.after(0, lambda: self.heatmap_var.set(False))

    def _show_image_pil(self, pil_img):
        # Resize for display
        display_img = pil_img.copy()
        display_img.thumbnail((400, 400), Image.Resampling.LANCZOS)
        tk_img = ctk.CTkImage(light_image=display_img, dark_image=display_img, size=display_img.size)
        self.canvas.configure(image=tk_img, text="")
        self.canvas.image = tk_img
    
    def start_model_loading(self):
        # 1. UI Update (Main Thread)
        self.status.configure(text="Loading Model... (Sync)")
        self.progress.pack(side="left", padx=10)
        self.progress.start()
        
        # Force UI update so the user sees the spinner before we freeze
        self.app.update()
        
        # 2. Run Synchronously (No Threading on macOS to avoid SegFault)
        # We use a slight delay to ensure UI is rendered first
        self.after(100, self._load_model_sync)

    def _load_model_sync(self):
        try:
            # Shared Libs Import
            import numpy as np
            import cv2
            self.app.np = np
            self.app.cv2 = cv2
            
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

            # 3. Base Keras Model (Standard CPU)
            # We prioritize this now because TFLite is causing SegFaults on macOS with GUI
            print("[INFO] Loading Keras model (Base)...")
            import tensorflow as tf
            if keras_path.exists():
                self.app.model = tf.keras.models.load_model(str(keras_path))
                self.app.model_type = 'keras'
                self._on_model_loaded("Keras Model (Base)")
                return

            # 4. Optimization: Try TFLite (Last Resort)
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

            # 5. Last Fallback if everything fails
            raise FileNotFoundError("No suitable model found (checked Custom, Tuned, Keras, TFLite)")

        except Exception as e:
            print(f"[ERROR] Init Failed: {e}")
            # Schedule Error UI Update
            self.after(0, lambda: self._on_loading_error(str(e)))
    
    def _on_loading_error(self, err_msg):
        self.status.configure(text=f"Error: {err_msg}", text_color="red")
        self.progress.stop()
        self.progress.pack_forget()
        
        # Aggiorna anche il HomeFrame
        home_frame = self.app.frames.get("Home")
        if home_frame:
            self.after(0, lambda: home_frame.update_model_status(
                f"❌ Error: {err_msg[:50]}...",
                "",
                is_ready=False
            ))

    def _on_model_loaded(self, msg):
        self.app.trainer_loaded = True
        # Schedule UI update on main thread
        self.after(0, lambda: self._update_status_ready(msg))
        
        # Aggiorna anche il HomeFrame
        home_frame = self.app.frames.get("Home")
        if home_frame:
            # Estrai il nome del modello dal messaggio
            model_name = msg
            if "Custom" in msg:
                model_name = "Custom Model (User Corrections)"
            elif "Tuned" in msg:
                model_name = "Fine-Tuned Model"
            elif "Base" in msg:
                model_name = "Base Model"
            elif "TFLite" in msg:
                model_name = "TFLite Model"
            else:
                model_name = msg
            
            self.after(0, lambda: home_frame.update_model_status(
                "✅ Model Ready",
                model_name,
                is_ready=True
            ))

    def _update_status_ready(self, msg):
        self.status.configure(text=f"Ready: {msg}", text_color="green")
        self.progress.stop()
        self.progress.pack_forget()

    def load_image(self):
        try:
            path = filedialog.askopenfilename(
                title="Select Scan",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff")],
            )
            if path:
                # --- Input validation ---
                p = Path(path)
                if p.suffix.lower() not in _ALLOWED_EXTENSIONS:
                    messagebox.showerror("Unsupported file", f"Unsupported extension: {p.suffix}\nAllowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}")
                    return

                try:
                    size = p.stat().st_size
                    if size > _MAX_IMAGE_BYTES:
                        messagebox.showerror("File too large", f"File size is {size/1024/1024:.1f}MB.\nMax allowed: {_MAX_IMAGE_BYTES/1024/1024:.0f}MB.")
                        return
                except Exception:
                    # Se non riusciamo a leggere la size, continuiamo e lasciamo gestire a PIL.
                    pass

                self.current_img_path = path
                self._show_image(path)
                self._run_inference(path)
        except Exception as e:
            print(f"Error: {e}")

    def _show_image(self, path):
        # Verifica base integrità/decodifica: `verify()` intercetta molti file corrotti.
        try:
            with Image.open(path) as _im:
                _im.verify()
        except Exception as e:
            messagebox.showerror("Invalid image", f"Could not read image.\nReason: {e}")
            return

        img = Image.open(path)
        self.current_pil_img = img
        self.heatmap_var.set(False) # Reset heatmap
        self._show_image_pil(img)
        
        # Abilita pulsante clear
        self.btn_clear_img_diag.configure(state="normal")
        


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
    
    def reload_model_diagnosis(self):
        """
        Ricarica il modello custom dal Diagnosis tab.
        Utile per aggiornare il modello dopo active training senza tornare al Training tab.
        """
        try:
            self.status.configure(text="Reloading model...", text_color="orange")
            self.app.update()
            
            # Aggiorna HomeFrame
            home_frame = self.app.frames.get("Home")
            if home_frame:
                home_frame.update_model_status("Reloading model...", "", is_ready=False)
            
            # Ricarica il modello custom se esiste, altrimenti usa il base
            custom_model_path = config.MODELS_DIR / "best_model_custom.h5"
            base_model_path = config.MODELS_DIR / "best_model.h5"
            
            import tensorflow as tf
            
            if custom_model_path.exists():
                self.app.model = tf.keras.models.load_model(str(custom_model_path))
                self.app.model_type = 'keras'
                self.status.configure(text="✅ Model Reloaded (Custom)", text_color="green")
                
                # Aggiorna HomeFrame
                if home_frame:
                    home_frame.update_model_status(
                        "✅ Model Ready",
                        "Custom Model (User Corrections)",
                        is_ready=True
                    )
                
                messagebox.showinfo("Model Reloaded", "Custom model (with your corrections) has been loaded.")
            elif base_model_path.exists():
                self.app.model = tf.keras.models.load_model(str(base_model_path))
                self.app.model_type = 'keras'
                self.status.configure(text="✅ Model Reloaded (Base)", text_color="green")
                
                # Aggiorna HomeFrame
                if home_frame:
                    home_frame.update_model_status(
                        "✅ Model Ready",
                        "Base Model",
                        is_ready=True
                    )
                
                messagebox.showinfo("Model Reloaded", "Base model has been loaded.\nNo custom model found.")
            else:
                messagebox.showwarning("Model Not Found", "No model found. Please run training first.")
                self.status.configure(text="No model found", text_color="red")
                
                # Aggiorna HomeFrame
                if home_frame:
                    home_frame.update_model_status(
                        "❌ No Model Found",
                        "",
                        is_ready=False
                    )
                
        except Exception as e:
            import logging
            logger = logging.getLogger("MultiCancerAI")
            logger.error(f"Error reloading model: {e}", exc_info=True)
            
            self.status.configure(text=f"Error: {str(e)}", text_color="red")
            
            # Aggiorna HomeFrame
            home_frame = self.app.frames.get("Home")
            if home_frame:
                home_frame.update_model_status(
                    f"❌ Error: {str(e)[:50]}...",
                    "",
                    is_ready=False
                )
            
            messagebox.showerror("Reload Error", f"Failed to reload model:\n{e}")
        
    def _goto_training(self, expert_label):
        # Switch to Training Tab and pass image if available
        train_frame = self.app.frames["Training"]
        
        # Pass image if we have it
        if self.current_img_path and self.current_pil_img:
            train_frame.set_training_image(self.current_img_path, self.current_pil_img)
        
        # Update prediction label
        train_frame.lbl_pred.configure(text=f"AI Prediction: {expert_label}")
        
        # Switch to training tab
        self.app.show_frame("Training")

class TrainingFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, corner_radius=0, fg_color="transparent")
        self.app = master
        self.controller = master # Alias for consistency
        
        # State variables for image
        self.training_img_path = None
        self.training_pil_img = None
        
        self.lbl_title = ctk.CTkLabel(self, text="Active Training Mode", font=ctk.CTkFont(size=20, weight="bold"))
        self.lbl_title.pack(pady=20)
        
        self.lbl_instr = ctk.CTkLabel(self, text="Teach the AI...", font=ctk.CTkFont(size=12))
        self.lbl_instr.pack(pady=5)
        
        # Image display frame
        self.img_frame = ctk.CTkFrame(self)
        self.img_frame.pack(pady=20)
        
        self.img_label = ctk.CTkLabel(self.img_frame, text="[No Image Selected]", width=300, height=300, fg_color="gray20", corner_radius=10)
        self.img_label.pack(pady=10, padx=10)
        
        # Buttons frame for image management
        self.img_buttons_frame = ctk.CTkFrame(self.img_frame)
        self.img_buttons_frame.pack(pady=5)
        
        # Button to load image manually
        self.btn_load_img = ctk.CTkButton(self.img_buttons_frame, text="Load Image", command=self.load_image_manual, fg_color="gray40", hover_color="gray50")
        self.btn_load_img.pack(side="left", padx=5)
        
        # Button to clear image
        self.btn_clear_img = ctk.CTkButton(self.img_buttons_frame, text="Clear Image", command=self.clear_image, fg_color="red", hover_color="darkred", state="disabled")
        self.btn_clear_img.pack(side="left", padx=5)
        
        self.lbl_pred = ctk.CTkLabel(self, text="AI Prediction: N/A", font=ctk.CTkFont(weight="bold"))
        self.lbl_pred.pack(pady=5)
        
        self.correction_frame = ctk.CTkFrame(self)
        self.correction_frame.pack(pady=10, fill="x", padx=50)
        
        self.lbl_select = ctk.CTkLabel(self.correction_frame, text="Select Correct Class:")
        self.lbl_select.pack(pady=5)
        
        self.class_var = ctk.StringVar(value=self.app.classes[0] if self.app.classes else "")
        self.combo = ctk.CTkOptionMenu(self.correction_frame, values=self.app.classes, variable=self.class_var)
        self.combo.pack(pady=5)
        
        self.btn_train = ctk.CTkButton(self.correction_frame, text="Confirm & Train", fg_color="green", command=self.run_training)
        self.btn_train.pack(pady=20)
        
        # Reload model button (appears after successful training)
        self.btn_reload_model = ctk.CTkButton(
            self.correction_frame, 
            text="🔄 Reload Model (After Training)", 
            command=self.reload_model, 
            fg_color="blue", 
            hover_color="darkblue",
            state="disabled"  # Enabled after training
        )
        self.btn_reload_model.pack(pady=10)
        
        self.status_lbl = ctk.CTkLabel(self, text="")
        self.status_lbl.pack(pady=5)
        
        # Init Language
        self.update_language(self.app.lang)
    
    def load_image_manual(self):
        """Permette di caricare manualmente un'immagine nel tab Training."""
        try:
            path = filedialog.askopenfilename(
                title="Select Image for Training",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff")],
            )
            
            if not path:
                # Utente ha annullato la selezione
                return
            
            if not path.strip():
                messagebox.showerror("Error", "No file selected.")
                return
            
            # Validazione (stessa logica di DiagnosisFrame)
            try:
                p = Path(path)
            except Exception as e:
                messagebox.showerror("Invalid Path", f"Invalid file path:\n{e}")
                return
            
            # Verifica estensione
            if p.suffix.lower() not in _ALLOWED_EXTENSIONS:
                messagebox.showerror(
                    "Unsupported file", 
                    f"Unsupported extension: {p.suffix}\nAllowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
                )
                return
            
            # Verifica esistenza file
            if not p.exists():
                messagebox.showerror("File Not Found", f"File does not exist:\n{path}")
                return
            
            # Verifica dimensione
            try:
                size = p.stat().st_size
                if size > _MAX_IMAGE_BYTES:
                    messagebox.showerror(
                        "File too large", 
                        f"File size is {size/1024/1024:.1f}MB.\nMax allowed: {_MAX_IMAGE_BYTES/1024/1024:.0f}MB."
                    )
                    return
            except OSError as e:
                messagebox.showerror("File Error", f"Cannot read file size:\n{e}")
                return
            
            # Carica e verifica immagine
            try:
                # Prima verifica che sia un'immagine valida
                with Image.open(path) as test_img:
                    test_img.verify()
            except Exception as e:
                messagebox.showerror("Invalid Image", f"Could not read image file:\n{str(e)}\n\nFile may be corrupted or not a valid image.")
                return
            
            # Ora carica l'immagine per l'uso (verify() chiude il file, quindi riapriamo)
            try:
                img = Image.open(path)
                # Converti in RGB se necessario (alcuni formati come PNG con trasparenza)
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Crea un'immagine RGB con sfondo bianco
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = rgb_img
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Imposta l'immagine
                self.set_training_image(str(path), img)
                
            except Exception as e:
                import logging
                logger = logging.getLogger("MultiCancerAI")
                logger.error(f"Error loading image in TrainingFrame: {e}", exc_info=True)
                messagebox.showerror("Load Error", f"Failed to load image:\n{str(e)}")
                
        except Exception as e:
            import logging
            logger = logging.getLogger("MultiCancerAI")
            logger.error(f"Unexpected error in load_image_manual: {e}", exc_info=True)
            messagebox.showerror("Error", f"Unexpected error loading image:\n{str(e)}")
    
    def clear_image(self):
        """Rimuove l'immagine corrente dal Training tab."""
        self.training_img_path = None
        self.training_pil_img = None
        self.img_label.configure(image=None, text="[No Image Selected]")
        self.img_label.image = None
        self.btn_clear_img.configure(state="disabled")
    
    def set_training_image(self, img_path: str, pil_img: Image.Image):
        """
        Imposta l'immagine da usare per il training.
        
        Args:
            img_path: Percorso del file immagine.
            pil_img: Oggetto PIL Image già caricato.
        """
        self.training_img_path = img_path
        self.training_pil_img = pil_img
        
        # Mostra l'immagine ridimensionata
        display_img = pil_img.copy()
        display_img.thumbnail((300, 300), Image.Resampling.LANCZOS)
        tk_img = ctk.CTkImage(light_image=display_img, dark_image=display_img, size=display_img.size)
        self.img_label.configure(image=tk_img, text="")
        self.img_label.image = tk_img  # Mantieni riferimento per evitare garbage collection
        
        # Abilita pulsante clear
        self.btn_clear_img.configure(state="normal")

    def update_language(self, lang):
        from multi_cancer_ai.src.localization import TRANSLATIONS
        t = TRANSLATIONS[lang]
        
        self.lbl_title.configure(text=t["train_title"])
        self.lbl_instr.configure(text=t["train_instr"])
        self.btn_load_img.configure(text=t["train_load_img_btn"])
        self.btn_clear_img.configure(text=t["train_clear_img_btn"])
        self.lbl_select.configure(text=t["train_select_label"])
        self.btn_train.configure(text=t["train_confirm_btn"])
        self.btn_reload_model.configure(text=t["train_reload_model_btn"])

    def run_training(self):
        # 1. Get image - prefer training frame image, fallback to diagnosis frame
        img_to_use = None
        img_path_to_use = None
        
        if self.training_pil_img:
            # Usa immagine caricata direttamente nel Training tab
            img_to_use = self.training_pil_img
            img_path_to_use = self.training_img_path
        else:
            # Fallback: prova a prendere da Diagnosis tab
            diag_frame = self.app.frames["Diagnosis"]
            if diag_frame.current_pil_img:
                img_to_use = diag_frame.current_pil_img
                img_path_to_use = diag_frame.current_img_path
            else:
                self.status_lbl.configure(
                    text="Error: No image loaded. Please load an image first.", 
                    text_color="red"
                )
                return
        
        if not img_to_use:
            self.status_lbl.configure(
                text="Error: No image available for training.", 
                text_color="red"
            )
            return
            
        # 2. Get Label
        label_str = self.class_var.get()
        if label_str not in self.app.classes:
            self.status_lbl.configure(text="Error: Invalid class selected.", text_color="red")
            return
        
        label_idx = self.app.classes.index(label_str)

        # Confirm Action
        from multi_cancer_ai.src.localization import TRANSLATIONS
        t = TRANSLATIONS[self.app.lang]
        
        if not messagebox.askyesno(t["train_confirm_dialog"], t["train_confirm_msg"].format(label_str)):
            self.status_lbl.configure(text="Training cancelled.", text_color="gray")
            return
        
        # 3. Preprocess
        img = img_to_use.resize((config.IMG_WIDTH, config.IMG_HEIGHT), Image.Resampling.LANCZOS)
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
            # Salva immagine etichettata (usa il path corretto)
            if img_path_to_use:
                self.app.active_trainer.save_labeled_image(img_path_to_use, label_str)
            
            from multi_cancer_ai.src.localization import TRANSLATIONS
            t = TRANSLATIONS[self.app.lang]
            
            # Messaggio informativo
            info_msg = ""
            if self.app.lang == "it":
                info_msg = (
                    f"✅ Training completato! {msg}\n\n"
                    "⚠️ IMPORTANTE:\n"
                    "1. Clicca 'Ricarica Modello' per usare il modello aggiornato\n"
                    "2. Il training su una singola immagine può richiedere più sessioni per vedere miglioramenti significativi\n"
                    "3. Per risultati migliori, correggi più immagini della stessa classe"
                )
            else:
                info_msg = (
                    f"✅ Training completed! {msg}\n\n"
                    "⚠️ IMPORTANT:\n"
                    "1. Click 'Reload Model' to use the updated model\n"
                    "2. Training on a single image may require multiple sessions to see significant improvements\n"
                    "3. For better results, correct more images of the same class"
                )
            
            self.status_lbl.configure(text=info_msg, text_color="green")
            
            # Abilita pulsante reload model
            self.btn_reload_model.configure(state="normal")
            
            # NOTA: Non ricarichiamo automaticamente il modello perché potrebbe essere pesante.
            # L'utente deve cliccare esplicitamente "Reload Model" quando è pronto.
        else:
            self.status_lbl.configure(text=f"Error: {msg}", text_color="red")
    
    def reload_model(self):
        """
        Ricarica il modello custom aggiornato dopo l'active training.
        Questo permette di usare il modello migliorato per nuove inferenze.
        """
        try:
            self.status_lbl.configure(text="Reloading model...", text_color="orange")
            self.app.update()
            
            # Aggiorna HomeFrame
            home_frame = self.app.frames.get("Home")
            if home_frame:
                home_frame.update_model_status("Reloading model...", "", is_ready=False)
            
            # Ricarica il modello custom se esiste
            custom_model_path = config.MODELS_DIR / "best_model_custom.h5"
            
            if not custom_model_path.exists():
                messagebox.showwarning(
                    "Model Not Found", 
                    "Custom model not found. Make sure you've completed at least one training session."
                )
                self.status_lbl.configure(text="No custom model found.", text_color="red")
                
                # Aggiorna HomeFrame
                if home_frame:
                    home_frame.update_model_status(
                        "❌ No Custom Model Found",
                        "",
                        is_ready=False
                    )
                return
            
            # Carica il modello
            import tensorflow as tf
            self.app.model = tf.keras.models.load_model(str(custom_model_path))
            self.app.model_type = 'keras'
            
            # Aggiorna anche il DiagnosisFrame per usare il nuovo modello
            diag_frame = self.app.frames["Diagnosis"]
            if hasattr(diag_frame, 'status'):
                diag_frame.status.configure(
                    text="✅ Model Reloaded (Custom)", 
                    text_color="green"
                )
            
            # Aggiorna HomeFrame
            if home_frame:
                home_frame.update_model_status(
                    "✅ Model Ready",
                    "Custom Model (User Corrections)",
                    is_ready=True
                )
            
            from multi_cancer_ai.src.localization import TRANSLATIONS
            t = TRANSLATIONS[self.app.lang]
            
            self.status_lbl.configure(
                text=t["train_reload_success"], 
                text_color="green"
            )
            
            # Disabilita il pulsante dopo il reload (può essere riabilitato dopo nuovo training)
            self.btn_reload_model.configure(state="disabled")
            
            messagebox.showinfo(
                t["train_confirm_dialog"].replace("Training", "Model"), 
                t["train_reload_success"]
            )
            
        except Exception as e:
            import logging
            logger = logging.getLogger("MultiCancerAI")
            logger.error(f"Error reloading model: {e}", exc_info=True)
            
            self.status_lbl.configure(
                text=f"Error reloading model: {str(e)}", 
                text_color="red"
            )
            
            # Aggiorna HomeFrame
            home_frame = self.app.frames.get("Home")
            if home_frame:
                home_frame.update_model_status(
                    f"❌ Error: {str(e)[:50]}...",
                    "",
                    is_ready=False
                )
            
            messagebox.showerror("Reload Error", f"Failed to reload model:\n{e}")

# Configuration for CustomTkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Legacy code removed

if __name__ == "__main__":
    app = CancerDiagApp()
    app.mainloop()
