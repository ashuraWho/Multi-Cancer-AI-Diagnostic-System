"""
Modulo per la valutazione del modello e interpretabilità (Grad-CAM).
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import cv2

class Evaluator:
    """Classe per calcolare metriche e generare visualizzazioni."""
    
    def __init__(self, model, validation_gen, class_indices):
        self.model = model
        self.validation_gen = validation_gen
        self.class_indices = class_indices
        # Reverse mapping: indice -> nome classe
        self.idx_to_class = {v: k for k, v in class_indices.items()}
        self.class_names = list(class_indices.keys())

    def evaluate(self):
        """
        Valuta il modello sul validation set e restituisce le metriche.
        Supporta tf.data.Dataset iterating to extract true labels.
        """
        print("Calcolo predizioni per evaluation...")
        
        y_true = []
        y_pred = []
        
        # Iterate over the dataset to get true labels and predictions batch by batch
        # This is more robust for tf.data than model.predict which might lose alignment if not careful,
        # although model.predict(ds) is usually fine. But we need y_true.
        
        for batch_images, batch_labels in self.validation_gen:
            # Predict batch
            preds = self.model.predict_on_batch(batch_images)
            
            # Extract labels
            batch_true = np.argmax(batch_labels.numpy(), axis=1)
            batch_pred = np.argmax(preds, axis=1)
            
            y_true.extend(batch_true)
            y_pred.extend(batch_pred)
            
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        return y_true, y_pred, None # Raw predictions not strictly needed for basic report

    def generate_report(self, y_true, y_pred):
        """
        Genera un report di classificazione testuale.
        """
        report = classification_report(y_true, y_pred, target_names=self.class_names)
        return report

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        Crea e salva la matrice di confusione.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        plt.title('Matrice di Confusione')
        plt.ylabel('Vero')
        plt.xlabel('Predetto')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.close()

class GradCAM:
    """Gradient-weighted Class Activation Mapping per interpretabilità."""
    
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()

    def _find_target_layer(self):
        """Trova l'ultimo livello convoluzionale 4D."""
        # Cerca nel modello base (EfficientNet è spesso un layer annidato o il modello stesso)
        # Se il modello è quello costruito in model.py, il base_model è efficientnetv2-b3
        for layer in reversed(self.model.layers):
            try:
                # Try getting output shape from the layer's output tensor
                if hasattr(layer, 'output'):
                    output_shape = layer.output.shape
                else:
                    output_shape = getattr(layer, 'output_shape', None)
            except (AttributeError, RuntimeError):
                output_shape = None

            if output_shape is not None and len(output_shape) == 4:
                return layer.name
            
            # Se è un modello funzionale annidato (es. transfer learning)
            if hasattr(layer, 'layers'):
                for sub_layer in reversed(layer.layers):
                     try:
                        if hasattr(sub_layer, 'output'):
                            sub_output_shape = sub_layer.output.shape
                        else:
                            sub_output_shape = getattr(sub_layer, 'output_shape', None)
                     except (AttributeError, RuntimeError):
                        sub_output_shape = None

                     if sub_output_shape is not None and len(sub_output_shape) == 4:
                        return layer.name # Ritorniamo il nome del container per accedere all'output
        
        # Fallback specifico per EfficientNetV2B3
        return 'top_activation' # Nome tipico ultimo layer attivazione in EffNet

    def compute_heatmap(self, image_array, class_idx=None, eps=1e-8):
        """
        Calcola la heatmap Grad-CAM per un'immagine e una classe specifica.
        """
        # Creiamo un modello che mappa l'input alle attivazioni dell'ultimo conv layer
        # e all'output delle predizioni
        grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )

        with tf.GradientTape() as tape:
            # Cast input to float32
            inputs = tf.cast(image_array, tf.float32)
            conv_outputs, predictions = grad_model(inputs)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            loss = predictions[:, class_idx]

        # Calcolo gradienti
        grads = tape.gradient(loss, conv_outputs)
        
        # Global Average Pooling dei gradienti
        guided_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Moltiplica le feature map per i pesi dei gradienti
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(guided_grads, conv_outputs), axis=-1)
        
        # Applica ReLU e normalizza
        heatmap = tf.maximum(heatmap, 0.0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()

    def overlay_heatmap(self, heatmap, original_image, alpha=0.4):
        """
        Sovrappone la heatmap all'immagine originale.
        """
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Resize heatmap to match original image
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Sovrapposizione
        superimposed_img = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        return superimposed_img
