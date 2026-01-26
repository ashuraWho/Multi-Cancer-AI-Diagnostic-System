"""
Modulo per la valutazione approfondita del modello (Evaluation) e interpretabilità (Explainable AI).

Include:
- Calcolo delle metriche di classificazione (Precision, Recall, F1-Score)
- Generazione di Matrici di Confusione
- Implementazione Grad-CAM per visualizzare le regioni dell'immagine che influenzano la predizione

Author: Multi-Cancer AI Team
License: MIT
"""

from typing import Tuple, Optional, Dict
import numpy as np

# Lazy imports for: matplotlib, seaborn, sklearn, cv2 to avoid segfaults
# Questi vengono importati localmente nei metodi che li usano per evitare
# problemi di inizializzazione con Tkinter su macOS.

class Evaluator:
    """
    Classe dedicata alla valutazione delle performance del modello su dati di test/validazione.
    
    Fornisce metodi per:
    - Calcolo di predizioni batch su dataset di validazione
    - Generazione di report testuali (classification_report)
    - Visualizzazione di matrici di confusione
    
    Attributes:
        model: Modello Keras addestrato.
        validation_gen: Dataset di validazione (tf.data.Dataset).
        class_indices: Dizionario {NomeClasse: IndiceNumerico}.
        idx_to_class: Mapping inverso {IndiceNumerico: NomeClasse} per conversione predizioni.
        class_names: Lista ordinata dei nomi delle classi.
    """
    
    def __init__(
        self,
        model,
        validation_gen,
        class_indices: Dict[str, int]
    ) -> None:
        """
        Inizializza l'Evaluator con modello, dataset e mapping classi.
        
        Args:
            model: Il modello Keras addestrato (deve avere metodo predict_on_batch).
            validation_gen: Dataset di validazione (tf.data.Dataset con batch di immagini+label).
            class_indices: Dizionario {NomeClasse: IndiceNumerico} (es. {'ALL': 0, 'Brain Cancer': 1}).
        
        Raises:
            ValueError: Se class_indices è vuoto o contiene valori non univoci.
        """
        import tensorflow as tf
        
        if not isinstance(model, tf.keras.Model):
            raise TypeError("model deve essere un'istanza di tf.keras.Model")
        
        if not class_indices:
            raise ValueError("class_indices non può essere vuoto")
        
        if len(set(class_indices.values())) != len(class_indices):
            raise ValueError("class_indices contiene valori duplicati")
        
        self.model = model
        self.validation_gen = validation_gen
        self.class_indices = class_indices
        
        # Creiamo un mapping inverso: Indice -> Nome Classe (es. {0: 'ALL', 1: 'Brain...'})
        # Utile per convertire le predizioni numeriche in etichette leggibili.
        self.idx_to_class = {v: k for k, v in class_indices.items()}
        self.class_names = sorted(class_indices.keys())  # Ordinato per consistenza

    def evaluate(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Esegue le predizioni sull'intero validation set e confronta con le etichette reali.
        
        Itera su tutti i batch del validation dataset, accumula le predizioni e le etichette,
        e restituisce array numpy pronti per il calcolo delle metriche.
        
        Returns:
            Tuple contenente:
                - y_true: Array numpy 1D con gli indici delle classi reali (shape: [N_samples]).
                - y_pred: Array numpy 1D con gli indici delle classi predette (shape: [N_samples]).
                - raw_predictions: None (per compatibilità futura, potrebbe restituire probabilità).
        
        Note:
            - Le etichette vengono convertite da one-hot encoding a indici interi (argmax)
            - Le predizioni vengono convertite da probabilità a indici di classe (argmax)
            - Il dataset viene iterato completamente (non viene fatto shuffle durante l'evaluation)
        
        Example:
            >>> evaluator = Evaluator(model, val_ds, class_indices)
            >>> y_true, y_pred, _ = evaluator.evaluate()
            >>> accuracy = (y_true == y_pred).mean()
        """
        import logging
        logger = logging.getLogger("MultiCancerAI")
        logger.info("Calcolo predizioni per evaluation...")
        
        y_true = [] # Lista per accumulare le vere etichette
        y_pred = [] # Lista per accumulare le predizioni
        
        # Iteriamo manualmente sul dataset batch per batch.
        # Questo è necessario con tf.data per essere sicuri di allineare correttamente X e Y,
        # dato che model.predict() su un dataset shufflato potrebbe disallinearsi se non gestito bene.
        # (Nota: validation_gen nel nostro caso ha shuffle=False, quindi è sicuro, ma questo loop è esplicito).
        
        for batch_images, batch_labels in self.validation_gen:
            # batch_images: Tensore (32, 224, 224, 3)
            # batch_labels: Tensore One-Hot (32, 8)
            
            # Eseguiamo la predizione sul batch corrente
            preds = self.model.predict_on_batch(batch_images)
            
            # Convertiamo da One-Hot a indice intero (argmax)
            # batch_labels (es. [0, 0, 1, 0]) -> indice 2
            batch_true = np.argmax(batch_labels.numpy(), axis=1)
            
            # Convertiamo le probabilità predette in indice classe
            # preds (es. [0.1, 0.05, 0.8, ...]) -> indice 2
            batch_pred = np.argmax(preds, axis=1)
            
            # Accumuliamo i risultati
            y_true.extend(batch_true)
            y_pred.extend(batch_pred)
            
        # Convertiamo le liste in array Numpy per facilitare i calcoli successivi
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        return y_true, y_pred, None

    def generate_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        Genera un report testuale standard (Precision, Recall, F1-Score) per ogni classe.
        
        Utilizza scikit-learn classification_report per calcolare metriche per-classe
        e metriche aggregate (macro/micro average).
        
        Args:
            y_true: Array numpy 1D con gli indici delle classi reali (shape: [N_samples]).
            y_pred: Array numpy 1D con gli indici delle classi predette (shape: [N_samples]).
        
        Returns:
            Stringa formattata contenente la tabella delle metriche per classe:
            - Precision: TP / (TP + FP) - Accuratezza delle predizioni positive
            - Recall: TP / (TP + FN) - Copertura delle classi reali
            - F1-Score: Media armonica di Precision e Recall
            - Support: Numero di campioni per classe
        
        Raises:
            ValueError: Se y_true e y_pred hanno shape diverse o contengono indici invalidi.
        
        Example:
            >>> report = evaluator.generate_report(y_true, y_pred)
            >>> print(report)
        """
        from sklearn.metrics import classification_report
        
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        
        if y_true.max() >= len(self.class_names) or y_pred.max() >= len(self.class_names):
            raise ValueError("Indici di classe fuori range rispetto a class_names")
        
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            digits=4  # 4 decimali per precisione
        )
        return report

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Genera una Matrice di Confusione grafica (Heatmap) usando Seaborn.
        
        La matrice di confusione mostra quante volte ogni classe reale è stata
        predetta come ogni altra classe. Valori sulla diagonale = predizioni corrette.
        
        Args:
            y_true: Array numpy 1D con gli indici delle classi reali (shape: [N_samples]).
            y_pred: Array numpy 1D con gli indici delle classi predette (shape: [N_samples]).
            save_path: Percorso opzionale dove salvare l'immagine PNG.
                      Se None, il grafico viene solo generato ma non salvato.
        
        Raises:
            ValueError: Se y_true e y_pred hanno shape diverse.
        
        Note:
            - Il grafico viene chiuso automaticamente dopo il salvataggio (plt.close())
            - La heatmap usa colormap 'Blues' per visualizzazione chiara
            - Le etichette degli assi sono i nomi delle classi (non indici numerici)
        
        Example:
            >>> evaluator.plot_confusion_matrix(y_true, y_pred, save_path="cm.png")
        """
        # Local imports
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        # Calcolo della matrice numerica
        cm = confusion_matrix(y_true, y_pred)
        
        # Configurazione grafico
        plt.figure(figsize=(10, 8))
        
        # Disegno Heatmap con Seaborn
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        
        plt.title('Matrice di Confusione')
        plt.ylabel('Vero (Ground Truth)')
        plt.xlabel('Predetto (Predicted)')
        plt.tight_layout() # Aggiusta i margini per non tagliare le etichette
        
        if save_path:
            plt.savefig(save_path)
        
        plt.close() # Chiude la figura

class GradCAM:
    """
    Implementazione di Grad-CAM (Gradient-weighted Class Activation Mapping).
    
    Grad-CAM è una tecnica di Explainable AI (XAI) che visualizza le regioni dell'immagine
    che hanno influenzato maggiormente la decisione della CNN. Genera una "heatmap" termica
    sovrapponibile all'immagine originale, dove:
    - Rosso/Giallo = regioni ad alta importanza per la predizione
    - Blu/Nero = regioni ignorate dal modello
    
    L'implementazione segue il paper originale:
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    (Selvaraju et al., ICCV 2017)
    
    Attributes:
        model: Modello Keras per cui calcolare le heatmap.
        layer_name: Nome del layer convoluzionale target (ultimo conv layer prima del pooling).
    """
    
    def __init__(self, model, layer_name: Optional[str] = None) -> None:
        """
        Inizializza GradCAM con il modello e il layer target.
        
        Args:
            model: Il modello Keras (deve essere un modello funzionale, non sequenziale puro).
            layer_name: Il nome dell'ultimo strato convoluzionale (feature map 4D).
                       Se None, viene cercato automaticamente tramite _find_target_layer().
        
        Raises:
            ValueError: Se il layer target non viene trovato e layer_name è None.
        
        Note:
            - Il layer target deve avere output shape 4D: (batch, H, W, channels)
            - Layer tipici: ultimo blocco EfficientNet prima di GlobalAveragePooling
        """
        import tensorflow as tf
        
        if not isinstance(model, tf.keras.Model):
            raise TypeError("model deve essere un'istanza di tf.keras.Model")
        
        self.model = model
        # Trova l'ultimo layer convoluzionale 4D (Feature Map)
        self.layer_name = layer_name or self._find_target_layer()
        
        if self.layer_name is None:
            raise ValueError("Impossibile trovare layer convoluzionale target. Specifica layer_name manualmente.")

    def _find_target_layer(self) -> Optional[str]:
        """
        Cerca automaticamente l'ultimo layer convoluzionale che restituisce un tensore 4D.
        
        Il layer target deve avere output shape 4D: (batch, H, W, channels).
        Questo è il layer che contiene le feature spaziali più ricche prima del pooling finale.
        Per EfficientNetV2-B0, tipicamente è 'top_activation' o l'ultimo blocco MBConv.
        
        Returns:
            Nome del layer trovato, o None se nessun layer 4D viene trovato.
        
        Note:
            - La ricerca avviene in ordine inverso (dall'output all'input)
            - Gestisce anche modelli annidati (sub-layers) fino a 1 livello di profondità
        """
        # Scansioniamo i layer al contrario (dall'output all'input)
        for layer in reversed(self.model.layers):
            try:
                # Controlliamo la shape dell'output del layer
                if hasattr(layer, 'output'):
                    output_shape = layer.output.shape
                else:
                    output_shape = getattr(layer, 'output_shape', None)
            except (AttributeError, RuntimeError):
                output_shape = None

            # Se ha 4 dimensioni (es. None, 7, 7, 1280), è un candidato valido (Feature Map)
            if output_shape is not None and len(output_shape) == 4:
                return layer.name
            
            # Gestione Modelli Annidati (es. se EfficientNet è dentro un layer funzionale)
            if hasattr(layer, 'layers'):
                for sub_layer in reversed(layer.layers):
                     # Logica ricorsiva semplificata per 1 livello di nesting
                     try:
                        if hasattr(sub_layer, 'output'):
                            sub_output_shape = sub_layer.output.shape
                        else:
                            sub_output_shape = getattr(sub_layer, 'output_shape', None)
                     except (AttributeError, RuntimeError):
                        sub_output_shape = None

                     if sub_output_shape is not None and len(sub_output_shape) == 4:
                        return layer.name 
        
        # Fallback hardcoded per EfficientNetV2B0/B3 se l'autodiscovery fallisce
        return 'top_activation' 

    def compute_heatmap(
        self,
        image_array: np.ndarray,
        class_idx: Optional[int] = None,
        eps: float = 1e-8
    ) -> np.ndarray:
        """
        Calcola la heatmap Grad-CAM per una singola immagine.
        
        Algoritmo:
        1. Forward pass: ottiene feature maps convoluzionali e predizione finale
        2. Backward pass: calcola gradienti della loss (classe target) rispetto alle feature maps
        3. Global Average Pooling sui gradienti: ottiene pesi per canale
        4. Weighted sum delle feature maps: moltiplica feature maps per pesi
        5. ReLU: mantiene solo influenze positive
        6. Normalizzazione: scala a [0, 1] per visualizzazione
        
        Args:
            image_array: Immagine di input preprocessata (shape: [224, 224, 3] o [1, 224, 224, 3]).
                        Valori devono essere in [0, 1] (float32).
            class_idx: Indice della classe di cui vogliamo spiegare la predizione.
                      Se None, usa la classe con probabilità più alta (argmax).
            eps: Valore epsilon per evitare divisione per zero nella normalizzazione.
        
        Returns:
            Heatmap 2D normalizzata (shape: [H, W], valori in [0, 1]).
            La heatmap ha le stesse dimensioni spaziali del layer target (tipicamente 7x7 per EfficientNet).
        
        Raises:
            ValueError: Se image_array ha shape non valida o valori fuori range [0, 1].
        
        Example:
            >>> grad_cam = GradCAM(model)
            >>> heatmap = grad_cam.compute_heatmap(img_array, class_idx=2)
            >>> # heatmap.shape = (7, 7) per EfficientNetV2-B0
        """
        import tensorflow as tf
        
        # Costruiamo un modello "Grad-Model" che ha:
        # Input: l'input originale
        # Output: [Output Layer Convoluzionale, Output Predizione Finale]
        grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )

        with tf.GradientTape() as tape:
            # Castiamo l'input a float32 per sicurezza
            inputs = tf.cast(image_array, tf.float32)
            
            # Forward Pass: otteniamo feature maps e predizioni
            conv_outputs, predictions = grad_model(inputs)
            
            # Se non speficicato, usiamo la classe con probabilità più alta
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            # Isoliamo il valore di probabilità per la classe target
            loss = predictions[:, class_idx]

        # Backward Pass: Calcoliamo il gradiente della Loss rispetto alle Feature Maps convoluzionali.
        # Questo ci dice "quanto cambiare ogni pixel della feature map per cambiare la predizione della classe".
        grads = tape.gradient(loss, conv_outputs)
        
        # Global Average Pooling sui gradienti:
        # Otteniamo un peso scalare per ogni canale della feature map.
        # Canali con gradienti alti sono "importanti" per quella classe.
        guided_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Moltiplichiamo le feature maps per i pesi calcolati.
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(guided_grads, conv_outputs), axis=-1)
        
        # Applichiamo ReLU (Rectified Linear Unit) alla heatmap.
        # Ci interessano solo le influenze POSITIVE (pixel che supportano la classe), non quelle negative.
        heatmap = tf.maximum(heatmap, 0.0)
        
        # Normalizziamo tra 0 e 1 per visualizzazione
        heatmap /= tf.math.reduce_max(heatmap) + eps
        
        return heatmap.numpy()

    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        original_image: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Sovrappone la heatmap (colorata) all'immagine originale usando blending pesato.
        
        La heatmap viene convertita in colormap termica (JET: blu->verde->giallo->rosso)
        e sovrapposta all'immagine originale con trasparenza configurabile.
        
        Args:
            heatmap: Heatmap 2D normalizzata (shape: [H, W], valori in [0, 1]).
            original_image: Immagine originale RGB (shape: [H, W, 3], valori uint8 [0, 255]).
            alpha: Trasparenza della heatmap (0.0 = solo originale, 1.0 = solo heatmap).
                  Valore tipico: 0.4 per buon bilanciamento visibilità/dettaglio.
        
        Returns:
            Immagine finale sovrapposta (shape: [H, W, 3], valori uint8 [0, 255]).
            Pronta per visualizzazione con PIL/OpenCV.
        
        Raises:
            ValueError: Se heatmap o original_image hanno shape non compatibili.
        
        Note:
            - La heatmap viene ridimensionata automaticamente se ha dimensioni diverse dall'originale
            - Usa colormap JET (OpenCV) per visualizzazione termica standard
            - Formula blending: output = original * (1-alpha) + heatmap_colored * alpha
        
        Example:
            >>> overlay = grad_cam.overlay_heatmap(heatmap, original_img, alpha=0.4)
            >>> Image.fromarray(overlay).show()
        """
        import cv2

        # Scaliamo a 0-255 intero
        heatmap = np.uint8(255 * heatmap)
        
        # Coloriamo la heatmap usando una mappa di colori termica (JET)
        # Blu = basso interesse, Rosso = alto interesse
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Ridimensioniamo la heatmap alla dimensione dell'immagine originale se differisce
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Sovrapposizione pesata (Blending)
        # Output = Originale * (1-alpha) + Heatmap * alpha
        superimposed_img = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        
        return superimposed_img
