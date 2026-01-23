"""
Modulo per la valutazione approfondita del modello (Evaluation) e interpretabilità (Explainable AI).
Include calcolo delle metriche di classificazione e generazione di Heatmap Grad-CAM per visualizzare
su quali aree dell'immagine l'AI si sta concentrando.
"""


# Lazy imports for: matplotlib, seaborn, sklearn, cv2 to avoid segfaults

class Evaluator:
    """
    Classe dedicata alla valutazione delle performance del modello su dati di test/validazione.
    """
    
    def __init__(self, model, validation_gen, class_indices):
        """
        Inizializza l'Evaluator.
        
        Args:
            model: Il modello addestrato.
            validation_gen: Dataset di validazione (tf.data.Dataset o generatore).
            class_indices: Dizionario {NomeClasse: Indice}.
        """
        self.model = model
        self.validation_gen = validation_gen
        self.class_indices = class_indices
        
        # Creiamo un mapping inverso: Indice -> Nome Classe (es. {0: 'ALL', 1: 'Brain...'})
        # Utile per convertire le predizioni numeriche in etichette leggibili.
        self.idx_to_class = {v: k for k, v in class_indices.items()}
        self.class_names = list(class_indices.keys())

    def evaluate(self):
        """
        Esegue le predizioni sull'intero validation set e confronta con le etichette reali.
        
        Returns:
            tuple: (y_true, y_pred, raw_predictions)
                - y_true: Array numpy con gli indici delle classi reali corretti.
                - y_pred: Array numpy con gli indici delle classi predette dal modello.
                - raw_predictions: Matrice delle probabilità raw (opzionale).
        """
        print("Calcolo predizioni per evaluation...")
        
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

    def generate_report(self, y_true, y_pred):
        """
        Genera un report testuale standard (Precision, Recall, F1-Score) per ogni classe.
        Usa scikit-learn classification_report.
        
        Args:
            y_true: Etichette reali.
            y_pred: Etichette predette.
            
        Returns:
            str: Stringa formattata contenente la tabella delle metriche.
        """
        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred, target_names=self.class_names)
        return report

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        Genera una Matrice di Confusione grafica (Heatmap).
        Mostra quante volte la classe X è stata confusa con la classe Y.
        
        Args:
            y_true: Etichette reali.
            y_pred: Etichette predette.
            save_path: Percorso dove salvare l'immagine PNG.
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
    Tecnica per visualizzare le regioni dell'immagine che hanno influenzato maggiormente la decisione della CNN.
    """
    
    def __init__(self, model, layer_name=None):
        """
        Args:
            model: Il modello Keras.
            layer_name: Il nome dell'ultimo strato convoluzionale (se None, prova a trovarlo automaticamente).
        """
        self.model = model
        # Trova l'ultimo layer convoluzionale 4D (Feature Map)
        self.layer_name = layer_name or self._find_target_layer()

    def _find_target_layer(self):
        """
        Cerca automaticamente l'ultimo layer convoluzionale che restituisce un tensore 4D (Batch, H, W, Channels).
        Questo è il layer che contiene le feature spaziali più richhe prima del pooling finale.
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

    def compute_heatmap(self, image_array, class_idx=None, eps=1e-8):
        """
        Calcola la heatmap dei gradienti.
        
        Args:
            image_array: Immagine di input (224, 224, 3).
            class_idx: Indice della classe di cui vogliamo spiegare la predizione (default: la classe vincente).
            
        Returns:
            numpy.ndarray: Heatmap 2D normalizzata (valori 0-1).
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

    def overlay_heatmap(self, heatmap, original_image, alpha=0.4):
        """
        Sovrappone la heatmap (colorata) all'immagine originale in scala di grigi o colore.
        
        Args:
            heatmap: Heatmap 2D (0-1).
            original_image: Immagine originale RGB.
            alpha: Trasparenza della heatmap (0-1).
            
        Returns:
            numpy.ndarray: Immagine finale sovrapposta.
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
