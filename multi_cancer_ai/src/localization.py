"""
Modulo di Localizzazione (Localization).
Gestisce tutte le traduzioni dell'interfaccia utente (UI) e dei contenuti testuali.
Supporta attualmente Italiano (IT) e Inglese (EN).

Inoltre, contiene un "Glossario Medico" mappato sulle classi del dataset, 
per fornire brevi descrizioni contestuali direttamente nella lingua selezionata.
"""

# Dizionario principale delle traduzioni.
# Struttura: { "codice_lingua": { "chiave_stringa": "Testo Tradotto" } }
TRANSLATIONS = {
    "it": {
        # --- Sidebar (Menu Laterale) ---
        "app_title": "Multi-Cancer AI Diagnostic System",
        "sidebar_home": "Home / Stato",
        "sidebar_diagnose": "Diagnostica",
        "sidebar_train": "Training Attivo",
        "sidebar_info": "Info Dataset",
        "sidebar_mode": "Tema:",
        "sidebar_lang": "Lingua / Language:",
        
        # --- Home Page ---
        "home_welcome": "Benvenuto nel Sistema Diagnostico AI",
        "home_status": "Stato del Sistema",
        "home_model_ok": "✅ Motore AI Pronto",
        "home_model_desc": "Modello caricato: {}", # Placeholder {} per il nome del file
        "home_instr": "Istruzioni Rapide:\n1. Vai su 'Diagnostica' per analizzare immagini.\n2. Usa 'Training Attivo' se l'AI sbaglia.\n3. Consulta 'Info Dataset' per dettagli medici.",
        
        # --- Pagina Diagnostica ---
        "diag_title": "Sistema Diagnostico",
        "diag_load_btn": "Carica Immagine",
        "diag_predict_btn": "Esegui Diagnosi",
        "diag_result": "Risultato: {}",
        "diag_confidence": "Confidenza: {:.1f}%", # Placeholder {:.1f} per numero float a 1 decimale
        "diag_time": "Tempo: {:.3f}s",            # Placeholder {:.3f} per tempo a 3 decimali
        "diag_wrong_btn": "Diagnosi Errata? Correggila!",
        "diag_heatmap": "Mostra Heatmap (Grad-CAM)",
        
        # --- Pagina Training Attivo ---
        "train_title": "Training Attivo (Insegnamento)",
        "train_instr": "Se l'AI ha sbagliato, seleziona qui la classe corretta.\nIl modello imparerà da questa immagine.",
        "train_select_label": "Classe Corretta:",
        "train_confirm_btn": "Conferma e Addestra",
        "train_success": "✅ Addestramento completato! Il modello è stato aggiornato.",
        
        # --- Pagina Info ---
        "info_title": "Struttura Dataset & Enciclopedia Medica",
        
        # --- Bottoni Generici ---
        "btn_toggle_lang": "🇮🇹 IT / 🇺🇸 EN"
    },
    "en": {
        # --- Sidebar ---
        "app_title": "Multi-Cancer AI Diagnostic System",
        "sidebar_home": "Home / Status",
        "sidebar_diagnose": "Diagnostics",
        "sidebar_train": "Active Training",
        "sidebar_info": "Dataset Info",
        "sidebar_mode": "Theme:",
        "sidebar_lang": "Language / Lingua:",
        
        # --- Home Page ---
        "home_welcome": "Welcome to AI Diagnostic System",
        "home_status": "System Status",
        "home_model_ok": "✅ AI Engine Ready",
        "home_model_desc": "Loaded Model: {}",
        "home_instr": "Quick Instructions:\n1. Go to 'Diagnostics' to analyze images.\n2. Use 'Active Training' if AI fails.\n3. Check 'Dataset Info' for medical details.",
        
        # --- Diagnostics Page ---
        "diag_title": "Diagnostic System",
        "diag_load_btn": "Load Image",
        "diag_predict_btn": "Run Diagnosis",
        "diag_result": "Result: {}",
        "diag_confidence": "Confidence: {:.1f}%",
        "diag_time": "Time: {:.3f}s",
        "diag_wrong_btn": "Wrong Diagnosis? Fix IT!",
        "diag_heatmap": "Show Heatmap (Grad-CAM)",
        
        # --- Active Training Page ---
        "train_title": "Active Training (Teaching)",
        "train_instr": "If AI was wrong, select the correct class below.\nThe model will learn from this image immediately.",
        "train_select_label": "Correct Class:",
        "train_confirm_btn": "Confirm & Train",
        "train_success": "✅ Training complete! Model updated.",
        
        # --- Info Page ---
        "info_title": "Dataset Structure & Medical Encyclopedia",
        
        # --- Generic Buttons ---
        "btn_toggle_lang": "🇮🇹 IT / 🇺🇸 EN"
    }
}

# Mapping del "Glossario Medico".
# Queste stringhe contengono descrizioni brevi delle patologie, mappate su codici o sottostringhe
# che possono apparire nei nomi delle classi o dei file.
MEDICAL_GLOSSARY = {
    "it": {
        # --- ALL (Leucemia) ---
        "all_pro": "Leucemia Linfoblastica Acuta (ALL-Pro): Cancro del sangue che colpisce i globuli bianchi immaturi nel midollo osseo.",
        
        # --- Brain (Cervello) ---
        "brain_glioma": "Glioma: Tumore che origina dalle cellule gliali del cervello.",
        "brain_menin": "Meningioma: Tumore che origina dalle meningi (le membrane che avvolgono il cervello). Spesso benigno.",
        "brain_tumor": "Tumore Cerebrale (Generico): Massa anomala di tessuto nel cervello.",
        
        # --- Breast (Seno) ---
        "breast_benign": "Seno (Benigno): Formazione non cancerosa, non si diffonde ad altre parti del corpo.",
        "breast_malignant": "Seno (Maligno): Carcinoma mammario invasivo, richiede trattamento immediato.",
        
        # --- Cervix (Cervice) - Terminologia Citologica Bethesda ---
        "cervix_dyk": "Cervice (Discheratosi): Alterazione cellulare anomala, potenziale precancerosi.",
        "cervix_koc": "Cervice (Koilocitosi): Cellule infettate da HPV (Papilloma Virus).",
        "cervix_mep": "Cervice (Metaplasia): Trasformazione di un tipo cellulare in un altro, spesso benigna ma da monitorare.",
        "cervix_pab": "Cervice (Parabasale): Cellule immature, possibile segno di atrofia o infiammazione.",
        "cervix_sfi": "Cervice (Squamoso Superficiale): Cellule piatte normali o con lievi anomalie.",
        
        # --- Kidney (Rene) ---
        "kidney_normal": "Rene (Sano): Tessuto renale normale senza anomalie visibili.",
        "kidney_tumor": "Tumore Renale: Massa anomala nel rene, spesso Carcinoma a Cellule Renali.",
        
        # --- Lung/Colon (Polmone/Colon) ---
        "lung_aca": "Polmone (Adenocarcinoma): Cancro che inizia nelle cellule che secernono muco. Comune anche nei non fumatori.",
        "lung_scc": "Polmone (Carcinoma Squamoso): Cancro che origina nelle cellule squamose delle vie aeree.",
        "lung_bnt": "Polmone (Benigno): Tessuto polmonare con noduli non cancerosi.",
        "colon_aca": "Colon (Adenocarcinoma): Cancro del colon che origina dalle ghiandole.",
        "colon_bnt": "Colon (Benigno): Polipi o tessuti non cancerosi.",
        
        # --- Lymphoma (Linfoma) ---
        "lymph_cll": "Leucemia Linfatica Cronica (CLL): Cancro del sangue a crescita lenta.",
        "lymph_fl": "Linfoma Follicolare (FL): Linfoma non-Hodgkin a crescita lenta.",
        "lymph_mcl": "Linfoma Mantellare (MCL): Forma rara e aggressiva di linfoma non-Hodgkin.",
        
        # --- Oral (Cavo Orale) ---
        "oral_normal": "Cavo Orale (Sano): Tessuto della bocca senza lesioni.",
        "oral_scc": "Cavo Orale (Carcinoma Squamoso): Cancro che colpisce le mucose della bocca.",
        
        # --- Fallback (Generici per le classi macro) ---
        "Lung and Colon Cancer": "Gruppo contenente vari tipi di tumori polmonari e del colon.",
        "Lymphoma": "Gruppo di tumori del sistema linfatico.",
        "Brain Cancer": "Gruppo di tumori cerebrali (Gliomi, Meningiomi, ecc.).",
    },
    "en": {
        # --- ALL ---
        "all_pro": "Acute Lymphoblastic Leukemia (ALL-Pro): Blood cancer affecting immature white blood cells in bone marrow.",
        
        # --- Brain ---
        "brain_glioma": "Glioma: Tumor originating from glial cells in the brain.",
        "brain_menin": "Meningioma: Tumor arising from the meninges. Often benign.",
        "brain_tumor": "Brain Tumor (Generic): Abnormal mass of tissue in the brain.",
        
        # --- Breast ---
        "breast_benign": "Breast (Benign): Non-cancerous growth, does not spread.",
        "breast_malignant": "Breast (Malignant): Invasive breast carcinoma, requires immediate treatment.",
        
        # --- Cervix ---
        "cervix_dyk": "Cervix (Dyskeratosis): Abnormal cell changes, potential precancerous condition.",
        "cervix_koc": "Cervix (Koilocytosis): Cells infected by HPV.",
        "cervix_mep": "Cervix (Metaplasia): Transformation of cell types, usually benign but monitored.",
        "cervix_pab": "Cervix (Parabasal): Immature cells, possible sign of atrophy or inflammation.",
        "cervix_sfi": "Cervix (Superficial Squamous): Normal flat cells or mild abnormalities.",
        
        # --- Kidney ---
        "kidney_normal": "Kidney (Normal): Healthy kidney tissue.",
        "kidney_tumor": "Kidney Tumor: Abnormal mass, often Renal Cell Carcinoma.",
        
        # --- Lung/Colon ---
        "lung_aca": "Lung (Adenocarcinoma): Starts in mucus-secreting cells. Common in non-smokers.",
        "lung_scc": "Lung (Squamous Cell Carcinoma): Starts in squamous cells of airways.",
        "lung_bnt": "Lung (Benign): Non-cancerous lung nodules.",
        "colon_aca": "Colon (Adenocarcinoma): Colon cancer originating from glands.",
        "colon_bnt": "Colon (Benign): Polyps or non-cancerous tissue.",
        
        # --- Lymphoma ---
        "lymph_cll": "Chronic Lymphocytic Leukemia (CLL): Slow-growing blood cancer.",
        "lymph_fl": "Follicular Lymphoma (FL): Slow-growing Non-Hodgkin lymphoma.",
        "lymph_mcl": "Mantle Cell Lymphoma (MCL): Rare and aggressive form of Non-Hodgkin lymphoma.",
        
        # --- Oral ---
        "oral_normal": "Oral (Healthy): Normal mouth tissue.",
        "oral_scc": "Oral (Squamous Cell Carcinoma): Cancer affecting mouth lining.",
        
        # --- Fallback ---
        "Lung and Colon Cancer": "Group containing various Lung and Colon cancer types.",
        "Lymphoma": "Group of Lymphatic system cancers.",
        "Brain Cancer": "Group of Brain tumors (Glioma, Meningioma, etc.).",
    }
}
