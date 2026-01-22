"""
Knowledge Base (KB) medico per il sistema di diagnostica AI.
Questo modulo funge da "Enciclopedia Interna" o database di conoscenza statica.
Fornisce spiegazioni dettagliate, natura della patologia (Benigno/Maligno) e raccomandazioni cliniche standard
per ciascuna classe di cancro riconosciuta dal modello.

Viene utilizzato dall'interfaccia utente (o dai report) per arricchire la semplice predizione numerica
(es. "Classe 1") con informazioni utili contestuali per il medico o il paziente.
"""

# Dizionario principale contenente la conoscenza medica.
# Le chiavi devono corrispondere alle CLASS_NAMES usate nel config o nel dataset.
MEDICAL_KB = {
    # -------------------------------------------------------------------------
    # Leucemia (ALL)
    # -------------------------------------------------------------------------
    'ALL': {
        'title': 'Leucemia Linfoblastica Acuta (ALL)',
        'nature': 'Maligno (Tumore del sangue)',
        'description': 'La Leucemia Linfoblastica Acuta è un tipo di cancro aggressivo del sangue e del midollo osseo. Colpisce i globuli bianchi immaturi (linfoblasti), impedendo la produzione di cellule sane.',
        'details': 'È la forma più comune di cancro in età pediatrica, ma colpisce anche gli adulti. Progredisce rapidamente se non trattata tempestivamente.',
        'recommendation': 'Richiede consulto ematologico urgente. Esami tipici: emocromo completo, aspirato midollare, biopsia e tipizzazione genetica.'
    },

    # -------------------------------------------------------------------------
    # Brain Cancer (Cancro al Cervello)
    # -------------------------------------------------------------------------
    'Brain Cancer': { 
        # Nota: Nel dataset questa classe è spesso un mix di Gliomi e Meningiomi.
        'title': 'Tumore Cerebrale',
        'nature': 'Variabile (Benigno o Maligno)',
        'description': 'Crescita anomala di cellule nel tessuto cerebrale. Le forme comuni includono il Glioma (spesso maligno, origina dalla glia) e il Meningioma (spesso benigno, origina dalle meningi).',
        'details': 'I sintomi dipendono dalla localizzazione della massa e possono includere mal di testa, convulsioni, problemi visivi o motori. La diagnosi precisa richiede imaging avanzato.',
        'recommendation': 'Consulto neurochirurgico immediato. Risonanza Magnetica (MRI) con contrasto necessaria. La biopsia stereotassica è spesso richiesta per la diagnosi istologica definitiva.'
    },

    # -------------------------------------------------------------------------
    # Breast Cancer (Cancro al Seno)
    # -------------------------------------------------------------------------
    'Breast Cancer': {
        'title': 'Carcinoma Mammario',
        'nature': 'Maligno',
        'description': 'Cancro che si sviluppa nelle cellule del seno. Può essere duttale (dai dotti galattofori) o lobulare. Può essere "in situ" (confinato) o invasivo.',
        'details': 'È il tumore più frequente nelle donne. L\'analisi istologica determina lo stato dei recettori ormonali (ER/PR) e HER2, cruciali per la terapia target.',
        'recommendation': 'Percorso senologico completo: Mammografia e Ecografia mammaria. Biopsia (core biopsy) necessaria per caratterizzazione biologica e pianificazione terapeutica (chirurgia/chemo/ormonale).'
    },

    # -------------------------------------------------------------------------
    # Cervical Cancer (Cancro alla Cervice)
    # -------------------------------------------------------------------------
    'Cervical Cancer': {
        'title': 'Cancro della Cervice Uterina',
        'nature': 'Maligno',
        'description': 'Tumore che colpisce la parte inferiore dell\'utero (collo dell\'utero). È quasi sempre associato a un\'infezione persistente da Papilloma Virus Umano (HPV) ad alto rischio.',
        'details': 'Spesso asintomatico nelle fasi iniziali. Le lesioni precancerose (CIN) possono evolvere in carcinoma invasivo squamoso o adenocarcinoma se non trattate.',
        'recommendation': 'Visita ginecologica con Colposcopia. Biopsia mirata delle aree sospette. Test HPV DNA per tipizzazione virale.'
    },

    # -------------------------------------------------------------------------
    # Kidney Cancer (Cancro al Rene)
    # -------------------------------------------------------------------------
    'Kidney Cancer': {
        'title': 'Tumore del Rene',
        'nature': 'Maligno (spesso Carcinoma a Cellule Renali)',
        'description': 'Crescita maligna nel tessuto renale. Il tipo istologico più comune è il Carcinoma a Cellule Renali (RCC), che origina nei tubuli che filtrano il sangue.',
        'details': 'Spesso scoperto incidentalmente durante ecografie addominali. Sintomi tardivi includono sangue nelle urine (ematuria), dolore al fianco e massa palpabile.',
        'recommendation': 'TC addome completo con mezzo di contrasto (fase arteriosa, venosa, tardiva). Visita urologica per valutare nefrectomia parziale o radicale.'
    },

    # -------------------------------------------------------------------------
    # Lung and Colon Cancer (Misto Polmone/Colon)
    # -------------------------------------------------------------------------
    'Lung and Colon Cancer': {
        'title': 'Cancro Polmonare o del Colon',
        'nature': 'Maligno',
        'description': 'Classe che raggruppa Adenocarcinomi (comuni sia nel polmone che nel colon) e Carcinomi Squamosi. Entrambi sono tumori epiteliali maligni.',
        'details': 'Polmone: principale causa di morte per cancro (fumo è fattore chiave). Colon: origina spesso da polipi adenomatosi preesistenti. La diagnosi differenziale richiede immunoistochimica (CK7, CK20, TTF-1, CDX2).',
        'recommendation': 'Necessaria distinzione istologica precisa tramite immunoistochimica per definire l\'organo di origine (Polmone vs Colon) e la terapia specifica.'
    },

    # -------------------------------------------------------------------------
    # Lymphoma (Linfoma)
    # -------------------------------------------------------------------------
    'Lymphoma': {
        'title': 'Linfoma',
        'nature': 'Maligno',
        'description': 'Cancro che ha origine nel sistema linfatico (linfonodi, milza, timo, midollo). Si divide in due categorie principali: Linfoma di Hodgkin e Non-Hodgkin.',
        'details': 'Si manifesta spesso con ingrossamento non dolente dei linfonodi, febbre, sudorazioni notturne. Coinvolge i linfociti B o T.',
        'recommendation': 'Biopsia escissionale di un linfonodo intero (non agoaspirato) per l\'architettura tissutale. PET-TC total body per la stadiazione (diffusione malattia).'
    },

    # -------------------------------------------------------------------------
    # Oral Cancer (Cancro Orale)
    # -------------------------------------------------------------------------
    'Oral Cancer': {
        'title': 'Cancro Orale',
        'nature': 'Maligno (Carcinoma spinocellulare)',
        'description': 'Tumore che colpisce le mucose della bocca, lingua, gengive o orofaringe. Il tipo più comune è il Carcinoma a cellule squamose (o spinocellulare).',
        'details': 'Fortemente associato al consumo di tabacco e alcol, e in alcuni casi al virus HPV (specie orofaringeo). Può apparire come un\'ulcera bianca o rossa che non guarisce.',
        'recommendation': 'Visita specialistica Otorinolaringoiatrica o Maxillo-Facciale. Biopsia incisionale della lesione. Risonanza Magnetica collo/faccia per valutare l\'estensione.'
    }
}

def get_explanation(class_name):
    """
    Funzione helper per recuperare le informazioni mediche dal dizionario MEDICAL_KB
    in modo sicuro (gestendo il caso di chiavi mancanti).
    
    Args:
        class_name (str): Il nome della classe predetta dal modello (es. 'ALL').
        
    Returns:
        dict: Un dizionario contenente 'title', 'nature', 'description', 'details', 'recommendation'.
              Restituisce un dizionario di default se la classe non viene trovata.
    """
    # .get() restituisce il valore per la chiave, o il secondo argomento (dict di default) se la chiave non esiste.
    return MEDICAL_KB.get(class_name, {
        'title': class_name,
        'nature': 'N/A', # Not Available
        'description': 'Informazioni dettagliate non disponibili per questa classe specifica nel database.',
        'details': '',
        'recommendation': 'Si consiglia di consultare un medico specialista per l\'interpretazione del risultato.'
    })
