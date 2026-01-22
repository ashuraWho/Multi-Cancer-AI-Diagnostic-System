"""
Knowledge Base medico per il sistema di diagnostica.
Fornisce spiegazioni dettagliate, natura (Benigno/Maligno) e raccomandazioni per ogni classe.
"""

MEDICAL_KB = {
    'ALL': {
        'title': 'Leucemia Linfoblastica Acuta (ALL)',
        'nature': 'Maligno (Tumore del sangue)',
        'description': 'La Leucemia Linfoblastica Acuta è un tipo di cancro del sangue e del midollo osseo. Colpisce i globuli bianchi immaturi.',
        'details': 'È la forma più comune di cancro nei bambini. Progredisce rapidamente se non trattata.',
        'recommendation': 'Richiede consulto ematologico urgente per biopsia midollare e tipizzazione.'
    },
    'Brain Cancer': { # General category, usually Glioma or Meningioma in dataset
        'title': 'Tumore Cerebrale',
        'nature': 'Variabile (Benigno o Maligno)',
        'description': 'Crescita anomala di cellule nel cervello. Può trattarsi di Glioma (spesso maligno) o Meningioma (spesso benigno).',
        'details': 'Richiede risonanza magnetica (MRI) per determinare l\'esatta natura e localizzazione.',
        'recommendation': 'Consulto neurochirurgico immediato. Biopsia necessaria per diagnosi definitiva.'
    },
    'Breast Cancer': {
        'title': 'Carcinoma Mammario',
        'nature': 'Maligno',
        'description': 'Cancro che si sviluppa nelle cellule del seno. Può essere invasivo o in situ.',
        'details': 'L\'istologia mostra cellule duttali o lobulari anomale. È uno dei tumori più trattabili se preso in tempo.',
        'recommendation': 'Mammografia, ecografia e biopsia core per definire recettori ormonali e HER2.'
    },
    'Cervical Cancer': {
        'title': 'Cancro della Cervice Uterina',
        'nature': 'Maligno',
        'description': 'Tumore che colpisce la parte inferiore dell\'utero. Spesso associato a infezione da HPV.',
        'details': 'Le lesioni precancerose (CIN) possono evolvere in carcinoma invasivo.',
        'recommendation': 'Colposcopia e biopsia mirata. Test HPV DNA.'
    },
    'Kidney Cancer': {
        'title': 'Tumore del Rene',
        'nature': 'Maligno (spesso Carcinoma a Cellule Renali)',
        'description': 'Il tipo più comune è il Carcinoma a cellule renali. Origina nei tubuli renali.',
        'details': 'Spesso asintomatico nelle fasi iniziali. Può causare ematuria (sangue nelle urine).',
        'recommendation': 'TC addome con mezzo di contrasto e visita urologica.'
    },
    'Lung and Colon Cancer': {
        'title': 'Cancro Polmonare o del Colon',
        'nature': 'Maligno',
        'description': 'Categoria mista che indica Adenocarcinoma (Polmone/Colon) o Carcinoma a cellule squamose (Polmone).',
        'details': 'Polmone: spesso legato al fumo. Colon: origina spesso da polipi. Entrambi richiedono stadiazione accurata.',
        'recommendation': 'Necessaria distinzione istologica precisa (immunoistochimica) per definire la terapia.'
    },
    'Lymphoma': {
        'title': 'Linfoma',
        'nature': 'Maligno',
        'description': 'Cancro del sistema linfatico (linfonodi, milza, timo).',
        'details': 'Si divide principalmente in Hodgkin e Non-Hodgkin. Colpisce le difese immunitarie.',
        'recommendation': 'Biopsia escissionale del linfonodo e PET-TC total body.'
    },
    'Oral Cancer': {
        'title': 'Cancro Orale',
        'nature': 'Maligno (Carcinoma spinocellulare)',
        'description': 'Tumore che colpisce bocca, lingua o orofaringe.',
        'details': 'Fattori di rischio: fumo, alcol, HPV. Può apparire come un\'ulcera che non guarisce.',
        'recommendation': 'Visita otorinolaringoiatrica o maxillo-facciale urgente. Biopsia incisionale.'
    }
}

def get_explanation(class_name):
    """Restituisce le info mediche per la classe data."""
    return MEDICAL_KB.get(class_name, {
        'title': class_name,
        'nature': 'N/A',
        'description': 'Informazioni non disponibili per questa classe specifica.',
        'details': '',
        'recommendation': 'Consultare un medico specialista.'
    })
