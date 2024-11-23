import os

def get_transcription_from_txt(audio_path: str):
    """
    Recherche la transcription d'un fichier audio à partir du fichier .trans.txt correspondant.
    
    Args:
        audio_path (str): Chemin complet du fichier audio (ex. 'data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000').
    
    Returns:
        str: La transcription correspondant à l'audio, ou un message d'erreur si non trouvé.
    """
    # Extraire le nom du fichier sans l'extension, puis retirer la partie après le dernier tiret pour obtenir '1272-128104'
    base_name = os.path.splitext(os.path.basename(audio_path))[0]  # '1272-128104-0000'
    base_name_tmp = '-'.join(base_name.split('-')[:2])  # '1272-128104'
    # Construire le chemin du fichier de transcription .trans.txt correspondant
    transcription_file_path = os.path.join(os.path.dirname(audio_path), f'{base_name_tmp}.trans.txt')
    
    # Vérifier si le fichier de transcription existe
    if not os.path.exists(transcription_file_path):
        return f"Le fichier de transcription {transcription_file_path} n'existe pas."
    
    # Ouvrir le fichier de transcription et rechercher la ligne correspondante
    with open(transcription_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Si la ligne commence par l'identifiant du fichier audio, retourner le reste de la ligne (la transcription)
            if line.startswith(base_name):
                # Retourner la transcription après l'identifiant
                return line.strip().split(' ', 1)[1] if ' ' in line else ""
    
    return f"Transcription pour {base_name} non trouvée dans {transcription_file_path}."

# Exemple d'utilisation
audio_path = 'data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000'
transcription = get_transcription_from_txt(audio_path)
print(transcription)
