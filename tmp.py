import os
import whisper
from whisper.utils import get_writer
import jiwer  # Import for WER calculation
import csv
# Load the Whisper model
model = whisper.load_model('base', device='cuda')

def append_to_csv(file_name, wer, per):
    """
    Appends WER and PER scores to a CSV file.
    
    Args:
        file_name (str): The name of the CSV file.
        wer (float): Word Error Rate.
        per (float): Phoneme Error Rate.
    """
    # If the CSV file doesn't exist, create it and write the header
    file_exists = os.path.exists(file_name)
    
    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write the header only if the file doesn't exist
        if not file_exists:
            writer.writerow(['audio_filename', 'wer', 'per'])
        
        # Append the audio filename, WER, and PER scores
        writer.writerow([file_name, wer, per])

def get_transcribe(audio: str, language: str = 'en'):
    """
    Transcribes the given audio file using the Whisper model.
    
    Args:
        audio (str): Path to the audio file.
        language (str): Language of the audio. Defaults to 'en'.
    
    Returns:
        dict: The transcription results.
    """
    return model.transcribe(audio=audio, language=language, verbose=False)

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

def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate the Word Error Rate (WER) between the reference and hypothesis text.
    
    Args:
        reference (str): The ground truth transcription.
        hypothesis (str): The transcription produced by the model.
    
    Returns:
        float: The WER score.
    """
    return jiwer.wer(reference, hypothesis)

def calculate_per(reference: str, hypothesis: str) -> float:
    """
    Calculate the Per-word Error Rate (PER) between the reference and hypothesis text.
    
    Args:
        reference (str): The ground truth transcription.
        hypothesis (str): The transcription produced by the model.
    
    Returns:
        float: The PER score.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Calculate substitutions, insertions, deletions
    s, d, i = 0, 0, 0
    min_len = min(len(ref_words), len(hyp_words))
    for idx in range(min_len):
        if ref_words[idx] != hyp_words[idx]:
            s += 1  # Substitution
    d += len(ref_words) - min_len  # Deletions (extra words in ref)
    i += len(hyp_words) - min_len  # Insertions (extra words in hypothesis)
    
    total_errors = s + d + i
    total_words = len(ref_words)
    return total_errors / total_words if total_words > 0 else 0.0

def save_file(results, file_name, format='tsv'):
    """
    Saves transcription results to a file in the specified format.

    Args:
        results (dict): Transcription results from Whisper.
        file_name (str): Output file name (without extension).
        format (str): File format to save as ('tsv', 'txt', 'srt').
    """
    writer = get_writer(format, './output/')
    writer(results, f'{file_name}.{format}')

def process_directory(directory: str):
    """
    Processes all audio files in a directory and transcribes them.
    
    Args:
        directory (str): Path to the directory containing audio files.
    """
    # Ensure output directory exists
    os.makedirs('./output/', exist_ok=True)
    
    # Iterate over all audio files in the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.wav', '.flac', '.mp3')):  # Supported formats
                audio_path = os.path.join(root, file)
                print(f"Processing: {audio_path}")
                
                # Transcribe the audio using Whisper model
                model_Transcription = get_transcribe(audio=audio_path)
                whisper_text = model_Transcription.get('text', '')
                
                # Get the ground truth transcription
                txt_Transcription = get_transcription_from_txt(audio_path)
                
                # Calculate WER and PER
                wer_score = calculate_wer(txt_Transcription, whisper_text)
                per_score = calculate_per(txt_Transcription, whisper_text)
                
                print(f"WER: {wer_score:.4f}")
                print(f"PER: {per_score:.4f}")
                # Save results in multiple formats
                base_file_name = os.path.splitext(file)[0]
                save_file(model_Transcription, base_file_name, 'txt')

                append_to_csv("metric", wer_score, per_score)
                # For debugging or further use
                #return wer_score, per_score

if __name__ == "__main__":
    # Path to the LibriSpeech dataset
    libri_path = 'data/LibriSpeech/dev-clean'
    process_directory(libri_path)
