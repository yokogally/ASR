# main.py ce ci est le main 
import os
import whisper
from whisper.utils import get_writer

# Load the Whisper model
model = whisper.load_model('base')

def get_transcribe(audio: str, language: str = 'en'):
    """
    Transcribes the given audio file using the Whisper model.
    
    Args:
        audio (str): Path to the audio file.
        language (str): Language of the audio. Defaults to 'en'.

    Returns:
        dict: The transcription results.
    """
    return model.transcribe(audio=audio, language=language, verbose=True)

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
                
                # Transcribe the audio
                result = get_transcribe(audio=audio_path)
                print('-' * 50)
                print(result.get('text', ''))
                
                # Save results in multiple formats
                base_file_name = os.path.splitext(file)[0]
                save_file(result, base_file_name, 'tsv')
                save_file(result, base_file_name, 'txt')
                save_file(result, base_file_name, 'srt')

if __name__ == "__main__":
    # Path to the LibriSpeech dataset
    libri_path = 'data/LibriSpeech/dev-clean'
    process_directory(libri_path)
