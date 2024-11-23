import os
import torchaudio
import torch
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample, MelSpectrogram
import torch.nn.functional as F
# Set the path where the data should be stored
data_dir = './data'

# Function to download the LibriSpeech datasets if not already present
def download_data():
    # Download the dev-clean dataset if not present
    if not os.path.exists(os.path.join(data_dir, 'LibriSpeech', 'dev-clean')):
        print("Dev-clean dataset not found. Downloading...")
        os.makedirs(data_dir, exist_ok=True)
        torchaudio.datasets.LIBRISPEECH(root=data_dir, url="dev-clean", download=True)
    else:
        print("Dev-clean dataset already downloaded. Skipping download.")

    # Download the train-clean-100 dataset if not present
    if not os.path.exists(os.path.join(data_dir, 'LibriSpeech', 'train-clean-100')):
        print("Train-clean-100 dataset not found. Downloading...")
        torchaudio.datasets.LIBRISPEECH(root=data_dir, url="train-clean-100", download=True)
    else:
        print("Train-clean-100 dataset already downloaded. Skipping download.")

    # Download the test-clean dataset if not present
    if not os.path.exists(os.path.join(data_dir, 'LibriSpeech', 'test-clean')):
        print("Test-clean dataset not found. Downloading...")
        torchaudio.datasets.LIBRISPEECH(root=data_dir, url="test-clean", download=True)
    else:
        print("Test-clean dataset already downloaded. Skipping download.")

# Define a custom DataLoader class
class LibriSpeechDataLoader:
    def __init__(self, data_dir, batch_size=32, sample_rate=16000, n_mels=23):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        
        # Initialize the transformations
        self.transform = torch.nn.Sequential(
            Resample(orig_freq=8000, new_freq=self.sample_rate),  # Resample to 16kHz
            MelSpectrogram(n_mels=self.n_mels)  # Convert to Mel spectrogram
        )

        # Load the datasets
        self.dev_clean_dataset = torchaudio.datasets.LIBRISPEECH(root=data_dir, url="dev-clean", download=False)
        self.train_clean_100_dataset = torchaudio.datasets.LIBRISPEECH(root=data_dir, url="train-clean-100", download=False)
        self.test_clean_dataset = torchaudio.datasets.LIBRISPEECH(root=data_dir, url="test-clean", download=False)

    def process_audio(self, waveform, sample_rate):
        # Apply the transformation pipeline (e.g., resampling and MelSpectrogram)
        return self.transform(waveform)



    def collate_fn(self, batch):
        # Unpack the elements in each sample
        waveforms, sample_rates, labels, speaker_ids, file_ids, segment_lengths = zip(*batch)

        # Find the max length of the waveforms in the batch
        max_length = max([waveform.shape[1] for waveform in waveforms])

        # Pad waveforms to the max length
        padded_waveforms = []
        for waveform in waveforms:
            # Pad the waveform to the maximum length
            padding = max_length - waveform.shape[1]
            if padding > 0:
                # Pad at the end of the tensor
                padded_waveform = F.pad(waveform, (0, padding))
            else:
                # No padding needed, truncate if necessary
                padded_waveform = waveform[:, :max_length]
            padded_waveforms.append(padded_waveform)

        # Stack the waveforms into a single tensor
        waveforms = torch.stack(padded_waveforms, dim=0)

        # Labels, speaker_ids, file_ids, and segment_lengths remain unchanged
        labels = list(labels)  # Transcriptions
        speaker_ids = torch.tensor(speaker_ids)
        file_ids = torch.tensor(file_ids)
        segment_lengths = torch.tensor(segment_lengths)

        return waveforms, labels, speaker_ids, file_ids, segment_lengths




    def get_data_loaders(self):
        # Create the DataLoader for dev-clean
        dev_clean_loader = DataLoader(
            self.dev_clean_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        
        # Create the DataLoader for train-clean-100
        train_clean_loader = DataLoader(
            self.train_clean_100_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        
        # Create the DataLoader for test-clean
        test_clean_loader = DataLoader(
            self.test_clean_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        
        return dev_clean_loader, train_clean_loader, test_clean_loader

# Download the datasets
download_data()

# Initialize the DataLoader class
data_loader = LibriSpeechDataLoader(data_dir)

# Get the data loaders for each subset
dev_clean_loader, train_clean_loader, test_clean_loader = data_loader.get_data_loaders()


