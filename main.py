import os
import time
import IPython
from IPython.display import Audio
import csv
import numpy
import matplotlib
import matplotlib.pyplot as plt
import requests
import torch
import torchaudio
import fnmatch
from jiwer import wer

matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(torchaudio.list_audio_backends())
print(torch.__version__)
print(torchaudio.__version__)
print("torchaudio backend:", torchaudio.get_audio_backend())
print(device)

FOLDERS_ROOT = r"testData"
# SPEECH_FILE = r"PD_intelligibilityData\15 Young Healthy Control\Arianna P\B1LBULCAAS94M100120171053.wav"

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
# print("Sample Rate:", bundle.sample_rate)
# print("Labels:", bundle.get_labels())

model = bundle.get_model().to(device)
# print(model.__class__)
with open('_assets/reference_b.txt', 'r') as file:
    reference_b = file.read().replace('\n', '')

with open('_assets/reference_pr.txt', 'r') as file:
    reference_pr = file.read().replace('\n', '')


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        # print("emission length: ", emission.__len__())
        # print("emission: ", emission)
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        # print("indices: ", indices)
        indices = torch.unique_consecutive(indices, dim=-1)
        # print("indices: ", indices)
        indices = [i for i in indices if i != self.blank]
        # print("indices: ", indices)

        return "".join([self.labels[i] for i in indices])


data2csv = []
start_time = time.time()
for folder in os.listdir(os.path.join(FOLDERS_ROOT)):
    folder_time = time.time()
    print("working on: " + folder)
    for file in os.listdir(os.path.join(FOLDERS_ROOT, folder)):

        one_file_time_start = time.time()

        if fnmatch.fnmatch(file, 'B1*.wav'):
            recordingType = "B1"
        elif fnmatch.fnmatch(file, 'B2*.wav'):
            recordingType = "B2"
        elif fnmatch.fnmatch(file, 'PR1*.wav'):
            recordingType = "PR1"
        else:
            recordingType = "not in data set"

        if fnmatch.fnmatch(file, '*.wav'):
            print("... listening to: " + file)
            waveform, sample_rate = torchaudio.load(os.path.join(FOLDERS_ROOT, folder, file))
            waveform = waveform.to(device)

            if sample_rate != bundle.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

            with torch.inference_mode():
                features, _ = model.extract_features(waveform)

                """
                fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
                for i, feats in enumerate(features):
                    ax[i].imshow(feats[0].cpu())
                    ax[i].set_title(f"Feature from transformer layer {i + 1}")
                    ax[i].set_xlabel("Feature dimension")
                    ax[i].set_ylabel("Frame (time-axis)")
                plt.tight_layout()
                plt.show()
                """

            # FEATURE CLASSIFICATION (in logits, not probability)
            print("... doing classification")
            with torch.inference_mode():
                inference, _ = model(waveform)

                """
                # VISUALIZATION
                plt.imshow(emission[0].cpu().T)
                plt.title("Classification result")
                plt.xlabel("Frame (time-axis)")
                plt.ylabel("Class")
                plt.show()
                #print("Class labels:", bundle.get_labels())
                """
            # GENERATING TRANSCRIPTS
            print("... generating transcript")
            decoder = GreedyCTCDecoder(labels=bundle.get_labels())
            transcript = decoder(inference[0])

            # print("transcript length: ", transcript.__len__())
            # print("transcript: ", transcript)

            # PREP DATA FOR CSV
            error = 0
            get_unique_words = transcript.split('|')
            get_unique_words = (" ".join(get_unique_words)).lower()

            if fnmatch.fnmatch(file, 'PR*.wav'):
                error = wer(reference_pr, get_unique_words)
            elif fnmatch.fnmatch(file, 'B*.wav'):
                error = wer(reference_b, get_unique_words)

            # english references and wer
            if fnmatch.fnmatch(file, '*part1*.wav'):
                with open('_assets/reference_part1.txt', 'r') as fileref:
                    reference_part1 = fileref.read().replace('\n', '')
                    error = wer(reference_part1, get_unique_words)

            if fnmatch.fnmatch(file, '*part2*.wav'):
                with open('_assets/reference_part2.txt', 'r') as fileref:
                    reference_part2 = fileref.read().replace('\n', '')
                    error = wer(reference_part2, get_unique_words)

            if fnmatch.fnmatch(file, '*part3*.wav'):
                with open('_assets/reference_part3.txt', 'r') as fileref:
                    reference_part3 = fileref.read().replace('\n', '')
                    error = wer(reference_part3, get_unique_words)

            if fnmatch.fnmatch(file, '*part4*.wav'):
                with open('_assets/reference_part4.txt', 'r') as fileref:
                    reference_part4 = fileref.read().replace('\n', '')
                    error = wer(reference_part4, get_unique_words)

            transcript_data = [folder, recordingType, get_unique_words, error]
            data2csv.append(transcript_data)

            one_file_time = time.time()
            one_file_time = one_file_time - one_file_time_start
            print(file + " was done in " + str(round(one_file_time, 2)) + "s")

    print("folder finished in " + str(round(time.time() - folder_time, 2)) + "s")

print("FINISHED IN " + str(round(time.time() - start_time, 2)) + "s")

header = ['name', 'code', 'transcribed text', 'WER']
csvFILE = "_transcripts/test.csv"


def rows2csv(data2write):
    if not os.path.exists(csvFILE):
        os.makedirs("_transcripts", exist_ok=True)
    with open(csvFILE, "w", newline="", encoding='UTF8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(data2write)
        csvfile.close()


rows2csv(data2csv)
