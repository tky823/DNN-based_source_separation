#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import glob
import random

import torchaudio

from utils.utils import set_seed

parser = argparse.ArgumentParser("Prepare json file of LibriSpeech for audio source separation")

parser.add_argument('--librispeech_root', type=str, default=None, help='Path for LibriSpeech dataset ROOT directory')
parser.add_argument('--wav_root', type=str, default=None, help='Path for wav ROOT directory')
parser.add_argument('--json_path', type=str, default=None, help='Path for json file')
parser.add_argument('--n_sources', type=int, default=2, help='Number of mixtures')
parser.add_argument('--sample_rate', '-sr', type=int, default=16000, help='Sampling rate')
parser.add_argument('--duration', type=float, default=2, help='Duration')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

def main(args):
    set_seed(args.seed)

    speakers_path = os.path.join(args.librispeech_root, "SPEAKERS.TXT")
    samples = int(args.sample_rate * args.duration)

    json_data = make_json_data(args.wav_root, args.json_path, speakers_path=speakers_path, samples=samples, n_sources=args.n_sources)

    with open(args.json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

def make_json_data(wav_root, json_path, speakers_path, n_sources=2, samples=32000):
    json_dir = os.path.dirname(json_path)
    folder_name = os.path.basename(json_dir) # "train-clean-100", "dev-clean" or "test-clean"
    os.makedirs(json_dir, exist_ok=True)

    with open(speakers_path) as f:
        lines = f.readlines()
        lines = lines[12:]

    print(folder_name)

    meta_data = {}

    for line in lines:
        line = line.replace('\n', '').replace(' ', '')
        speaker_ID, sex, subset, minutes, name = line.split('|', maxsplit=4)

        if subset != folder_name:
            continue

        meta_data[speaker_ID] = []

        speech_IDs = sorted(glob.glob(os.path.join(wav_root, folder_name, speaker_ID, "*")))
        speech_IDs = [os.path.basename(speech_ID) for speech_ID in speech_IDs]

        for speech_ID in speech_IDs:
            wav_names = sorted(glob.glob(os.path.join(wav_root, folder_name, speaker_ID, speech_ID, "*.flac")))
            wav_names = [os.path.basename(wav_name) for wav_name in wav_names]

            for wav_name in wav_names:
                utterance_ID, _ = os.path.splitext(wav_name)
                relative_path = os.path.join(folder_name, speaker_ID, speech_ID, "{}.flac".format(utterance_ID))
                wav_path = os.path.join(wav_root, relative_path)
                wave, sample_rate = torchaudio.load(wav_path) # wave, sample_rate = sf.read(wav_path)
                T = wave.size(1)

                for idx in range(0, T, samples):
                    if idx + samples > T:
                        break
                    meta_data[speaker_ID].append({
                        'speaker-ID': speaker_ID,
                        'speech-ID': speech_ID,
                        'utterance-ID': utterance_ID,
                        'sex': sex,
                        'start': idx,
                        'end': idx + samples,
                        'path': relative_path
                    })
        print("Speaker {}: {} segments".format(speaker_ID, len(meta_data[speaker_ID])))

    json_data = []

    while len(meta_data.keys()) >= n_sources:
        possible_speaker_IDs = meta_data.keys()
        speaker_IDs = random.sample(possible_speaker_IDs, n_sources)
        data = {
            'sources': {}
        }

        for source_idx in range(n_sources):
            speaker_ID = speaker_IDs[source_idx]
            idx = random.randint(0, len(meta_data[speaker_ID])-1)
            meta = meta_data[speaker_ID].pop(idx)

            if len(meta_data[speaker_ID]) == 0:
                del meta_data[speaker_ID]

            data['sources']['source-{}'.format(source_idx)] = meta

        json_data.append(data)

    return json_data

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
