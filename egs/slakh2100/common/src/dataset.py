import os
import glob

import yaml
import torch
import torchaudio

SAMPLE_RATE_SLAKH2100 = 16000
__sources__ = [
    "Piano", "Chromatic Percussion", "Organ", "Guitar", "Bass", "Strings", "Strings (continued)", "Brass", "Reed", "Pipe", "Synth Lead", "Synth Pad", "Sound Effects", "Ethnic", "Percussive", "Sound effects", "Drums"
]

class Slakh2100Dataset(torch.utils.data.Dataset):
    def __init__(self, slakh2100_root, sample_rate=SAMPLE_RATE_SLAKH2100, sources=__sources__, target=None):
        """
        Args:
            slakh2100_root <str>: Path to Slakh2100 root.
            sample_rate <int>: Sampling rate.
            sources <list<str>>: Sources for mixture.
            target <str> or <list<str>>: Target source(s). If None is given, `sources` is used by default.
        """
        _assert_sample_rate(sample_rate)

        if target is not None:
            if type(target) is list:
                for _target in target:
                    assert _target in sources, "`sources` doesn't contain target {}".format(_target)
            else:
                assert target in sources, "`sources` doesn't contain target {}".format(target)
        else:
            target = sources

        self.slakh2100_root = os.path.abspath(slakh2100_root)
        self.tracks = []

        self.sources = sources
        self.target = target

class WaveDataset(Slakh2100Dataset):
    def __init__(self, slakh2100_root, sample_rate=SAMPLE_RATE_SLAKH2100, sources=__sources__, target=None):
        """
        Args:
            slakh2100_root <int>: Path to Slakh2100.
            sample_rate: Sampling frequency. Default: 16000.
            sources <list<str>>: Sources included in mixture.
            target <str> or <list<str>>: Target source(s)
        """
        super().__init__(slakh2100_root, sample_rate=sample_rate, sources=sources, target=target)

        self.json_data = None

    def __getitem__(self, idx):
        """
        Args:
            idx <int>: index
        Returns:
            waveform_mixture <torch.Tensor>: (1, n_mics, T) if `target` is list, otherwise (n_mics, T)
            waveform_target <torch.Tensor>: (len(target), n_mics, T) if `target` is list, otherwise (n_mics, T)
            name <str>: Artist and title of track
        """
        data = self.json_data[idx]

        trackID = data["trackID"]
        track = self.tracks[trackID]
        name = track["name"]
        paths = track["path"]
        start = data["start"]
        samples = data["samples"]

        if set(self.sources) == set(__sources__):
            waveform_mixture, _ = torchaudio.load(paths["mixture"], frame_offset=start, num_frames=samples)
        else:
            waveform_sources = []

            for _source in self.sources:
                waveforms_source = []
                for source_path in paths[_source]:
                    waveform_source, _ = torchaudio.load(source_path, frame_offset=start, num_frames=samples)
                    waveforms_source.append(waveform_source)

                waveforms_source = torch.stack(waveforms_source, dim=0)
                waveforms_source = waveforms_source.sum(dim=0)
                waveform_sources.append(waveforms_source)

            waveform_sources = torch.stack(waveform_sources, dim=0)
            waveform_mixture = waveform_sources.sum(dim=0)

        if type(self.target) is list:
            raise NotImplementedError
        else:
            waveforms_target = []
            for target_path in paths[self.target]:
                waveform_target, _ = torchaudio.load(target_path, frame_offset=start, num_frames=samples)
                waveforms_target.append(waveform_target)
            waveforms_target = torch.stack(waveforms_target, dim=0)
            waveform_target = waveforms_target.sum(dim=0)

        return waveform_mixture, waveform_target, name

    def __len__(self):
        return len(self.json_data)

class WaveTrainDataset(WaveDataset):
    def __init__(self, slakh2100_root, sample_rate=SAMPLE_RATE_SLAKH2100, samples=4*SAMPLE_RATE_SLAKH2100, overlap=None, sources=__sources__, target=None):
        """
        Args:
            include_valid <bool>: Include validation data for training.
        """
        super().__init__(slakh2100_root, sample_rate=sample_rate, sources=sources, target=target)

        track_dirs = sorted(glob.glob(os.path.join(slakh2100_root, "train", "*")))
        names = [
            os.path.basename(name) for name in track_dirs
        ]

        if overlap is None:
            overlap = samples // 2

        self.tracks = []
        self.json_data = []

        for trackID, name in enumerate(names):
            mixture_path = os.path.join(slakh2100_root, "train", name, "mix.flac")
            yaml_path = os.path.join(slakh2100_root, "train", name, "metadata.yaml")

            with open(yaml_path) as f:
                yaml_data = yaml.safe_load(f)

            if type(target) is str:
                stemIDs = {
                    source: [] for source in sources
                }
                inst_classes = set()

                for stemID, data in yaml_data["stems"].items():
                    inst_class = data["inst_class"]
                    inst_classes.add(inst_class)
                    stemIDs[inst_class].append(stemID)

                if not target in inst_classes:
                    continue

                for inst_class in inst_classes:
                    if len(stemIDs[inst_class]) == 0:
                        del stemIDs[inst_class]
            else:
                raise NotImplementedError

            audio_info = torchaudio.info(mixture_path)
            track_samples = audio_info.num_frames
            track = {
                "name": name,
                "samples": track_samples,
                "path": {
                    "mixture": mixture_path
                }
            }

            for inst_class, inst_stemID in stemIDs.items():
                track["path"][inst_class] = []
                for stemID in inst_stemID:
                    source_path = os.path.join(slakh2100_root, "train", name, "{}.flac".format(stemID))
                    track["path"][inst_class].append(source_path)

            self.tracks.append(track)

            for start in range(0, track_samples, samples - overlap):
                if start + samples >= track_samples:
                    break
                data = {
                    "trackID": trackID,
                    "start": start,
                    "samples": samples,
                }
                self.json_data.append(data)

    def __getitem__(self, idx):
        """
        Returns:
            waveform_mixture <torch.Tensor>: (1, n_mics, T) if `target` is list, otherwise (n_mics, T)
            waveform_target <torch.Tensor>: (len(target), n_mics, T) if `target` is list, otherwise (n_mics, T)
        """
        waveform_mixture, waveform_target, _ = super().__getitem__(idx)
        return waveform_mixture, waveform_target

def _assert_sample_rate(sample_rate):
    assert sample_rate == SAMPLE_RATE_SLAKH2100, "sample_rate should be {}.".format(SAMPLE_RATE_SLAKH2100)