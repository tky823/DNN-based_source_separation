import torch

class SpeakerToIndex:
    def __init__(self):
        self.table = {}
    
    def add_speaker(self, speaker_id: str):
        if not speaker_id in self.table:
            self.table[speaker_id] = len(self.table)
    
    def __call__(self, speaker_id: str, as_tensor=True):
        spk_idx = self.table[speaker_id]
        
        if as_tensor:
            spk_idx = torch.tensor(spk_idx)
        
        return spk_idx
    
    def __len__(self):
        return len(self.table)