from utils.embedding import SpeakerToIndex

def create_spk_to_idx(list_path):
    spk_to_idx = SpeakerToIndex()

    with open(list_path) as f:
        for line in f:
            ID = line.strip()
            spk_ids = ID.split('_')[0::2]

            for spk_id in spk_ids:
                spk_to_idx.add_speaker(spk_id)
    
    return spk_to_idx
