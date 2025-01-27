import os
import torchaudio
import random
import pickle
import utils

def judge_wav(wav_path, start_time=1, end_time=5):
    waveform, sample_rate = torchaudio.load(wav_path)
    duration = waveform.size(1) / sample_rate
    if start_time <= duration <= end_time:
        return True
    else:
        return False

def process_librispeech(root_dir):
    dataset = []
    spks = os.listdir(root_dir)
    for spk in spks:
        spk_dir = os.path.join(root_dir, spk)
        stys = os.listdir(spk_dir)
        for sty in stys:
            sty_dir = os.path.join(spk_dir, sty)
            wavs = os.listdir(sty_dir)
            txt_file = spk + '-' + sty + '.trans.txt'
            txt_file = os.path.join(sty_dir, txt_file)
            transcriptions = open(txt_file).read().strip().split("\n")
            transcriptions = {t.split()[0].split("-")[-1]: " ".join(t.split()[1:]) for t in transcriptions}
            for wav in wavs:
                wav_path = os.path.join(sty_dir, wav)
                if wav.endswith(".flac") and judge_wav(wav_path):
                    key = wav_path.replace(".flac", "").split("-")[-1]
                    assert key in transcriptions, "{} is not in the transcriptions".format(key)
                    dataset.append((wav_path, transcriptions[key]))
    return dataset

if __name__ == '__main__':
    hparams = utils.get_hparams_from_file("../configs/protection.json")
    dev_clean_dir = hparams.data.libri_dataset_root
    dataset = process_librispeech(dev_clean_dir)
    print(len(dataset))

    random.shuffle(dataset)
    new_list = dataset[:hparams.data.training_dataset_size]

    with open(hparams.data.libridata, "wb") as f:
        pickle.dump(new_list, f)