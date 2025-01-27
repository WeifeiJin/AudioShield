import os
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
import utils
from utils import get_text
from text.symbols import symbols
from models import SynthesizerTrn
import pickle
import random

class LibriDataset(Dataset):
    def __init__(self, data_list):
        self.file_list = data_list
        self.labels = ['_', "'", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                       'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ']
        self.labels_map = dict([(self.labels[i], i) for i in range(len(self.labels))])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        wav, txt, z, y = self.file_list[idx]
        txt_tensor = self.parse_transcript(txt)
        return wav, txt_tensor, z, y

    def parse_transcript(self, transcript):
        transcript = transcript.replace('\n', '')
        transcript_n = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        transcript_tensor = torch.tensor(transcript_n, dtype=torch.long)
        return transcript_tensor


class TextAudioCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[2].size(2) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[1]) for x in batch])
        max_z_len = max([x[2].size(2) for x in batch])
        max_wav_len = max([x[0].size(0) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        z_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        z_padded = torch.FloatTensor(len(batch), batch[0][2].size(1), max_z_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        z_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[1]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            z = row[2]
            z_padded[i, :, :z.size(2)] = z[0, :, :]
            z_lengths[i] = z.size(2)

            wav = row[0]
            wav_padded[i, :, :wav.size(0)] = wav
            wav_lengths[i] = wav.size(0)


        return wav_padded, wav_lengths, text_padded, text_lengths, z_padded, z_lengths


def get_loader(hparams):
    with open(hparams.data.libridata, "rb") as f:
        data_list = pickle.load(f)
    libri_dataset = LibriDataset(data_list)
    data_loader = DataLoader(libri_dataset, batch_size=hparams.optimization.batch_size, shuffle=True, collate_fn=TextAudioCollate())
    return data_loader

def pre_process(hparams):
    with open(hparams.data.libridata, "rb") as f:
        data_list = pickle.load(f)

    hps = utils.get_hparams_from_file("../configs/vctk_base.json")
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda()
    net_g.eval()
    utils.load_checkpoint(hparams.model.vits, net_g, None)

    new_dataset = []
    for idx, now in enumerate(data_list):
        wav_path, txt = now
        waveform, sample_rate = torchaudio.load('../' + wav_path)
        waveform = waveform.squeeze(0)

        src_tst = get_text(txt, hps)
        x_tst = src_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([src_tst.size(0)]).cuda()
        spk = random.randint(0, 100)
        sid = torch.LongTensor([spk]).cuda()
        z_x, y_mask, g = net_g.get_latent_feat(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=1.0,
                                   length_scale=1)
        new_dataset.append((waveform, txt, z_x, y_mask))

    with open(hparams.data.libri_latent_data, "wb") as f:
        pickle.dump(new_dataset, f)


if __name__ == '__main__':
    hparams = utils.get_hparams_from_file("../configs/protection.json")
    pre_process(hparams)

