import random
import os
import torch
import torchaudio
import torchaudio.transforms
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import utils
from utils import get_text
from models import SynthesizerTrn
from text.symbols import symbols
import commons
from torch.cuda.amp import autocast
import time
import datetime
from deepspeech_pytorch.configs.inference_config import TranscribeConfig
from deepspeech_pytorch.utils import load_decoder, load_model
from datasets import librispeech
from torchaudio.transforms import Resample
import soundfile as sf
import argparse

class Solver:
    def __init__(self, args):
        """Model Settings"""
        self.args = args
        self.target_layer = 9
        self.segment_size = 48000
        self.device = torch.device(self.args.device if torch.cuda.is_available() else 'cpu')
        self.input_txt = self.args.input_txt
        self.output_dir = self.args.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.hps = utils.get_hparams_from_file("./configs/vctk_base.json")
        self.protection_hps = utils.get_hparams_from_file('./configs/protection.json')
        self.net_g = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model).to(self.device)
        utils.load_checkpoint(self.protection_hps.model.vits, self.net_g, None)
        self.tgt_tst = get_text(self.args.tgt_text.lower(), self.hps)

        """Protection Settings"""
        self.training_iters = self.args.training_iters
        self.tgt_text = self.args.tgt_text
        ########################### for DeepSpeech #################################
        self.labels = ['_', "'", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                       'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ']
        self.labels_map = dict([(self.labels[i], i) for i in range(len(self.labels))])
        self.criterion = nn.CTCLoss(blank=self.labels.index('_'), reduction='sum', zero_infinity=True)
        tgt_model_path = protection_hps.model.deepspeech
        self.tgt_model = load_model(device=self.device, model_path=tgt_model_path)
        self.deep_speech_cfg = TranscribeConfig()
        self.decoder = load_decoder(labels=self.tgt_model.labels, cfg=self.deep_speech_cfg.lm)
        self.sample_rate = self.protection_hps.audio.sample_rate
        self.window_size = self.protection_hps.audio.window_size
        self.window_stride = self.protection_hps.audio.window_stride
        self.precision = 32
        self.wav2spec = torchaudio.transforms.Spectrogram(n_fft=int(self.sample_rate * self.window_size),
                                                win_length=int(self.sample_rate * self.window_size),
                                                hop_length=int(self.sample_rate * self.window_stride)).to(self.device)
        self.mfcc_transform = torchaudio.transforms.MFCC(sample_rate=self.hps.data.sampling_rate).to(self.device)

        self.resample = Resample(orig_freq=22050, new_freq=16000).to(self.device)



    def parse_transcript(self, transcript):
        transcript = transcript.replace('\n', '')
        transcript_n = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        transcript_tensor = torch.tensor(transcript_n, dtype=torch.long)
        return transcript_tensor

    def calc_spectrogram(self, y):
        spectrogram = self.wav2spec(y.to(self.device, torch.float32))

        magnitudes = torch.abs(spectrogram)
        spec = torch.log1p(magnitudes)

        mean = spec.mean()
        std = spec.std()
        normalized_spec = (spec - mean) / std

        return normalized_spec

    def spec2trans(self, spec, input_sizes):
        hs = None
        spec = spec.contiguous()
        spect = spec.view(spec.size(0), 1, spec.size(1), spec.size(2))
        spect = spect.to(self.device)

        with autocast(enabled=self.precision == 16):
            out, output_sizes, hs = self.tgt_model(spect, input_sizes, hs)
        decoded_output, _ = self.decoder.decode(out)
        txts = []
        for txt_l in decoded_output:
            txts.append(txt_l[0])
        return out, txts

    def get_test_transcription(self, x, model_name):
        if 'deepspeech' in model_name:
            with torch.no_grad():
                self.tgt_model.to(self.device)
                _, txts = self.calc_ds_loss(x.unsqueeze(0))
                txt = txts[0]
        return txt


    def get_loss(self, x, model_name):
        """calc loss using the given local model"""
        if 'deepspeech' in model_name:
            loss, txts = self.calc_ds_loss(x)
        # can extension to more local models

        return loss, txts


    def calc_ds_loss(self, syn_audio):
        trans_tgt = self.parse_transcript(self.tgt_text.upper()).to(self.device)

        batch_size = syn_audio.shape[0]
        trans_tgt = trans_tgt.repeat(batch_size).view(batch_size, -1)
        x = syn_audio.to(self.device)  # / 32768.0
        out_spec = self.calc_spectrogram(x).to(self.device)

        input_lengths = torch.LongTensor(out_spec.size(0) * [out_spec.size(2)]).to(self.device)

        out_trans, txt = self.spec2trans(out_spec, input_lengths)

        input_lengths = torch.tensor(out_trans.size(0) * [out_trans.size(1)]).to(self.device)
        target_lengths = torch.tensor(batch_size * [trans_tgt.size(1)]).to(self.device)

        out_trans = out_trans.transpose(0, 1)
        out_trans = out_trans.log_softmax(-1)

        asr_loss = self.criterion(out_trans, trans_tgt, input_lengths, target_lengths)
        del out_spec
        del out_trans
        return asr_loss, txt

    def target_audio_selection(self):
        tgt_tst = self.tgt_tst.to(self.device).unsqueeze(0)
        tgt_tst_len = torch.LongTensor([self.tgt_tst.size(0)]).to(self.device)

        n = self.protection_hps.optimization.target_audio_number
        s = self.protection_hps.optimization.decay_rate
        min_loss = 2147483647
        z_selected = None
        t_selected = None
        g_selected = None
        for i in range(n):
            # noise_scale = random.uniform(0, 1)
            # noise_scale_w = random.uniform(0, 1)
            tid = torch.LongTensor([random.randint(0, 100)]).to(self.device)
            z_tgt, t_mask, g = self.net_g.get_latent_feat(tgt_tst, tgt_tst_len, sid=tid, noise_scale=.667, noise_scale_w=0.8,
                                     length_scale=1)
            tgt_audio = self.net_g.decode_latent_feat(z_tgt, t_mask, g)
            tgt_audio = tgt_audio.squeeze(0)
            loss_now, txts = self.get_loss(tgt_audio, "deepspeech")
            w = 1.0
            txt_now = txts[0]
            while txt_now == self.tgt_text:
                w = w * s
                tgt_audio = self.net_g.decode_latent_feat(w * z_tgt, t_mask, g)
                tgt_audio = tgt_audio.squeeze(0).squeeze(0)
                loss_now, txts = self.get_loss(tgt_audio, "deepspeech")
                txt_now = txts[0]
            # w = w / s
            tgt_audio = self.net_g.decode_latent_feat(w * z_tgt, t_mask, g)
            tgt_audio = tgt_audio.squeeze(0)
            loss_now, _ = self.get_loss(tgt_audio, "deepspeech")
            print(f'w = {w}, loss = {loss_now}')
            if loss_now < min_loss:
                min_loss = loss_now
                z_selected = z_tgt
                t_selected = t_mask
                g_selected = g
        return z_selected, t_selected, g_selected

    def train(self):
        print('Start training iteration...')
        tot_loss = 0

        z_tgt, t_mask, g = self.target_audio_selection()
        uap = z_tgt.detach().clone().to(self.device)
        uap_len = uap.shape[2]
        uap = nn.Parameter(uap.requires_grad_(True))

        org_lr = 0.001
        optimizer = torch.optim.Adam([uap], org_lr, (0.9, 0.999), weight_decay=1e-4)

        start_time = time.time()

        train_models = ['deepspeech']
        test_models = ['deepspeech']
        loss_dict = {"ASR_loss": None, "sim_loss": None}

        max_epochs = 3
        train_loader = librispeech.get_loader(self.protection_hps)
        all_data = []
        num_all = 0
        for idx, batch in enumerate(train_loader):
            wav, wav_lengths, text, text_lengths, z_batch, z_lengths = batch
            num_all += len(z_batch)
            all_data.append((z_batch, z_lengths))

        tau = self.args.tau
        for epoch in range(max_epochs):
            for idx, batch in enumerate(train_loader):
                _, _, text, text_lengths, z_batch, z_lengths = batch
                y_mask = torch.unsqueeze(commons.sequence_mask(z_lengths, None), 1).to(z_batch.dtype)
                z_x = z_batch.detach().clone().to(self.device)
                y_mask = y_mask.detach().clone().to(self.device)
                n = z_x.shape[2]
                if n < uap_len:
                    continue
                start_index = torch.randint(0, n - uap_len + 1, (1,)).item()
                z_x = z_x[:, :, start_index: start_index + uap_len]

                for i in range(self.training_iters):
                    train_model = random.choice(train_models)

                    noise = torch.randn_like(z_x).to(self.device) * 2.0
                    z = z_x + uap + noise
                    y_mask = y_mask[:, :, :uap_len]
                    converted_audio = self.net_g.decode_latent_feat(z, y_mask, g)
                    converted_audio = converted_audio.squeeze(1)

                    asr_loss, _ = self.get_loss(converted_audio, train_model)

                    ts1 = z_tgt.view(z_tgt.shape[1], -1)
                    ts2 = uap.view(uap.shape[1], -1)
                    cos_sim = torch.nn.functional.cosine_similarity(ts1, ts2, dim=1)
                    sim_loss = 1.0 - torch.mean(cos_sim)

                    loss = asr_loss + 50 * sim_loss

                    loss_dict["ASR_loss"] = asr_loss.item()
                    loss_dict["sim_loss"] = sim_loss.item()

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)


                    uap.grad = torch.sign(uap.grad)
                    optimizer.step()
                    uap.data = torch.clamp(uap.data, -tau, tau)

                    tot_loss += loss.item()

                    torch.cuda.empty_cache()

                    """logs"""
                    if (i + 1) % 10 == 0:
                        now_loss = tot_loss / 10
                        tot_loss = 0
                        et = time.time() - start_time
                        et = str(datetime.timedelta(seconds=et))[:-7]
                        out_log = f'Elapsed [{et}], loss [{i + 1}] = {now_loss:.7f}'
                        for key, value in loss_dict.items():
                            out_log += f', {key} = {value:.7f}'
                        print(out_log)

                    if (i + 1) % 100 == 0:
                        """test"""
                        with torch.no_grad():
                            batch_size = z_x.shape[0]
                            r_idx = random.randint(0, batch_size - 1)
                            z = z_x[r_idx].unsqueeze(0) + uap
                            converted_audio = self.net_g.decode_latent_feat(z, y_mask[r_idx], g)
                            converted_audio = converted_audio.squeeze(0).squeeze(0)
                            test_transes = []
                            for test_model in test_models:
                                test_trans = self.get_test_transcription(converted_audio, test_model)
                                test_transes.append((test_model, test_trans))
                        out_log = f'Test log:'
                        for test_res in test_transes:
                            out_log += f'\n{test_res[0]}: {test_res[1]}'
                        print(out_log)

                    del converted_audio
                    del loss
                    torch.cuda.empty_cache()

                del z_batch
                del z_lengths
                self.save(uap, z_tgt, t_mask, epoch, idx)

    def save(self, uap, z_tgt, t_mask, epoch, iters):
        output_file = os.path.join(self.output_dir, f'{epoch}_{iters}_{self.tgt_text}.pth')
        tensor_dict = {'LS-TUAP': uap, 'z_tgt': z_tgt, 'y_mask': t_mask}
        torch.save(tensor_dict, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AudioShield.')

    parser.add_argument('--training_iters', type=int, default=5000, help='Number of training iterations.')
    parser.add_argument('--tau', type=float, default=0.5, help='Tau value for the perturbation.')
    parser.add_argument('--device', type=str, default='cuda', help='Device for training.')
    parser.add_argument('--tgt_text', type=str, default='OPEN THE DOOR', help='Target text for training.')
    parser.add_argument('--output_dir', type=str, default='./result', help='Directory to save output files.')

    args = parser.parse_args()

    solver = Solver(args)

    solver.train()
