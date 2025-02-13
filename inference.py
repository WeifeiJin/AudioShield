import torch
import torchaudio
from torchaudio.transforms import Resample
import soundfile as sf
from models import SynthesizerTrn
from text.symbols import symbols
import utils
from mel_processing import spectrogram_torch

class AudioShield:
    def __init__(self, args):
        """Model Settings"""
        self.filter_length = 1024
        self.hop_length = 256
        self.sampling_rate = 22050
        self.win_length = 1024
        self.args = args
        self.device = torch.device(self.args.device if torch.cuda.is_available() else 'cpu')
        self.ptb_path = args.ptb_path
        self.output_dir = args.output_dir
        self.hps = utils.get_hparams_from_file("./configs/vctk_base.json")
        self.net_g = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model).to(self.device)
        self.net_g.eval()
        utils.load_checkpoint("pretrained/vits/pretrained_vctk.pth", self.net_g, None)
        self.resample_22k = Resample(orig_freq=22050, new_freq=16000)
        self.resample_44k = Resample(orig_freq=44100, new_freq=16000)
        print(f'device: {self.device}')

        self.ptb = self.load_ptb(self.ptb_path).detach().to(self.device)
        self.ptb.requires_grad_(False)

    def load_ptb(self, path):
        ts_dict = torch.load(path)
        return ts_dict['MAP']

    def convert_1(self, y, y_lengths, sid_src, sid_tgt):
        assert self.net_g.n_speakers > 0, "n_speakers have to be larger than 0."
        g_src = self.net_g.emb_g(sid_src).unsqueeze(-1)
        g_tgt = self.net_g.emb_g(sid_tgt).unsqueeze(-1)
        z, m_q, logs_q, y_mask = self.net_g.enc_q(y, y_lengths, g=g_src)
        z_p = self.net_g.flow(z, y_mask, g=g_src)
        z_hat = self.net_g.flow(z_p, y_mask, g=g_tgt, reverse=True)
        return z_hat, y_mask, g_tgt

    def perturb_z(self, tensor1, tensor2):
        repeat_times = tensor2.size(2) // tensor1.size(2) + 1
        extended_tensor1 = tensor1.repeat(1, 1, repeat_times)[:, :, :tensor2.size(2)]
        result = tensor2 + extended_tensor1 #* 0.5
        return result

    def convert_2(self, z_hat, y_mask, g_tgt):
        o_hat = self.net_g.dec(z_hat * y_mask, g=g_tgt)
        return o_hat

    def run_conversion_tensor(self, audio_tensor, sid):
        if audio_tensor.ndim > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

        if self.args.sampling_rate != 16000:
            print(f'sample_rate = {self.args.sampling_rate}')
            resampler = Resample(orig_freq=self.args.sampling_rate, new_freq=16000)
            audio_tensor = resampler(audio_tensor)

        spec = spectrogram_torch(audio_tensor, self.filter_length,
                                 self.sampling_rate, self.hop_length, self.win_length,
                                 center=False)
        spec = torch.squeeze(spec, 0).unsqueeze(0).to(self.device)
        spec_length = torch.LongTensor([spec.shape[2]]).to(self.device)
        sid = torch.LongTensor([sid]).to(self.device)
        z_hat, y_mask, g_tgt = self.convert_1(spec, spec_length, sid, sid)
        z_adv = self.perturb_z(self.ptb, z_hat)
        converted_audio = self.convert_2(z_adv, y_mask, g_tgt)
        wav = converted_audio.squeeze(0).squeeze(0)
        x = self.resample_22k(wav.cpu())
        return x.detach().cpu().numpy()

    def run_conversion_spec(self, spec, spec_length, sid):
        z, y_mask, g = self.convert_1(spec, spec_length, sid, sid)
        z_adv = self.perturb_z(self.ptb, z)
        converted_audio = self.convert_2(z_adv, y_mask, g)
        wav = converted_audio.squeeze(0).squeeze(0)
        x = self.resample_22k(wav.cpu())
        return x.detach().cpu().numpy()

