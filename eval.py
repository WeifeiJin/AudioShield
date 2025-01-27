import os
import pickle
from tqdm import tqdm
import torch
from inference import Attacker
import re
import soundfile as sf
from api.iflytek_ASR import iflytek_ASR
from api.Aliyun_ASR import aliyun_ASR
from api.google_ASR import google_ASR
from api.amazon_ASR import amazon_ASR
import uuid

def clean_text(text, punctuation_to_remove):

    pattern = f"[{re.escape(punctuation_to_remove)}]"
    text = re.sub(pattern, '', text)
    text = text.lower()
    return text

def calc_wer(reference, hypothesis, punctuation_to_remove=".,!?\"'()-"):

    reference = clean_text(reference, punctuation_to_remove)
    hypothesis = clean_text(hypothesis, punctuation_to_remove)

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i - 1][j] + 1,
                          d[i][j - 1] + 1,
                          d[i - 1][j - 1] + cost)

    wer_value = d[len(ref_words)][len(hyp_words)] / len(ref_words)
    return wer_value

def calc_cer(reference, hypothesis, punctuation_to_remove=".,!?\"'()-"):

    reference = clean_text(reference, punctuation_to_remove)
    hypothesis = clean_text(hypothesis, punctuation_to_remove)

    ref_chars = list(reference)
    hyp_chars = list(hypothesis)

    d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]

    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j

    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i - 1][j] + 1,
                          d[i][j - 1] + 1,
                          d[i - 1][j - 1] + cost)

    cer_value = d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)
    return cer_value

def calc_metrics(res_list):
    total_cer = 0
    total_wer = 0
    cnt = 0
    num_texts = 0
    for reference, hypothesis in res_list:
        if hypothesis == '' or hypothesis == 'NA':
            continue
        num_texts += 1
        wer = calc_wer(reference, hypothesis)
        cer = calc_cer(reference, hypothesis)
        if cer >= 0.5:
            cnt += 1
        total_cer += cer
        total_wer += wer

    average_sr = cnt / num_texts
    average_cer = total_cer / num_texts
    average_wer = total_wer / num_texts
    return cnt, num_texts, average_sr, average_cer, average_wer

def save_wav(output_dir, wav, iters):
    output_file = os.path.join(output_dir, f'AudioShield_{iters}.wav')

    sf.write(output_file, wav, 16000)

def get_test_transcription(self, x, model_name):
    if 'deepspeech' in model_name:
        with torch.no_grad():
            self.tgt_model.to(self.device)
            _, txts = self.calc_ds_loss(x.unsqueeze(0))
            txt = txts[0]
    elif 'wav2vec2' in model_name:
        re_x = self.resample_22k(x.cpu())
        txt = self.wav2vec2_pipe(re_x.detach().cpu().numpy())['text']
    elif 'whisper' in model_name:
        x = self.resample_22k(x.cpu())
        txt = self.whisper_pipe(x.detach().cpu().numpy())['text']
    elif 'iflytek' in model_name:
        x = self.resample_22k(x.cpu())
        audio = x.detach().cpu().numpy()
        path = f"cache/tmp_{uuid.uuid4()}.wav"
        sf.write(path, audio, 16000)
        txt = iflytek_ASR(path)
        os.remove(path)
    elif 'aliyun' in model_name:
        x = self.resample_22k(x.cpu())
        audio = x.detach().cpu().numpy()
        path = f"cache/tmp_{uuid.uuid4()}.wav"
        sf.write(path, audio, 16000)
        txt = aliyun_ASR(path)
        os.remove(path)
    elif 'google' in model_name:
        x = self.resample_22k(x.cpu())
        audio = x.detach().cpu().numpy()
        path = f"cache/tmp_{uuid.uuid4()}.wav"
        sf.write(path, audio, 16000)
        txt = google_ASR(path)
        os.remove(path)
    elif 'amazon' in model_name:
        x = self.resample_22k(x.cpu())
        audio = x.detach().cpu().numpy()
        path = f"cache/tmp_{uuid.uuid4()}.wav"
        sf.write(path, audio, 16000)
        txt = amazon_ASR(path)
        os.remove(path)
    return txt

def test_dataset(dataset, test_models, attacker):
    res = {}
    trans_ans = {}

    for test_model in test_models:
        trans_ans[test_model] = []
    for i, batch in tqdm(enumerate(dataset)):
        wav, spec, sid, txt = batch
        spec = spec.unsqueeze(0).to(attacker.device)
        spec_length = torch.LongTensor([spec.shape[2]]).to(attacker.device)
        sid = torch.LongTensor([sid]).to(attacker.device)

        audio = attacker.run_conversion_spec(spec, spec_length, sid)
        save_wav(attacker.args.output_dir, audio.detach().cpu().numpy(), i)

        txts_out = []

        for test_model in test_models:
            trans = get_test_transcription(wav, test_model)
            trans_ans[test_model].append((txt, trans))
            txts_out.append(trans)
        out_log = f'Example {i}:\norigin: {txt}'

        for j, test_model in enumerate(test_models):
            out_log += f'{test_model}: {txts_out[j]}\n'
        print(out_log)
    for test_model in test_models:
        cnt, num, sr, cer, wer = calc_metrics(trans_ans[test_model])
        res[test_model] = (cnt, num, sr, cer, wer)
    return res


def run_test(attacker):

    data_dir_path = "./datasets"
    test_models = ['deepspeech', 'wav2vec2', 'whisper', 'aliyun', 'iflytek']
    res = {}
    for test_model in test_models:
        res[test_model] = (0, 0, 0)
    data_path = os.path.join(data_dir_path, f"vctk_2000_wav_spec_sid_txt.pkl")
    with open(data_path, "rb") as f:
        dataset = pickle.load(f)
        res_now = test_dataset(dataset, test_models, attacker)
        print(f'{attacker.ptb_path}\n')
        for key, value in res_now.items():
            print(f'Model: {key}:\nSR = {value[0]}/{value[1]} = {value[2]}\nCER = {value[3]}\nWER = {value[4]}\n')

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Attack model arguments")

    parser.add_argument('--ptb_path', type=str, default="./LS_TUAP.pth", help='Path to the pre-trained model')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for computation')
    parser.add_argument('--output_dir', type=str, default='./result', help='Directory for saving output')
    parser.add_argument('--sampling_rate', type=int, default=16000, help='Sampling rate to use')

    args = parser.parse_args()

    attacker = Attacker(args)

    run_test(attacker)