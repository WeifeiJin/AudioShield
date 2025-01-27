import time
import threading
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
import nls
import json


ACCESSKEY_ID = "YOUR_ACCESSKEY_ID"
ACCESSKEY_SECRET = "YOUR_ACCESSKEY_SECRET"
URL = "wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1"
APPKEY = "ZdOsz3IummdikLCQ"
RESULT = []


def get_token():
    client = AcsClient(
        ACCESSKEY_ID,
        ACCESSKEY_SECRET,
        "cn-shanghai"
    )

    request = CommonRequest()
    request.set_method('POST')
    request.set_domain('nls-meta.cn-shanghai.aliyuncs.com')
    request.set_version('2019-02-28')
    request.set_action_name('CreateToken')

    try:
        response = client.do_action_with_exception(request)
        # print(response)

        jss = json.loads(response)
        if 'Token' in jss and 'Id' in jss['Token']:
            token = jss['Token']['Id']
            expireTime = jss['Token']['ExpireTime']
            # print("token = " + token)
            # print("expireTime = " + str(expireTime))
            return str(token)
    except Exception as e:
        print(e)
        exit()





def wav2pcm(input_dir):
    with open(input_dir, 'rb') as wavfile:
        ori_data = wavfile.read()
        wavfile.close()
    out_dir = input_dir[:-3]+"pcm"
    with open(out_dir, 'wb') as pcmfile:
        pcmfile.write(ori_data)
        pcmfile.close()
        return out_dir


def on_start(message, *args):
    # print("[EVENT] on_start: {}".format(message))
    return

def on_error(message, *args):
    # print("[EVENT] error args=> {}".format(args))
    return

def on_close(*args):
    # print("[EVENT] on_close: args=> {}".format(args))
    return

def on_result_chg(message, *args):
    # print("[EVENT] test_on_chg: {}".format(message))
    return

def on_completed(message, *args):
    # print("[EVENT] on_completed:args=> {} message=> {}".format(args, message))
    RESULT.append(json.loads(message)["payload"]["result"])


def aliyun_ASR(filepath):
    """
    :param filepath: only support wav/pcm format audio input
    :return: text
    """
    if filepath[-3:] == "wav":
        filepath = wav2pcm(filepath)

    if filepath[-3:] != "pcm":
        raise Exception("Invalid file type!")
    # print(filepath)

    response = []
    sr = nls.NlsSpeechRecognizer(
        url=URL,
        token=get_token(),
        appkey=APPKEY,
        on_start=on_start,
        on_result_changed=on_result_chg,
        on_completed=on_completed,
        on_error=on_error,
        on_close=on_close,
        callback_args=response
    )

    with open(filepath, "rb") as f:
        data = f.read()
    sr.start(aformat="pcm")

    slices = zip(*(iter(data),) * 640)
    for i in slices:
        sr.send_audio(bytes(i))
        time.sleep(0.01)

    sr.stop()
    while True:
        if RESULT:
            return RESULT.pop()
        else:
            print("Waiting ... ")


if __name__ == "__main__":
    filepath = "../../benign_audio/p225_001.wav"
    import soundfile as sf
    wav, sr = sf.read(filepath)
    print(sr)
    # sf.write(filepath[:-4]+'(rewrite).pcm',wav,16000)
    text = aliyun_asr(filepath)
    print(f"Result: {text}")