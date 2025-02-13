import sys
import io
import os
from NISQA.NISQA_model import nisqaModel
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def NISQA_score(audio_file, pretrained_model='nisqa_tts.tar', ms_channel=None, tr_bs_val=1, tr_num_workers=0, output_dir=None):
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), pretrained_model)
    
    # divert stdout to null stream 
    null_stream = io.StringIO() 
    sys.stdout = null_stream
    
    # run code
    nisqa = nisqaModel({
        'mode': 'predict_file',
        'pretrained_model': model_path,
        'deg': audio_file,
        'ms_channel': ms_channel,
        'tr_bs_val': tr_bs_val,
        'tr_num_workers': tr_num_workers,
        'output_dir': output_dir
    })
    nisqa_res = nisqa.predict()
    
    # divert stream back
    sys.stdout = sys.__stdout__
    
    return nisqa_res.iloc[0]['mos_pred']