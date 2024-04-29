### audio embedding ###

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import librosa
import matplotlib.pyplot as plt
import transformers
from transformers import Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2FeatureExtractor




def get_audio_embedding(audio_file:str, raw_feature_type:str=None, huggingface_model_name:str = None):
    
    # Load audio
    SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(audio_file)
    if SAMPLE_RATE != 16000:
        print("Resampling")
        SPEECH_WAVEFORM = torchaudio.transforms.Resample(SAMPLE_RATE, 16000)(SPEECH_WAVEFORM)
        SAMPLE_RATE = 16000
    
    # LFCC
    if raw_feature_type=="LFCC":
        
        n_fft = 2048
        win_length = None
        hop_length = 512
        n_lfcc = 256

        lfcc_transform = T.LFCC(
            sample_rate=SAMPLE_RATE,
            n_lfcc=n_lfcc,
            speckwargs={
                "n_fft": n_fft,
                "win_length": win_length,
                "hop_length": hop_length,
            },
        )

        feature = lfcc_transform(SPEECH_WAVEFORM)
        return feature
    # MFCC
    elif raw_feature_type=="MFCC":
        
        n_fft = 2048
        win_length = None
        hop_length = 512
        n_mels = 256
        n_mfcc = 256

        mfcc_transform = T.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "n_mels": n_mels,
                "hop_length": hop_length,
                "mel_scale": "htk",
            },
        )

        feature = mfcc_transform(SPEECH_WAVEFORM)
        return feature

    
    # HUGGINGFACE MODEL AUDIO EMBEDDING
    if huggingface_model_name is not None:
        model = Wav2Vec2Model.from_pretrained(huggingface_model_name)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(huggingface_model_name)
        input= feature_extractor(
            SPEECH_WAVEFORM.squeeze().numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )
        with torch.no_grad():
            o= model(input.input_values)
        return o.last_hidden_state
    else:
        raise ValueError("Please provide a valid huggingface model name")
    
        


if __name__ == "__main__":
    audio_path = "/home/georg/projects/university/C5/multimodal_dataset/First_Impressions_v3_multimodal/test/1/3ZwXifihvtA.000.wav"
    hugging_face_model_name = "versae/wav2vec2-base-finetuned-coscan-age_group"

    mfcc = get_audio_embedding(audio_path, raw_feature_type="MFCC")
    print(mfcc.shape)
    lfcc = get_audio_embedding(audio_path, raw_feature_type="LFCC")
    print(lfcc.shape)
    huggingface_model_name = "facebook/wav2vec2-base-960h"
    wav2vec2 = get_audio_embedding(audio_path, huggingface_model_name=huggingface_model_name)
    print(wav2vec2.shape)
    
    # plot mfcc
    plt.figure(figsize=(10, 4))
    plt.imshow(mfcc[0].detach().numpy(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("MFCC")
    plt.savefig('mfcc.png')
    
    # plot lfcc
    plt.figure(figsize=(10, 4))
    plt.imshow(lfcc[0].detach().numpy(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("LFCC")
    plt.savefig('lfcc.png')

    # plot wav2vec2
    plt.figure(figsize=(10, 4))
    plt.imshow(wav2vec2[0].detach().numpy(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Wav2Vec2")
    plt.savefig('wav2vec2.png')    
    print("Done")