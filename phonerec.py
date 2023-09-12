
import torch
import scipy.io.wavfile as wav
import json
import webrtcvad
import sys
from transformers import AutoProcessor, AutoModelForCTC
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import os

def init():
    if  torch.cuda.is_available():
        device = 'cuda'
    elif  torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    device='cpu'
    print('device=',device)
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft").to(device)
    return device,processor,model


# vadlen = frame length in sec must be 0.01, 0.02 or 0.03
def getvad(fs,y,vadlen):

    assert(fs==16000)
    vad = webrtcvad.Vad(3)
    i = 0
    vaddata = []
    #tvad = []
    # add 10 dummy frames because the vad can be unstable initially (these are removed from the output)
    y2 = np.hstack((np.zeros(int(fs*vadlen*10),dtype='int16'),y)) 
    while True:
        vadframe = y2[i:i+int(fs*vadlen)].tobytes()
        if len(vadframe)<2*fs*vadlen:
            break
        
        vaddata.append(vad.is_speech(vadframe,fs))
        #tvad.append(i/fs)
        i += int(fs*vadlen)
    xvad = np.array(vaddata).astype(np.int16)
    xvad = xvad[10:] # remove the ten dummy frames
    
    #plt.plot(np.arange(xvad.size)*vadlen,xvad,'.')
    #plt.plot(np.arange(y.size)/fs,y/(2**15))
    #plt.show()

    return xvad
   
def recognize(fs,y,device,processor,model,xvad=None):
    
    input_values = processor(y.astype('float32'), sampling_rate=fs, return_tensors="pt").input_values.to(device)
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    output = processor.batch_decode(predicted_ids[0],output_char_offsets=True)
    recframes = output['char_offsets']
    nframes = len(recframes)
    #import pdb;pdb.set_trace()
    rate=50
    currentphone = ''
    phoneseq = []
    if xvad is not None:
        vadlist = xvad.tolist()
    else:
        vadlist = [1]*nframes
    
    for rec,vad,count in zip(recframes,vadlist,range(nframes)):
        token=''
        for xx in rec:
            token += xx['char']
        #print('token=',token)
        if token != '<pad>':
            currentphone = token
        if vad:
            phoneseq.append(currentphone)
        else:
            phoneseq.append('sil')

    nphoneseq = []
    currentphone = ''
    for phone in reversed(phoneseq):
        if phone != '':
            currentphone = phone
        else:
            phone = currentphone
        nphoneseq.append(phone)
    nphoneseq.reverse()
    phoneseq = nphoneseq
         
    lastphone = ''
    labels = []
    label = []
    for i,phone in enumerate(phoneseq):
        if phone != lastphone:
            if label:
                label.append(i/rate) #end time
                label.append(lastphone) # phone
                labels.append(label)
            label = [i/rate]
        lastphone = phone
    return labels

if __name__ == '__main__':
    device,processor,model = init()
    
    for wavfile in sys.argv[1:]:
        print(wavfile)
        jsonfile = wavfile.replace('.wav','.json')
        labfile = wavfile.replace('.wav','.lab')
        print(wavfile,jsonfile,labfile)
        
        fs,y = wav.read(wavfile)
        y = y.astype(np.int16)
        assert(fs==16000)
        assert(len(y.shape)==1)

        vadlen = 0.02
 
        xvad = getvad(fs,y,vadlen)

        phones = recognize(fs,y,device,processor,model,xvad)
        
        print(y.size/fs,'s')

        json.dump({'tiers':{'phones':{'entries':phones}}},open(jsonfile,'w'))
        with open(labfile,'w') as f:
            for p in phones:
                t0,t1,sym = p
                f.write('{}\t{}\t{}\n'.format(t0,t1,sym))
