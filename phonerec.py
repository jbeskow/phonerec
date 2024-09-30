
import torch
import librosa
import json
import webrtcvad
import sys
from transformers import AutoProcessor, AutoModelForCTC
import numpy as np

class PhoneRecognizer:
    def __init__(self):
        if  torch.cuda.is_available():
            self.device = 'cuda'
        elif  torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        self.device='cpu'
        self.processor = AutoProcessor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
        self.model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft").to(self.device)

    # vadlen = frame length in sec must be 0.01, 0.02 or 0.03
    def getvad(self,y,fs,vadlen):

        assert(fs==16000)
        assert(len(y.shape)==1)
        assert(y.dtype==np.int16)

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
  
        return xvad
    
    def recognize(self,y=None,fs=16000,file=None):

        if file:
            y, fs = librosa.load(file, sr=16000) # Downsample and convert as needed
            yint = (y*32767).astype(np.int16)
        if y is None:
            error('No audio data')
        elif y.dtype == np.float32:
            yint = (y*32767).astype(np.int16)
        elif y.dtype == np.int16:
            yint = y
            y = yint.astype('float32')/32767
        else:
            error('Audio data must be float32 or int16')

        input_values = self.processor(y.astype('float32'), sampling_rate=fs, return_tensors="pt").input_values.to(self.device)
        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        output = self.processor.batch_decode(predicted_ids[0],output_char_offsets=True)
        recframes = output['char_offsets']
        nframes = len(recframes)
        rate=50
        currentphone = ''
        phoneseq = []

        vadlen = 0.02
        vadlist = self.getvad(yint,fs,vadlen).tolist()

        #         
        for rec,vad,count in zip(recframes,vadlist,range(nframes)):
            token=''
            for xx in rec:
                token += xx['char']
            if token != '<pad>':
                currentphone = token
            if vad:
                phoneseq.append(currentphone)
            else:
                phoneseq.append('sil')
        
        # reverse pass to fill empty phones at the beginning
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
        phoneseq.append('<endtoken>') 

        # decode frame-based phoneseq into times + labels
        prevphone = ''
        labels = []
        label = []
        for i,phone in enumerate(phoneseq):
            if phone != prevphone:
                if label:
                    label.append(i/rate) #end time
                    label.append(prevphone) # phone
                    labels.append(label)
                label = [i/rate]
            prevphone = phone
        return labels

if __name__ == '__main__':

    phonerec = PhoneRecognizer()
    
    for wavfile in sys.argv[1:]:
        jsonfile = wavfile.replace('.wav','.json')
        labfile = wavfile.replace('.wav','.lab')
        print(wavfile,'->',jsonfile,labfile)
        
        phones = phonerec.recognize(file=wavfile)

        json.dump({'tiers':{'phones':{'entries':phones}}},open(jsonfile,'w'))
        with open(labfile,'w') as f:
            for p in phones:
                t0,t1,sym = p
                f.write('{}\t{}\t{}\n'.format(t0,t1,sym))
