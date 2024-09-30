# phonerec
wav2vec2-based multi-lingual phone recognition with time alignment

## how to use

```pip install -r requirements.txt```

```python phonerec.py [wavfile ...]```

For each .wav file, one .json file and one .lab file (wavesurfer format) will be generated in the same directory, containing the recognized phones and the start and end times.

## python usage

```
from phonerec import PhoneRecognizer

phonerec = PhoneRecognizer()

# from filename
res = phonerec.recognize(file='test.wav')

# from numpy vector
import scipy.io.wavfile as wav
fs,x = wav.read('test.wav')
res = phonerec.recognize(x,fs)   # <-- NOTE: argument order follows librosa convention


# output will be start,end,phone in a list of lists, e.g: 
# [[0.0, 0.34, 'sil'], [0.34, 0.36, 'h'], [0.36, 0.48, 'iː'], [0.48, 0.5, 'w'], [0.5, 0.58, 'ʌ'], [0.58, 0.6, 'z'] ... ]

```

