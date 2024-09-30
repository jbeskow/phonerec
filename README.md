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
phonerec.recognize(x,fs)   # <-- NOTE: argument order follows librosa convention
```

