# phonerec
wav2vec2-based multi-lingual phone recognition with time alignment

## how to use

```pip install -r requirements.txt```

```python phonerec.py [wavfile ...]```

requirments: wavfile must be 16 khz, 16-bits, mono

For each .wav file, one .json file and one .lab file (wavesurfer format) will be generated in the same directory, containing the recognized phones and the start and end times.
