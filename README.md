# OCR


## Requirement
- ```pip install jamo```

## Update

- 05.25.21 : add ```ASTER``` for encode. ([https://github.com/ayumiymk/aster.pytorch])

## Train
``` 
python run.py -pt [bool] \
                 -m [mode] \
                 -n [name] \
                 -d [bool] \
                 -cm [model] \
```
- ```-pt``` (default-False): training using phoneme(True) or character(False)
- ```-m```(default-Test) : training mode(Train) or test mode(Test)
- ```-d```(default-False) : delete your savinf folder if True
- ```-n``` (required): result folder
- ```-cm``` (required): recognition model(default : CRNN, list:(CRNN, ASTER))

```python run.py -pt True -m Train -n test_1 -d True -cm CRNN```