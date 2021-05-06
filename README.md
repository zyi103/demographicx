```bash
!pip install git+https://github.com/sciosci/demographicx
```

```python
from demographicx.classifier import GenderEstimator, EthnicityEstimator
```


```python
gender_classifier = GenderClassifier()
```


```python
gender_classifier.predict('Daniel')
```




    {'male': 0.9886190672823015,
     'unknown': 0.011367974526753396,
     'female': 1.2958190945360288e-05}

```python
race_classifier = classifier.EthnicityEstimator()
```


```python
race_classifier.predict('lizhen liang')
```




    {'Black': 2.1461191541442314e-06,
     'Hispanic': 4.0070474029127346e-05,
     'White': 0.0002176521167431309,
     'Asian': 0.999740131290074}




```python
race_classifier.predict('daniel wegmann')
```




    {'Black': 4.120965729769303e-06,
     'Hispanic': 0.0023926903023342287,
     'White': 0.9963380370701861,
     'Asian': 0.00126515166175015}


