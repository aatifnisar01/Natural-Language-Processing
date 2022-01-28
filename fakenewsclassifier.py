from ctgan import CTGANSynthesizer
from ctgan import load_demo

data = load_demo()

# Names of the columns that are discrete
discrete_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income'
]

ctgan = CTGANSynthesizer(epochs=10)
ctgan.fit(data, discrete_columns)

# Synthetic copy
samples = ctgan.sample(1000)


    

