import pandas as pd
from dataset import dataset

# Create DataFrame - first row is headers
df = pd.DataFrame(dataset[1:], columns=dataset[0])
print(df)

