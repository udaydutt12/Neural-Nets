import pandas as pd

# create empty df
df = pd.DataFrame()
df['name'] = ['sarah','jay','ray']
df['employed'] = ['yes','yes','no']
df['age'] = [1,24,94]

def mean_age_by_group(dataframe,col):
    return dataframe.groupby(col).mean()

def uppercase_column_name(dataframe):
    dataframe.columns = dataframe.columns.str.upper()
    return dataframe

#create a pipeline that applies both functions
(df.pipe(mean_age_by_group,col='employed')
    .pipe(uppercase_column_name)
)



