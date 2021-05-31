"""
Understanding what data indicates and how can it be leveraged
Parameteres made:
    - Correlation
    - Hourly data (mean of all 5 minutes interval)
    - Data with due point, temp, rel hum seprated (might have error in WEST) [advancedDF]
    - All data with necessory elements seprated [reducedDF]
"""
__author__ = 'BlackDChase'
__version__ = '1.5.3'


import pandas as pd

df = pd.read_csv(
    '../Dataset/finalwork.txt',
    sep=',',
    parse_dates={'dt':['Date']},
    infer_datetime_format=True,
    index_col='dt',
    na_values=['nan','?'],
)

# adding hours and minutes
df.index = df.index + pd.Series([pd.core.indexes.datetimes.timedelta(hours=x) for x in df['Hour']])
df.index = df.index + pd.Series([pd.core.indexes.datetimes.timedelta(minutes=x) for x in df['Minute']])
df=df.drop(columns=['Hour','Minute','Count'])
# Don't know whats count

# Ignoring Temp, Dew Point Temp and Rel Hum
reducedDF = pd.DataFrame(df,columns=[
    'Market Demand',
    'Ontario Demand',
    'Ontario Price',
    'Northwest',
    'Northeast',
    'Ottawa',
    'East',
    'Toronto',
    'Essa',
    'Bruce',
    'Southwest',
    'Niagara',
    'West',
])

# Only considering data with Temp, Dew Point Temp and Rel Hum
advancedDF = df.dropna(axis=0)

# Correlation Calculation
reCorr1 = reducedDF.corr()
adCorr1 = advancedDF.corr()

# Saving data
advancedDF.to_csv('../Dataset/Advanced Dataset.csv')
reducedDF.to_csv('../Dataset/Reduced Dataset.csv')

# Taking hourly samples
reducedDF = reducedDF.resample('H').mean()
advancedDF = advancedDF.resample('H').mean()

# Correlation Calculation
reCorr2 = reducedDF.corr()
adCorr2 = advancedDF.corr()

# Saving data
advancedDF.to_csv('../Dataset/Hourly Advanced Dataset.csv')
reducedDF.to_csv('../Dataset/Hourly Reduced Dataset.csv')
#reCorr2.to_csv('../Dataset/Hourly reCorr.csv')
#adCorr2.to_csv('../Dataset/Hourly adCorr.csv')

adCorr = [[0]*43]*43
reCorr = [[0]*13]*13


a=list(adCorr1)
a=a[1:]
r=list(reCorr1)
r=r[1:]

for i in range(len(a)):
    for j in range(len(adCorr1[a[i]])):
        z=adCorr1[a[i]][j]-adCorr2[a[i]][j]
        if z<1 or z>1 or z==1:
            adCorr[i][j]=z
for i in range(len(r)):
    for j in range(len(reCorr1[r[i]])):
        reCorr[i][j]=reCorr1[r[i]][j]-reCorr2[r[i]][j]

rNc = sum(map(lambda x:sum(map(lambda i:i**2,x)),reCorr))
aNc = sum(map(lambda x:sum(map(lambda i:i**2,x)),adCorr))
# Both being very small
# Proves that correlation wont change much after size reduction
# Hence reCorr2,adCorr2 correlation is enough


for i in range(len(a)):
    for j in range(len(adCorr1[a[i]])):
        z=adCorr1[a[i]][j]
        if z>0 or z==0:
            adCorr1[a[i]][j]=z
        elif z<0:
            adCorr1[a[i]][j]=-z
        else:
            adCorr1[a[i]][j]=0
for i in range(len(r)):
    for j in range(len(reCorr1[r[i]])):
        z=reCorr1[r[i]][j]
        if z>0 or z==0:
            reCorr1[r[i]][j]=z
        elif z<0:
            reCorr1[r[i]][j]=-z
        else:
            reCorr1[r[i]][j]=0

# Saving data
reCorr1.to_csv('../Dataset/reCorr.csv')
adCorr1.to_csv('../Dataset/adCorr.csv')


# Making Dataset
full = pd.read_csv('finalwork.csv')
full = full.drop(axis=0,columns=['Count','Date','Hour','Minute','Market Demand'])
new=full.ffill().bfill()
deduct=np.random.rand(len(full))                                           
new['supply'] = new['Ontario Demand'] - deduct*new['Ontario Price']
new=new[['Ontario Price', 'Ontario Demand','supply' , 'Northwest', 
       'Northwest Temp', 'Northwest Dew Point Temp', 'Northwest Rel Hum', 
       'Northeast', 'Northeast Temp', 'Northeast Dew Point Temp', 
       'Northeast Rel Hum', 'Ottawa', 'Ottawa Temp', 'Ottawa Dew Point Temp', 
       'OttawaRel Hum', 'East', 'East Temp', 'East Dew Point Temp', 
       'East Rel Hum', 'Toronto', 'Toronto Temp', 'Toronto Dew Point Temp', 
       'Toronto Rel Hum', 'Essa', 'Essa Temp', 'Essa Dew Point Temp', 
       'Essa Rel Hum', 'Bruce', 'Bruce Temp', 'Bruce Dew Point Temp', 
       'Bruce Rel Hum', 'Southwest', 'Southwest Temp', 
       'Southwest Dew Point Temp', 'Southwest Rel Hum', 'Niagara', 
       'Niagara Temp', 'Niagara Dew Point Temp', 'Niagara Rel Hum', 'West', 
       'West Temp', 'West Dew Point Temp', 'West Rel Hum']]
profit=(new['Ontario Demand']-new['supply'])*new['Ontario Price']
profit.mean()     
#908.532587007038
profit.std()      
#26168.12415430964
profit.min()      
#0.0
profit.max()      
#3946874.814376206
profit.median()   
#153.44608280964934

maxi=[]
mini=[]
for i in new.columns: 
    m1=max(new[i]) 
    m2=min(new[i]) 
    new[i]=(new[i]-m2)/(m1-m2) 
    maxi.append(m1) 
    mini.append(m2)  
new.to_csv('normalized_weird_43_columns_with_supply.csv',index=False)  
x=pd.DataFrame(columns=['max','min'])                                                                      
x['max'],x['min']=maxi,mini
x.to_csv('min_max_values_43_columns_with_supply.csv',index=False)



# Supply demand data Normalized
norm = pd.read_csv('normalized_weird_13_columns_with_supply.csv')
maxmin = pd.read_csv('min_max_values_13_columns_with_supply.csv')
denorm = pd.DataFrame(columns=norm.columns)


for i in range(len(norm.columns)):
    col=norm.columns[i]
    denorm[col]= norm[col]*(maxmin['max'][i]-maxmin['min'][i]) + maxmin['min'][i]
denorm['Profit']=(denorm['Ontario Demand']-denorm['supply'])*denorm['Ontario Price']
profit=denorm['Profit']
profit.mean()
#106272.987
profit.std()
#74348.8939
profit.min()
#0.19366888
profit.max()
#5860463.26
profit.median()
#104574.458
