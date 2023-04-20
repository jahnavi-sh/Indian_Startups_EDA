#INDIAN STARTUPS - EDA

#Load libraries 
#linear algebra 
import numpy as np 
#data preprocessing and exploration 
import pandas as pd 
#data visualisation 
import matplotlib.pyplot as plt 
import seaborn as sns 
import missingno as msno

#It contains the following columns - 
#1. Company - Name of the startup
#2. City - City in which the company was started 
#3. Starting year - Year in which the company was started 
#4. Founders - Name of the founders of the company
#5. Industries - Industry domain which the company belongs to 
#6. Number of employees - Number of employees in the company 
#7. Funding amount in USD - Total funding amount to the startup (in USD)
#8. Funding rounds - Funding ronds are the number of times a startup goes back to the market to raise capital. The goal 
#   of every round in for founders to trade company's equity for capital that they can use to advance their company to the 
#   next level 
#9. Number of investors - Number of investors in the company 

#load data
df = pd.read_csv(r'Indian_startup.csv')

#Understanding the data 

#view rows and columns of dataset 
df.shape
#There are 300 rows (300 data points) and 11 columns (11 features)

#view column name  
df.columns 
#as listed above 

#view statistical measures of the dataset 
df.info()

#Checking for missing values 
df.isnull().sum()
#There are no missing values in the dataset 

#check null values in visualisation 
msno.bar(df, color='blue', fontsize=25)

#checking for duplicate values 
df.duplicated().sum()
#There are no duplicate values in the dataset 

#checking unique values 
for i in df.columns:
    print(i,'--------', df[i].unique(),'--------', df[i].nunique())

#statistical description of the dataset 
df.describe()

#Visualising the correlation 
sns.heatmap(df.corr(), annot=True)
#There is no correlation between the columns 

#data preprocessing and cleaning 

df.sample()

#removing column - unnamed and description 
df.drop(columns=['Unnamed: 0','Description'], inplace=True)

df["Industries_type 1"]=df["Industries"].str.split(",",expand=True)[0]
df["Industries_type 2"]=df["Industries"].str.split(",",expand=True)[1]
df[df["Industries_type 2"].isnull()]

# Filling industries 
l=[29,52,105,134,137,154,189,201,273,277]
for i in l:
    df["Industries_type 2"].iloc[i]="None"

df["Industries_type 2"].isnull().sum()

#creating a column of age through feature engineering to check how old the company is 
df["age"]=df["Starting Year"].max()-df["Starting Year"]
#visualising
sns.distplot(df['age'])

#divide the age of company in 3 segments 
#creating a column of age category through feature engineering to check which category the company belongs to
df["age_category"]=pd.cut(df.age,[-1,5,15,np.inf],labels=["new company","old company","older company"])

#exploratory data analysis  

#in which year maximum startups are launched 
plt.figure(figsize=(16,10))
ax=sns.countplot(x="Starting Year",data=df,palette='twilight_shifted',edgecolor="black")
plt.xticks(rotation=45)
plt.title('which year maximum startups are launched')

for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center',fontsize=16,
                xytext = (0, 10),
                textcoords = 'offset points')

#In 2015, maximum startups were created - 63

# In which city maximum startups are launched
from wordcloud import WordCloud as word
wc = word(background_color='white', width=1000, height=400,colormap="twilight_shifted")
wc.generate_from_frequencies(df["City"].value_counts())
plt.figure(figsize=(10,13),dpi=100)
plt.imshow(wc)
plt.axis('off')
plt.show()

#Therefore, in Bengaluru maximum startups are found, followed by Mumbai, Gurgaon and New Delhi 

# Which founders were involved in more than 1 startups 
x=df["Founders"].value_counts().sort_values(ascending=False).head(4)
print(x)
color=sns.color_palette("twilight_shifted")
plt.figure(figsize=(7,7))
plt.pie(x,labels=x.index)
plt.legend(loc="lower center",bbox_to_anchor=(0.5,-0.15),ncol=4)
plt.title('Founders with 2 or more startups')

#The founder with maximum startups - 9 is not available. 
#Next, the founder with maximum startups is Vijay Shekhar Sharma with 9 startups, Amit Jain and Anurag Jain with 2 startups and Mukesh Ambani 
#with 2 startups as well

#Number of employees in the startup 
plt.figure(figsize=(18,10))
ax=sns.countplot(x="No. of Employees",data=df)
plt.xticks(rotation=45)
plt.title('No of Employees')

for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center',fontsize=20,
                xytext = (0, 10),
                textcoords = 'offset points')
    
#Mostly startups have 101-250 employees 

#Number of funding rounds for the startup  
plt.figure(figsize=(18,10))
ax=sns.countplot(x="Funding Round",data=df)
plt.xticks(rotation=45)
plt.title('Funding Rounds')

for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center',fontsize=20,
                xytext = (0, 10),
                textcoords = 'offset points')
    
#47 startups have 3 funding rounds and 42 startups have 5 funding rounds 

#Number of investors 
plt.figure(figsize=(18,10))
ax=sns.countplot(x="No. of Investors",data=df)
plt.xticks(rotation=45)
plt.title('No. of Investors')

for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center',fontsize=20,
                xytext = (0, 10),
                textcoords = 'offset points')
    
#Mostly startups have 4 and 6 investors 

#top 10 companies in Industry type 1 and Industry type 2
b=df["Industries_type 1"].value_counts().head(10)
plt.figure(figsize=(18,20),dpi=100)
plt.subplot(2,2,1)
ax=sns.countplot(x="Industries_type 1",data=df,order=b.index)
plt.xticks(rotation=45)
plt.title('Top 10 Industries type 1')
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center',fontsize=20,
                xytext = (0, 10),
                textcoords = 'offset points')

plt.subplot(2,2,2)
c=df["Industries_type 2"].value_counts().head(10)
ay=sns.countplot(x="Industries_type 2",data=df,order=c.index)
plt.xticks(rotation=45)
plt.title('Top 10 Industries type 2')

for y in ay.patches:
    ay.annotate(format(y.get_height(), '.0f'),
                (y.get_x() + y.get_width() / 2., y.get_height()),
                ha = 'center', va = 'center',fontsize=20,
                xytext = (0, 10),
                textcoords = 'offset points')
    
#The top industry type is E-commerce and Financial services 

#how old the company is in the age category they belong to 
plt.figure(figsize=(25,22),dpi=100)
plt.subplot(2,2,1)
ab= sns.countplot(x="age",data=df)
plt.xticks(rotation=45)
plt.title('Age')
for p in ab.patches:
    ab.annotate(format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center',fontsize=15,
                xytext = (0, 10),
                textcoords = 'offset points')


plt.subplot(2,2,2)
ac=sns.countplot(x="age_category")
plt.xticks(rotation=45)
plt.title('Age category')

for j in ac.patches:
    ac.annotate(format(j.get_height(), '.0f'),
                (j.get_x() + j.get_width() / 2., j.get_height()),
                ha = 'center', va = 'center',fontsize=20,
                xytext = (0, 10),
                textcoords = 'offset points')
    
#Most of the companies are 5 years old and they are new companies 

#top companies who got the maximum funding amount 
df[df["Funding Amount in $"]==df["Funding Amount in $"].max()]

#Reliance Jio has the maximum funding 

#top 10 companies which got the maximum funding and city it belongs to 
df.groupby(["Starting Year","Company","City"])["Funding Amount in $"].max().sort_values(ascending=False).to_frame().head(10).style.background_gradient(cmap='twilight_shifted')

#Which city startups has got the Highest funding
df.groupby("City")["Funding Amount in $"].sum().sort_values(ascending=False).to_frame().style.background_gradient(cmap='twilight_shifted')

#Mumbai, Bengaluru and Gurgaon have startups with highest funding amount 

#visualising 
plt.figure(figsize=(15,15))
sns.barplot(x="Funding Amount in $",y="City",edgecolor="black",estimator=sum,data=df,palette='twilight_shifted');
plt.title('City With Highest funding', fontdict={'fontsize': 20, 'color': 'black', 'fontweight': 'bold'}, bbox=dict( facecolor='grey', alpha=0.5,edgecolor='black'));

#startups with highest number of employees 
df[df["No. of Employees"]=="10001+"][["Company","Funding Amount in $","City","Industries_type 1","No. of Investors"]].sort_values(by="Funding Amount in $",ascending=False).style.background_gradient(cmap='twilight_shifted')

#the oldest company 
df[df["age"]==df["age"].max()]

#Five Star Business Finance is the oldest company - 36 years old 

#Companies older than 15 years 
df[df["age"]>15][["Company","age","age_category"]].sort_values(by="age",ascending=False).style.background_gradient(cmap='twilight_shifted')
#visualising 
x=df[df["age"]>15][["Company","age"]].sort_values(by="age",ascending=False)
plt.figure(figsize=(15,13))
sns.barplot(x="age",y="Company",data=x,edgecolor="black",palette='twilight_shifted')
plt.title('Oldest Companies', fontdict={'fontsize': 28, 'color': 'black', 'fontweight': 'bold'}, bbox=dict( facecolor='grey', alpha=0.5,edgecolor='black'));

#Minimum, maximum and average funding as well as the number of investors
df[["Funding Amount in $","No. of Investors","age"]].agg(["mean","min","max"]).style.background_gradient(cmap='twilight_shifted')

#company with zero funding amount 
df[df["Funding Amount in $"]== 0]

#We see that WOW skin science has zero funding 

#Companies with 15+ funding rounds 
df[df["Funding Round"]>15][["Company","Starting Year","Funding Round","Funding Amount in $","Industries_type 1"]].sort_values(by="Funding Round",ascending=False).style.background_gradient(cmap="twilight_shifted")

#Ola has the maximum funding rounds - 25

#Number of investors 
df[df["No. of Investors"]>25][["Company","Starting Year","No. of Investors","Industries_type 1"]].sort_values(by="No. of Investors",ascending=False).style.background_gradient(cmap="twilight_shifted")

#Ola has the highest number of investors - 45, followed by Byju's - 38 