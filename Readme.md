# Master Thesis - data analysis over personal data

##Install dependencies

#### Install virtual enviroment
Download and install Miniconda from here https://conda.io/miniconda.html
>conda create -n py36 numpy pandas python=3.6 (or python 2.7)

#### install kernel for jupyter in the current env
>pip install ipykernel

#### install ipython kernel
>ipython kernel install --user --name=projectname

#### At this point, you can start jupyter:
>jupyter notebook

#### create a new notebook and select the kernel that lives inside your environment.

## install further needed lib

### inside your env:

#### install bs4:
>pip install bs4

#### install sklearn:
>pip install sklearn

#### install scipy:
>pip install scipy

#### install matplotlib:
>pip install matplotlib

#### install nltk:
>pip install nltk

#### install mediawiki:
>pip install pymediawiki

#### install geopy:
>pip install geopy

#### install html5 parser:
>pip install HTMLParser
>pip install html5lib

#### create a jupyter notebook in the directory you putted the downloaded lib file and import
## import class for analysis
>from lib.AnalysisClass import *

#### importclass for generating csv from facebook/google 
>from lib.CreateCsvClass import *

## classes structure

### CreateCsvClass

##### converting google html files
>g=gCsvConverter('readingPath','writing path')
>g.execute()

##### converting facebook html files
>f=fCsvConverter('readingPath','writing path')
>f.execute()

readingPath= the directory path of your google data directory
writingPath=directory where you want to save your csv

###AnalysisClass

####get google analysis
>gg=PersonalDataTopicAnalysis('readPath')
>gg.execute()
>gg.info_table('month','year')

readPath=csvPath of file you want to analyse
month-year:for filtering data according to the given month-year










