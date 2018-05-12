
# coding: utf-8

# In[1]:


#All libraries and dependences

import pandas as pd
import numpy as np
import os
from urllib.request import urlopen
from urllib.request import Request
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import matplotlib.pyplot as pt
from matplotlib import *
import re
import datetime
import collections
from time import time
import pyLDAvis
import pyLDAvis.sklearn 
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.collocations import *
import string
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from time import time
#from sklearn.neighbors import KNeighborsClassifier
from mediawiki import MediaWiki
from mediawiki import DisambiguationError,PageError
from collections import OrderedDict
from geopy.geocoders import Nominatim


# In[2]:


#
#Class used for splittid the time: the initial given format is year-month-day-time-> it generates a column for each of them
#
class get_splitted_time:    
        def __init__(self):
            
            self.times=[]
            self.years=[]
            self.days=[]
            self.months=[]

            self.time_splitted=pd.DataFrame(
                {
                    'day':[],
                    'month':[],
                    'year': [],
                    'time':[]
                })

            self.calendar={
                'Jan':1,
                'Feb':2,
                'Mar':3,
                'Apr':4,
                'May':5,
                'Jun':6,
                'Jul':7,
                'Aug':8,
                'Sep':9,
                'Oct':10,
                'Nov':11,
                'Dec':12   
            }
        
        def split_time(self,time):
            temp= time.strip().split(',')
            if(len(temp)>1):
                time=temp[1]
                temp1=temp[0].strip().split(' ')
                if(len(temp1)>2):
                    day=temp1[0]
                    month=temp1[1]
                    year=temp1[2]

                    self.times.append(time)
                    self.years.append(year)
                    self.days.append(day)
                    self.months.append(month)

        def get_day(self,year,month,day):
            datetime.datetime.today()
            dd=datetime.date(year,month,day)
            return(dd.strftime("%A"))
    
        def function(self,data):
            dic={}
            for x in range(0,24):
                dic[x]=0

                for i in data:
                    tmp=i.split(':')
                    key=int(tmp[0])
                    dic[key]+=int(tmp[1])

                    dic={k: f(v) for k,v in dic.items()}
                    return collections.OrderedDict(sorted(dic.items()))  


        def execute(self,frame):
            frame['when'].apply(self.split_time)
        
            tempFrame=pd.DataFrame({
                'day':self.days,
                'month':self.months,
                'year':self.years,
                'time':self.times
            })      
            tempFrame['day_name']=tempFrame.apply(lambda row:self.get_day(int(row['year']),self.calendar[row['month']],int(row['day'])),axis=1)
            return tempFrame


# In[5]:


#Class that performs the analysis on text data generated from google or facebook, it returns a table with the
#topics-time-location of the data
#
#
#three type_frame:google_search,image_search,youtube_search
#
class PersonalDataTopicAnalysis:
    
    #initi function
    def __init__(self,file_path):
        #check correctness of the path
        try:
            self.file_path=file_path
            self.dFrame=pd.read_csv(self.file_path,delimiter='\t')
        except Exception as e:
            print('-path error try another path-')
            
    def print_time(text,seconds):
        m=str((second/60)%60).split('.')
        print('{}: {} minutes and {} seconds'.format(text,m[0],m[1]))
           
    #
    #
    #function used for parsening the time stamp: for future time filtering   
    def parse_time_frame(self):
        t1=time()
        timeParser=get_splitted_time()
        self.splitted_time=timeParser.execute(self.dFrame)
        self.splitted_time['activity']=self.dFrame['activity']
        self.splitted_time['name_activity']=self.dFrame['name_activity']
        self.splitted_time['typeSearch']=self.dFrame['typeSearch']
        self.splitted_time['location']=self.dFrame['where']
        t2=time()
        
        print('time elapsed splitting timestamp: {}'.format(t2-t1))
   
    #
    #
    #function used for retreiving title information from url (inner function)
    def get_title(url):              
        hdr = {'User-Agent': 'Mozilla/5.0'}
        title=''
        try:
            req = Request(url,headers=hdr)
            page = urlopen(req)
            soup = BeautifulSoup(page,'html5lib')
            title=soup.title.string
        except Exception as e:
                print(e)
        return title
   
    #
    #
    #function used for retreiving titile information (outer function)
    def retrieveUrlTitles(self):
        titles=[]
        init_t=time()
        print('n url {}'.format(len(self.groupped_visited_url.index)))
        for url in grouped_url.index:
            titles.append(self.get_title(str(url)))
            
    #
    #
    #dividing the frame into two frames: pages visited -pages searched
    def split_entry(self):
        self.searchedF=self.splitted_time[self.splitted_time['typeSearch']=='Searched for'].reset_index()
        self.visitedF=self.splitted_time[self.splitted_time['typeSearch']=='Visited'].reset_index()
        
        self.searchedF=self.searchedF.drop(['activity','typeSearch'],axis=1)
        self.visitedF=self.visitedF.drop('typeSearch',axis=1)
        self.searchedF.rename(columns={'name_activity':'activity'},inplace=True)
    
        
    #
    #
    #filtering the frame acording to the time specified by the user: year-month
    def filter_time(self,year,month,table):
        splitted_time=table[table['year']==year]
        fsplitted_time=splitted_time[splitted_time['month']==month]

        return fsplitted_time
        
            
    #
    #
    #retrive len information about visited-searched tables
    def get_n_entry(self,type_):
        if type_ == 'searched':
            return 'you searched for {} pages in the specified period'.format(len(self.searchedF))
        if type_ == 'visited':
            return 'you visited: {} different pages in the specified period'.format(len(self.groupped_visited_url.index))
    
    #
    ####
    ############Time analysis###################
    #####
    #
     
    #Main function ->show the 'quantity' of time spent for every hour for every day of the week for the
    #specified month of the year
    def time_stats(self,month,year):
        
        #utility functions
        #
        #prepeare data for anaylsis and visualization 
        def preprocess_time(splitted_time):
            
            #normalize time value to 0-10 range
            def normalize(x,max_v,min_v):
                max_n=10
                min_n=0
                v=(((x-min_v)/(max_v-min_v))*(max_n-min_n))+min_n
                return v
    
            #
            def function(data):
                dic={}
                for x in range(0,24):
                    dic[x]=0
                
                for i in data:
                    tmp=i.split(':')
                    key=int(tmp[0])
                    dic[key]+=int(tmp[1])
        
                max_v=max(dic.values())
                min_v=min(dic.values())
    
                dic={k: normalize(v,max_v,min_v) for k,v in dic.items()}
                return collections.OrderedDict(sorted(dic.items()))  
   
            time_stat_s=splitted_time.groupby('day_name')['time'].agg(function)
            time_stat=pd.DataFrame(time_stat_s)
        
            return time_stat
        
        #
        #
        #utility function for plotting the informations
        def show_time_stat(time_stat,month,year):
            fig, axes = pt.subplots(nrows=3,ncols=3,sharey=True,figsize=(15,15))
            fig.suptitle('quantity per hours per day {} for {} {}'.format('\n',month,year))
            x=0
            y=0
            i=0
    
            while i < len(time_stat):
                t=time_stat.iloc[i]
                day=t.name
                dk=t.get_values()
                dic=dict(dk[0])

                tt=pd.DataFrame.from_dict(dic,orient='index')

                if i <=2:
                    x=0
                    y=i
                    tt.plot.bar(ax=axes[x,y])
                    axes[x,y].set(xlabel=day)
                
                elif 3<=i<=5:
                    x=1
                    y=i-3
                    tt.plot.bar(ax=axes[x,y])
                    axes[x,y].set(xlabel=day,ylabel='quantity')
                elif i==6:
                    x=2
                    y=i-6
                    tt.plot.bar(ax=axes[x,y])
                    axes[x,y].set(xlabel=day)
                
                i+=1
                
        #MTime: ain Part##
        #filter according to month of the year specified
        t1=time()
        print('filtering time to {} {}...'.format(month,year))
        splitted_time=self.filter_time(year,month,self.splitted_time)  
        print('preprocessing time...')
        time_stat=preprocess_time(splitted_time)
        show_time_stat(time_stat,month,year)
        t2=time()
        print('time elapsed performing time analysis: {}'.format(t2-t1))
        
    #
    ######
    ######Location Analysis###################
    #####
    
    
    def retrieve_location(self,month,year,tableTopic):
    
        def inner_loc(filtered_time):
            loc_time=pd.DataFrame()
            loc_time[['day','month','time','year','day_name']]=filtered_time[['day','month','time','year','day_name']].copy()
            loc_time['where']=self.splitted_time['location']
        
            groupped_locations=loc_time.groupby('where')['day'].count().reset_index(name='count').sort_values('count',ascending=False).reset_index()
            groupped_locations.drop([0],inplace=True)
            groupped_locations.head(5)
        
            locations=[]
            groupped_locations=groupped_locations[groupped_locations['where'] != '-']
            for loc in groupped_locations['where'].head(100):
                tmp=loc.strip().split('=')
                latlong=tmp[1]
                geolocator = Nominatim()
                try:
                    info=geolocator.reverse(latlong)
                    locations.append(str(info))
                except Exception as e:
                    locations.append('-')
        
            locFrame=pd.DataFrame({
                'loc':locations,
                'count':groupped_locations['count'].head(100)
            })
        
            nations={}
            places={}
            city_nations={}
        
            def split_loc_info(row):
                entry=row['loc']
                info=entry.strip().split(',')
                if(len(info)>2):
                    nation=info[-1]
                    city=info[-4]
                    count=row['count']
    
                    if nation in nations:
                        nations[nation]+=count
                    else:
                        nations[nation]=count
        
                    city_nation=nation+'-'+city
                    if city_nation in city_nations:
                        city_nations[city_nation]+=count 
                    else:
                        city_nations[city_nation]=count
        
            locFrame.apply(split_loc_info,axis=1)
        
            labels=nations.keys()
            sizes=nations.values()

            labels2=city_nations.keys()
            sizes2=city_nations.values()

            fig=pt.figure(figsize=(7,3), dpi=110)
            ax2=pt.subplot2grid((1,2),(0,0))
            ax2.set_title('nations')

            pt.pie(sizes,labels=labels)
            ax3=pt.subplot2grid((1,2),(0,1))
            ax3.set_title('city_nation')
            pt.pie(sizes2,labels=labels2)

            pt.tight_layout(pad=1, w_pad=10.0)
            #pt.show()
            
            return city_nations
        
        #
        #Generate the most frequent location for that given topic
        #
        def retrive_topic_location(tableTopic,filtered_time):
            
            final_tt=filtered_time[['location','activity']]
            final_tt=final_tt.reset_index()
            
            location_for_topic=[]
            l=0
            for l in range(0,self.n_comp):
                t1=tableTopic.iloc[l]['words']
                loc=[]
                for w in t1:
                    i=final_tt[final_tt['activity'].str.find(w)>-1].index
                    t=list(i.values)
  
                    for tt in t:
                        loc.append(final_tt['location'].iloc[tt])
                    table=pd.DataFrame({
                    'where':loc,
                    'count':1
                        })
                table=table[table['where']!='-']
                table=table.groupby('where').count()

                table.sort_values(by=['count'],ascending=False,inplace=True)

                tt=table.index[0]
                tt=tt.split('=')[1]

                latlong=tt
                geolocator = Nominatim()
                try:
                    info=geolocator.reverse(latlong)
                    location_for_topic.append(str(info))
                except Exception as e:
                    pass
                
            return location_for_topic
        

        
        #
        ###Location: Main Part###
        #
        print('filtering time to {} {}...'.format(month,year))
        filtered_time=self.filter_time(year,month,self.splitted_time) 
        print('generating locations...')
        t1=time()
        city_nations=inner_loc(filtered_time)
        loc_topic=retrive_topic_location(tableTopic,filtered_time)
        t2=time()
        print('time elapsed for generating locations: {}'.format(t2-t1))

        return city_nations,loc_topic
        
     
    #
    #####
    ##############Semantic analysis################
    #####topics generations######
    #
    #
    
    #Main function
    def generate_topic(self,month,year,alg):
        
        max_features=300
        n_top_word=20
        self.n_comp=8
        
        #
        #
        #Print most frequent words in a topic/cluster
        def print_top_words(self,model, feature_names, n_top_words):
            for topic_idx, topic in enumerate(model.components_):
                message = "Topic #%d: " % topic_idx
                message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
                print(message)
            print()
    
        #
        #
        #same as above but return the top n words in a frame, for future usage
        def table_topic(model,feature_names,top_words):
            rows=[]
            indx=[]
            for topic_idx,topic in enumerate(model.components_):
                indx.append(topic_idx)
                rows.append([feature_names[i] for i in topic.argsort()[:-top_words - 1:-1]])
 

            topicFrame=pd.DataFrame({
                'f_topic':indx,
                'words':rows
            })
            return topicFrame
    
    
    
        #function that pre-process data before starting the ML algorithms:
        #remove stopwords according to language
        #remove punctuation
        #lemmatize words
        #input pandas_series (corpus)
        
        def clean_words_table(corpus):
    
            stopWords=stopwords.words(['english','italian'])
            stopWords.append('google')
            stopWords.append('facebook')
            stopWords.append('wikipedia')
        
            def hasNumbers(inputString):
                return any(char.isdigit() for char in inputString) 
        
            lemmatize = WordNetLemmatizer()
            punctuation = set(string.punctuation) 
            filtered_corpus=[]
            
            for idx in range(len(corpus)):
                first= ' '.join([word.lower() for word in corpus.iloc[idx].split(' ') if not hasNumbers(word) and word.lower() not in stopWords])
                second = "".join(i for i in first if i not in punctuation)
                corpus_f = " ".join(lemmatize.lemmatize(i) for i in second.split(' '))
                
                if(len(corpus_f)>1):
                    filtered_corpus.append(corpus_f)
            return filtered_corpus
    
        #
        #generate tfidf of the given corpus (for NMF)
        #
        def get_tfidf(corpus,ngram,max_features):
            tfidf=TfidfVectorizer(max_features=max_features,max_df=0.95,lowercase=False,min_df=2,stop_words='english',analyzer='word',token_pattern='(?u)\\w{4,40}\\b',ngram_range=ngram)
            tfidf_matrix=tfidf.fit_transform(corpus)
            #print(tfidf_matrix.shape)
            #print(tfidf_matrix[:5])
    
            return tfidf_matrix,tfidf

        #
        #generating tf (for LDA)
        #
        def get_tf(corpus,ngram,max_features):
            tf=CountVectorizer(max_df=0.97, min_df=2,max_features=max_features,stop_words='english',token_pattern='(?u)\\w{4,40}\\b',ngram_range=ngram)
            tf_matrix=tf.fit_transform(corpus)
            
            return tf_matrix,tf
        
        #
        #generate a pandasFrame from the tfidf/tf factors for future uses
        #
        def get_table(matrix,terms):
            table1=pd.DataFrame(matrix.toarray(),columns=terms)
            #summing over all the tfidf 
            tfidf_table1=table1.sum(axis=0).reset_index().sort_values(0,ascending=False)
            tfidf_table1=tfidf_table1.rename(index=str,columns={tfidf_table1.columns[0]:'word',tfidf_table1.columns[1]:'tfidf'})   
    
            tfidf_table1.head(20).set_index('word').plot(kind='bar')
            pt.ylabel('tfidf')
            
            pt.show()
    
            return tfidf_table1
    
        #
        #Avarage number of words searched
        #
        def avg_wc_search(corpus):
            l=len(corpus)
            n=0
            for entry in corpus:
                n+=len(entry.split(' '))
                
            print('wc in search avg: {}'.format(len_l/n))
                            
         #
        #perform Non-negative-matrix-clustering
        #
        def get_nmf(tfidf_matrix,n_comp):
            nmf=NMF(n_components=n_comp,random_state=1,max_iter=1000, beta_loss='frobenius',solver='mu', l1_ratio=.5,init='nndsvda')
            topic_nmf=nmf.fit_transform(tfidf_matrix)
            
            return nmf
            
        def get_lda(tf_matrix,n_comp):
            lda = LatentDirichletAllocation(n_components=n_comp, max_iter=200,learning_method='batch')
            lda.fit(tf_matrix)
            tf_feature_names = tf.get_feature_names()
            
            return lda
            
        #
        ######functions for automatic labeling######
        ##
        
        #
        #get label candidate using MediaWiki api
        #->querying wikipedia using the api above in the following way:
        #wikipedia category gerarchy and disambiguation category
        #
        #
        
        def get_candidate(table,t_range):
            wiki= MediaWiki()
            cat_dic={}
            cat_dic_freq={}
            cat_miss={}
            #for every cluster created
            for tt in range(0,t_range):
                n_disamb=0
                dic={}
            #for every element in the cluster
                for k in table.iloc[tt]:
                #print('###'+k)
                    try:
                        l=wiki.page(k)
                        for w in l.links[:10]:
                            if '(' in w:
                                if 'disambiguation' not in w:
                                    ww=w.split('(')[1][:-1] 
                                    if len(ww)>3 and len(ww.split(' '))<3:
                                        if ww in dic:
                                            dic[ww]+=1
                                        else:
                                            dic[ww]=1
                #for w in l.categories[:10]:
                #    for k in dic:
                #        if k in w:
                #            dic[k]+=1
                    except DisambiguationError as e:
                        n_disamb+=1
                    #print('*** disamb')
                        for w in e.options[:10]:
                            if '(' in w:
                                if 'disambiguation' not in w:
                                    ww=w.split('(')[1][:-1] 
                                    if len(ww)>4 and len(ww.split(' '))<3:
                                        if ww in dic:
                                            dic[ww]+=1
                                        else:
                                            dic[ww]=1
                                
                    except PageError:
                        n_disamb+=1
                        pass
                                                            
            #print(list(dic.items())[:3])
            #return only the title of topic->no freq
                cat_dic[tt]=list(OrderedDict(sorted(dic.items(),key=lambda x:x[1],reverse=True)[:2]))
                cat_dic_freq[tt]=(list(OrderedDict(sorted(dic.items(),key=lambda x:x[1],reverse=True)).values())[:3],len(dic))
                cat_miss[tt]=n_disamb
            return cat_dic,cat_dic_freq
                                                            
          
                
        
        
        
        #
        ##############Topic: Main part########à#######
        #
        
        
        print('filtering time to {} {}...'.format(month,year))
        
        splitted_time=self.filter_time(year,month,self.searchedF) 
        print('preprocessing corpus...')
        corpusdata=splitted_time['activity']
        f_corpusdata=clean_words_table(corpusdata)
        
        if alg == 'NMF':
            print('generating tfidf factor...')
            tfidf_matrix,tfidf=get_tfidf(f_corpusdata,(1,2),max_features)
            tfidf_table1=get_table(tfidf_matrix,tfidf.get_feature_names())
        
            print('generating topic with NMF...')
            t1=time()
            nmf=get_nmf(tfidf_matrix,self.n_comp)
            tableTopic=table_topic(nmf,tfidf.get_feature_names(),20)
            t2=time()
            print('time elapsed for topic generation: {}'.format(t2-t1))
            
        elif alg == 'LDA':
            print('generating tf factor...')
            tf_matrix,tf=get_tf(f_corpusdata,(1,2),max_features)
            tf_table1=get_table(tf_matrix,tf.get_feature_names())
        
            print('generating topic with LDA...')
            t1=time()
            lda=get_lda(tf_matrix,self.n_comp)
            tableTopic=table_topic(lda,tf.get_feature_names(),20)
            t2=time()
            print('time elapsed for topic generation: {}'.format(t2-t1))
            
        print('generating labels for topics...')
        t1=time()
        topic_candidates,f=get_candidate(tableTopic['words'],self.n_comp)
        t2=time()
        print(' time elapsed for labeling generation: {}'.format(t2-t1))
        
        tableTopic['labels']=topic_candidates.values()
        tableTopic['year']=year
        tableTopic['month']=month
        return tableTopic
    
    def info_table(self,month,year):
        tableTopic=self.generate_topic(month,year,'NMF')
        
        city_nations,location_for_topic=self.retrieve_location(month,year,tableTopic)
        l_=[]
        print(location_for_topic)
        for l in location_for_topic:
            t=l.split(',')
            if re.match('^[0-9]*[/ \w]*',t[0]):
                ll=t[1]+':'+t[4]+':'+t[-1]
            else:
                ll=t[0]+':'+t[4]+':'+t[-1]
                
            l_.append(ll)
        tableTopic['location']=l_
        return tableTopic
            
           
    #
    #exe function
    def execute(self):
        self.parse_time_frame()
        self.split_entry()
        
        #group visited pages by over same url
        self.groupped_visited_url=self.visitedF.groupby('activity').count()

