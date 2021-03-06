
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
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.collocations import *
import string
nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from time import time
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
            self.punctuation = set(string.punctuation)
            return tempFrame


# In[37]:


#Class that performs the analysis on text data generated from google or facebook, it returns a table with the
#topics-time-location of the data
#
#
#three type_frame:google_search,image_search,youtube_search
#
# the path needed is the path location of the table you want to use for analysis, generated by the CreateCsvClass
#
class PersonalDataTopicAnalysis:
    
    #initi function
    def __init__(self,file_path):
        #check correctness of the path
        try:
            self.file_path=file_path
            self.dFrame=pd.read_csv(self.file_path,delimiter='\t')
            print('data frame succesfuly created...')
            self.timeParser=get_splitted_time()
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
        
        self.splitted_time=self.timeParser.execute(self.dFrame)
        self.splitted_time['activity']=self.dFrame['activity']
        self.splitted_time['name_activity']=self.dFrame['name_activity']
        self.splitted_time['typeSearch']=self.dFrame['typeSearch']
        self.splitted_time['location']=self.dFrame['where']
        t2=time()
        
        print('time elapsed splitting timestamp: {}'.format(t2-t1))
            
    #
    #
    #dividing the frame into two frames: pages visited -pages searched
    def split_entry(self):
        if 'Visited' in list(self.splitted_time['typeSearch']):
            self.type_table='search'
            self.searchedF=self.splitted_time[self.splitted_time['typeSearch']=='Searched for'].reset_index()
            self.visitedF=self.splitted_time[self.splitted_time['typeSearch']=='Visited'].reset_index()
        elif 'Watched' in list(self.splitted_time['typeSearch']):
            self.type_table='youtube'
            self.searchedF=self.splitted_time[self.splitted_time['typeSearch']=='Searched for'].reset_index()
            self.visitedF=self.splitted_time[self.splitted_time['typeSearch']=='Watched'].reset_index()
        
        elif 'Viewed image from' in list(self.splitted_time['typeSearch']):
            self.type_table='image'
            self.searchedF=self.splitted_time[self.splitted_time['typeSearch']=='Searched for'].reset_index()
            self.visitedF=self.splitted_time[self.splitted_time['typeSearch']=='Viewed image from'].reset_index()
            
        self.searchedF=self.searchedF.drop(['activity','typeSearch'],axis=1)
        self.visitedF=self.visitedF.drop(['typeSearch','location','name_activity'],axis=1)
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
                min_v=0
    
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
    
    
    def retrieve_location(self,year,month,tableTopic):
    
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
            pt.pie(sizes, shadow=True, startangle=90)
            pt.legend(labels,loc='lower right')
            pt.axis('equal')
            pt.tight_layout()
            ax3=pt.subplot2grid((1,2),(0,1))
            ax3.set_title('city_nation')
            pt.pie(sizes2, shadow=True, startangle=90)
            pt.legend(labels2,loc='lower right')
            pt.axis('equal')
            #pt.tight_layout(pad=1, w_pad=10.0)
            pt.tight_layout()

            pt.show()
            
            return city_nations
        
        #
        #Generate the most frequent location for that given topic
        #
        def retrive_topic_location(tableTopic,filtered_time):
            
            final_tt=filtered_time[['location','name_activity']]
            final_tt=final_tt.reset_index()
            
            location_for_topic=[]
            
            l=0
            #for every topic
            for l in range(0,self.n_comp):
                loc=[]
                #take the words 
                t1=tableTopic.iloc[l]['words']
                #for every words in topic
                for w in t1:
                    #find the index in the words->for retreving the locations
                    i=final_tt[final_tt['name_activity'].str.find(w)>-1].index
                    t=list(i.values)
                      
                    #for every accurance of that word
                    for tt in t:
                        loc.append(final_tt['location'].iloc[tt]) 
                        #all the locations for that given word
                        
                table=pd.DataFrame({
                    'where':loc,
                    'count':1
                    })
                
                table=table[table['where']!='-']
                table=table.groupby('where').count()

                table.sort_values(by=['count'],ascending=False,inplace=True)

                most_loc=table.index[0]
                most_loc=most_loc.split('=')[1]

                geolocator = Nominatim()
                try:
                    info=geolocator.reverse(most_loc)
                    location_for_topic.append(str(info))
                except Exception as e:
                    pass
                
            return location_for_topic
        

        
        #
        ###Location: Main Part###
        #
        #print('filtering time to {} {}...'.format(month,year))
        filtered_time=self.filter_time(year,month,self.splitted_time) 
        city_nations=inner_loc(filtered_time)
        loc_topic=retrive_topic_location(tableTopic,filtered_time)

        return city_nations,loc_topic
        
     
    #
    #####
    ##############Semantic analysis################
    #####topics generations######
    #
    #
    
    #Main function
    def generate_topic(self,month,year,alg):
        
        max_features=100
        n_top_word=22
        self.n_comp=7
        
        stopWords=stopwords.words(['english','italian'])
        stopWords.append('google')
        stopWords.append('facebook')
        stopWords.append('wikipedia')
        stopWords.append('yahoo')
        stopWords.append('wiki')
        stopWords.append('wikia')
        
        
    
    #
    #
    #function used for retreiving title information from url (inner function)
        def get_title(url):              
            hdr = {'User-Agent': 'Mozilla/5.0'}
            title=''
            try:
                req = Request(url,headers=hdr)
                page = urlopen(req,timeout=30)
                soup = BeautifulSoup(page,'html5lib')
                if soup.title != None:
                    title=soup.title.string
                else:
                    return None
            except Exception as e:
                return None
            return title
   
    #
    #
    #function used for retreiving titile information (outer function)
        def retrieveUrlTitles(visitedT):
            titles=[]
            totN=0
            print('retreiving visited pages titles...')
            print('n visited pages {}'.format(len(visitedT)))
            if len(visitedT)>150:
                print('Exeeding number of visited pages, trimming them...')
                totN=150
            else:
                totN=len(visitedT)
            
            init_time=time()
            
            #print('n url {}'.format(len(visitedT.index)))
            for url in list(visitedT.index)[:totN]:
                title= get_title(str(url))
                if title != None:
                    titles.append(title)
            f_time=time()
            print('retriving titles elappsed time: {}'.format(f_time-init_time))
            return titles
        
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
        def hasNumbers(inputString):
            return any(char.isdigit() for char in inputString)
        
        def clean_words_table(corpus):

            lemmatize = WordNetLemmatizer()
            punctuation = set(string.punctuation) 
            filtered_corpus=[]
            for idx in range(len(corpus)):
                first= ' '.join([word.lower() for word in corpus.iloc[idx].split(' ') if not hasNumbers(word) and word.lower() not in stopWords])
                second = "".join(i for i in first if i not in punctuation)
                corpus_f = " ".join(lemmatize.lemmatize(i) for i in second.split(' '))
                
                if(len(corpus_f)>1):
                    filtered_corpus.append(corpus_f)
            return pd.DataFrame({'activity':filtered_corpus})
        
        def clean_words(corpus):
            #remove stopwords according to language
            #remove punctuation
            #lemmatize words
            lemmatize = WordNetLemmatizer()
            punctuation = set(string.punctuation) 
            filtered_corpus=[]
            for entry in corpus:
                entry=str(entry).strip()
                entry=entry.replace('-','')
                entry=entry.replace('–','')
                entry=entry.replace('&','')
                entry=entry.replace(',','')
                entry=entry.replace('|','')
                entry=entry.replace('\n','')
                entry=entry.replace('\t','')
                entry=entry.replace('\t','')
                entry=entry.replace('•','')
                entry=entry.replace('»','')
                entry=entry.replace(r'\w*\.\w*','')
                #print(entry.split(' '))
        
                first= ' '.join([word.lower() for word in entry.split(' ') if not hasNumbers(word) and word.lower() not in stopWords])
                first_=''.join([w for w in first if not re.match(r'\w*\.\w*',str(w))])
                second = "".join(i for i in first_ if i not in punctuation)
                final = " ".join(lemmatize.lemmatize(i) for i in second.split(' '))
                corpus_f=' '.join(final.split())
                if(len(corpus_f)>1):
                    filtered_corpus.append(corpus_f.strip())
            
            return filtered_corpus
        
        def pre_process_url(urls_vect):
            tt=[w for w in urls_vect if w!='']
            tt=[w for w in tt if str(w).find("Google")==-1]
            
            title_=[]
            for t in tt:
                if t is not None:
                    if '|' in t:
                        title_.append(' '.join(t.strip().split('|')[:2]))
                    elif '–' in t:
                        title_.append(' '.join(t.strip().split('–')[:2]))
                    else:
                        title_.append(t)
                        
            table_title=pd.DataFrame({'titles':title_})
            groupped_titles=table_title.groupby('titles').count().reset_index()
            return groupped_titles
            
    
        #
        #generate tfidf of the given corpus (for NMF)
        #
        def get_tfidf(corpus,ngram):
            tfidf=TfidfVectorizer(max_df=0.9,lowercase=False,min_df=2,stop_words='english',analyzer='word',token_pattern='(?u)\\w{4,40}\\b',ngram_range=ngram)
            tfidf_matrix=tfidf.fit_transform(corpus)
            #print(tfidf_matrix.shape)
            #print(tfidf_matrix[:5])
    
            return tfidf_matrix,tfidf

        #
        #generating tf (for LDA)
        #
        def get_tf(corpus,ngram):
            tf=CountVectorizer(max_df=0.97, min_df=2,stop_words='english',token_pattern='(?u)\\w{4,40}\\b',ngram_range=ngram)
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
        
        def get_candidate(table):
            
            def to_sing(s):
                if s[-1] == 's' and s[-2]!='s':
                    return s[:-1]
                else:
                    return s
                
            def n_grams(s):
                l=list(itertools.combinations(s,2))
                f=[]
                for el in l:
                    f.append(' '.join(el))
                return f
            
            def update_dic(dic,ww,inc,n_cat):
                ww=ww.lower()
                ww=to_sing(ww)
                if ww in dic:
                    dic[ww]+=inc
                elif n_cat>0:
                    n_cat-=1
                    dic[ww]=inc
            
                return n_cat
            
            def operate_on_label(dic,ww,n_cat):
                if len(ww)>3 and 'disambiguation' not in ww and not re.match(r'^[0-9]+[\w \s]*',ww): 
                    if len(ww.split(' '))>1 and len(ww.split(' '))<3:
                        #category of multiple words
                        n_cat=update_dic(dic,ww,1.1,n_cat)  
                        w_list=list(ww.split(' '))
                        #l=n_grams(ww.split(' '))
                        #w_list.extend(l)
                        for www in w_list:
                            n_cat=update_dic(dic,www,0.6,n_cat)
                        w_list.clear()
                    else:
                        #single word category
                        n_cat=update_dic(dic,ww,1,n_cat)
        
                return n_cat
            
            def control_final_labels(dic):

                ff={}
                final_labels={}
                for i in range(0,self.n_comp):
                    tt=list(dic[i].keys())

                    l=len(dic[i])
                    index_f=0
                    index_s=1

                    #most rated label
                    f=tt[index_f]
                    #second most rated label
                    s=tt[index_s]
    
                    #if the combination of the two most candidate label is present in the list->take the combination
                    if f+' '+s in tt:
                        f=f+' '+s
                        tt.remove(f)
                        l=l-1
                        #take the next most candidate label
                        index_s+=1
                        s=tt[index_s]
    
                    #if one candidate label is cointained in another
                    #if the first one is contained in the second one
                    elif s.find(f)>-1:
                        l=l-1
                        tt.remove(f)
                        f=s
                        index_s+=1
                        s=tt[index_s]
                
                    #if the second one is contained in the first secoon 
                    elif f.find(s)>-1:
                        tt.remove(s)
                        l=l-1
                        index_s+=1
                        s=tt[index_s]
                
                    
                    #setting treshold for checking label validity
                    if len(s.split(' '))>1:
                        tr=1.1
                    else:
                        tr=1
                    if len(f.split(' '))>1:
                        tr1=1.1
                    else:
                        tr1=1
            
                    if dic[i][s]>tr:
                        final_labels[i]=[f,s]
                    elif dic[i][f]>tr1:
                        final_labels[i]=[f]
                    else:
                        final_labels[i]=None
                
                    if final_labels is not None:
                        ff[i]=([dic[i][f],dic[i][s]],l)
                    else:
                        ff[i]=(0,0,0)
                        
                return final_labels,ff
                    
            wiki= MediaWiki()
            cat_dic={}
            cat_dic_freq={}
            cat_dissamb={}
            cat_not_found={}
            punctuation=set(string.punctuation)
            #for every cluster created
            for tt in range(0,self.n_comp):
                n_disamb=0
                n_miss=0
                dic={}
                n_cat=30
            #for every element in the cluster
                for k in table.iloc[tt]:
                #print('###'+k)
                    try:
                        l=wiki.page(k)
                        for w in l.links[:21]:
                            if '(' in w and w.endswith(')'):
                                ww=w[w.find('(')+1:w.find(')')]
                                if len(ww.split(' '))<3:
                                    n_cat=operate_on_label(dic,ww,n_cat)
                        for w in l.categories[:21]:
                            if '(' in w and w.endswith(')'):
                                ww=w[w.find('(')+1:w.find(')')]
                                if len(ww.split(' '))<3:
                                    n_cat=operate_on_label(dic,ww,n_cat)
                            elif len(w.split(' '))<4 and len(w)>3:
                                    n_cat=operate_on_label(dic,w,n_cat)
                            
                    except DisambiguationError as e:
                        n_disamb+=1
                        for w in e.options[:21]:
                            if '(' in w and w.endswith(')'):
                                ww=w[w.find('(')+1:w.find(')')]
                                if len(ww.split(' '))<3:
                                    n_cat=operate_on_label(dic,ww,n_cat)
                                
                    except PageError:
                        n_miss+=1
                                                            
                cat_dic[tt]=OrderedDict(sorted(dic.items(),key=lambda x:x[1],reverse=True))
                print(cat_dic[tt])
                #cat_dic_freq[tt]=(list(OrderedDict(sorted(dic.items(),key=lambda x:x[1],reverse=True)).values())[:2],len(dic))
                cat_dissamb[tt]=n_disamb
                cat_not_found[tt]=n_miss
        
            final_labels={}
            final_labels,cat_dic_freq=control_final_labels(cat_dic)
            
            return final_labels,cat_dic_freq,[cat_dissamb,cat_not_found]
        def truncate(f, n):
        #Truncates/pads a float f to n decimal places without rounding
            s = '{}'.format(f)
            if 'e' in s or 'E' in s:
                return '{0:.{1}f}'.format(f, n)
            i, p, d = s.partition('.')
            return float('.'.join([i, (d+'0'*n)[:n]]))
        
        #
        #
        #calculate topic labeling precision
        def calculate_precision(freq,n_comp):
    
            acc_vect=[]
            s_t=0
            for i,k in freq.items():
                y=sum(k[0][:2])
                s_perc=y/k[1]
                acc_vect.append(truncate(s_perc*100,2))
            return acc_vect
    
        #
        #
        #return disamb and page not found total percentage
        def calculate_failure_rate(f_vec,n_el):
            failure_v=[]
            for k,v in f_vec[1].items():
                perc_failure_topic=v/n_el
                failure_v.append(truncate(perc_failure_topic*100,2))
            return failure_v    
                                                         
        #
        ##############Topic: Main part########à#######
        #
        
        
        print('filtering time to {} {}...'.format(month,year))
        
        searchedF_=self.filter_time(year,month,self.searchedF) 
        visitedF_=self.filter_time(year,month,self.visitedF) 
        
        print('preprocessing corpus...')
        
        #searched entries preporeccing
        f_corpusdata_searched=clean_words_table(searchedF_['activity'])
        gg=f_corpusdata_searched.groupby('activity').count().reset_index()
        final_searched=list(gg[gg['activity']!='']['activity'])
        print('final processed visited entries: {}'.format(len(final_searched)))

        #visited entry preprocessing
        grouped_url=visitedF_.groupby('activity').count()
        titles=retrieveUrlTitles(grouped_url)

        f_corpusdata_visited=pre_process_url(titles)
        title_cleaned=clean_words(f_corpusdata_visited['titles'])
        
     
        final_entries=[]
        final_entries=final_searched.copy()
        final_entries.extend(title_cleaned) 
        
        ##
        
        if alg == 'NMF' or alg == 'nmf':
            print('generating tfidf factor...')
            tfidf_matrix,tfidf=get_tfidf(final_entries,(1,2))
            tfidf_table1=None
            tfidf_table1=get_table(tfidf_matrix,tfidf.get_feature_names())
        
            print('generating topic with NMF...')
            t1=time()
            nmf=get_nmf(tfidf_matrix,self.n_comp)
            tableTopic = None
            tableTopic=table_topic(nmf,tfidf.get_feature_names(),n_top_word)
            t2=time()
            print('time elapsed for topic generation: {}'.format(t2-t1))
            
        elif alg == 'LDA' or alg == 'lda':
            print('generating tf factor...')
            tf_matrix,tf=get_tf(final_entries,(2,2))
            tf_table1=None
            tf_table1=get_table(tf_matrix,tf.get_feature_names())
        
            print('generating topic with LDA...')
            t1=time()
            lda=get_lda(tf_matrix,self.n_comp)
            tableTopic = None
            tableTopic=table_topic(lda,tf.get_feature_names(),n_top_word)
            t2=time()
            print('time elapsed for topic generation: {}'.format(t2-t1))
            
        print('generating labels for topics...')
        t1=time()
        topic_candidates = f = failure_val = None
        topic_candidates,f,failure_val=get_candidate(tableTopic['words'])
        t2=time()
        print(' time elapsed for labeling generation: {}'.format(t2-t1))
        
        tableTopic['labels']=topic_candidates.values()
        tableTopic['year']=year
        tableTopic['month']=month
        ####here
        self.precision=calculate_precision(f,self.n_comp)
        self.failure=calculate_failure_rate(failure_val,n_top_word)
        return tableTopic
    
    
    def info_table(self,month,year,type_alg):
        #group visited pages by over same url
        #check if month and year are present:
        if self.searchedF[(self.searchedF['month']==month) & (self.searchedF['year']==year)].empty:
            print('combination of month and year not present: please try another month')
            return None
        tableTopic=self.generate_topic(month,year,type_alg)
        
        if self.type_table=='search':
            print('generation locations for topic...')
            t1=time()
            city_nations,location_for_topic=self.retrieve_location(year,month,tableTopic)
        
            l_=[]
            #print(location_for_topic)
            for l in location_for_topic:
                t=l.split(',')
                if re.match(r'^[0-9]*[/ \w]*',t[0]):
                    ll=t[1]+':'+t[4]+':'+t[-1]
                else:
                    ll=t[0]+':'+t[4]+':'+t[-1]
                
                l_.append(ll)
            tableTopic['location']=l_
            t2=time()
            print('time ellapsed for generatin locations :{}'.format(t2-t1))
        else:
            print('the following table does not support location generation')
        
        
        tableTopic['words']=tableTopic['words'].apply(lambda x: ','.join(x))
        tableTopic['labels']=tableTopic['labels'].apply(lambda x:','.join(x) if (x is not None) else None)
        return tableTopic
            
           
    #
    #exe function
    def execute(self):
        try:
            self.parse_time_frame()
            self.split_entry()
            print('resource data type: {}'.format(self.type_table))
        
            self.groupped_visited_url=self.visitedF.groupby('activity').count()
        except AttributeError:
            print('error parsing data frame')
        

