
# coding: utf-8

# In[1]:


#all libraries needed

import pandas as pd
import json as js
from bs4 import BeautifulSoup
import codecs
import os 
import re
from time import time


# In[3]:


#
#Class that taking the directory path of your google data set, will generate the cvs from My Activity directory, which contains
#the most interesting data to process
#
#instantiate the class and call .execute()
#

class gCsvConverter():
    
    def __init__(self,readingPath,writingPath):
        
        #checking validity of the inputs
        if(os.path.exists(readingPath)==False or os.path.isdir(readingPath)==False):
            print("reading path doesn't exist or is not a valid directory path, please enter another one")
            
        if(os.path.exists(writingPath)==False or os.path.isdir(writingPath)==False ):
            print("writing path doesn't exist or is not a valid directory path, please enter another one")
                
        self.readingPath=readingPath
        self.writingPath=writingPath
        
        self.readingPath=self.readingPath+'/My Activity'
        
        self.elements=[]
        for el in os.listdir(self.readingPath):
            self.elements.append(el)
            
    def googleHtml_toCsv(self,path):
        name=path.split('/')[-1]
        name=name+'.csv'
        
        print('generating table: {}'.format(name))
        
        #final path to write to
        w_path=self.writingPath+'/'+name
        
        #reading and validating html file from reading directory
        inner_path=self.readingPath+'/'+path
        html=[el for el in os.listdir(inner_path) if re.match('^.*\.html',el)][0]
                
        if len(html)>1: 
            r_html=inner_path+'/'+html
            f = codecs.open(r_html, 'r', 'utf-8')
            document= BeautifulSoup(f.read(),'html5lib')

            contents=document.body.find('div',attrs={'class':'mdl-grid'})
            elements=contents.findAll('div',attrs={'class':'outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp'})

            activity=[]
            name=[]
            time_list=[]
            location=[]
            typeSearch=[]
    
            for el in elements:
                if el.a !=None:
                    activity.append(el.a['href'])
                    name.append(el.a.text)
                    time_list.append(el.a.next_sibling.next_sibling)
                    if el.a.previousSibling !=None:
                        type_=el.a.previousSibling.strip().split('/')[0]
                        typeSearch.append(type_)
                    else:
                        typeSearch.append('-')
                else:
                    typeSearch.append('-')
                    tt=el.find('div',attrs={'class':'content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1'})
                    if tt != None:
                        ts=str(tt).replace("<br/>","-")
                        tn=BeautifulSoup(ts,"html5lib").text.strip().split('-')
                        activity.append(tn[0])
                        name.append(tn[0])
                        time_list.append(tn[1])
            
                loc=el.find('div',attrs={'class':'content-cell mdl-cell mdl-cell--12-col mdl-typography--caption'})
                if loc !=None:
                    if loc.a !=None:
                        location.append(loc.a['href'])
                    else:
                        location.append('-')
        
        
            dFrame=pd.DataFrame({
                'activity':activity,
                'name_activity':name,
                'when':time_list,
                'where':location,
                'typeSearch':typeSearch
            })
            dFrame.to_csv(w_path,sep='\t')
        
                
    
          
    def execute(self):
        t1=time()
        print('generating tables...')
        for path in self.elements:
            self.googleHtml_toCsv(path)
            
        t2=time()
        print('time elapsed generating csv tables {}'.format(t2-t1))
            
                
        
                  


# In[4]:


g=gCsvConverter('/home/janz/keras/datas/json_google','/home/janz/keras/datas/csv/chrome')
g.execute()


# In[1]:


#class for generating csv from facebook

class fCsvConverter:
    def __init__(self,readingPath,writingPath):
        if(os.path.exists(readingPath)==False or os.path.isdir(readingPath)==False):
            print("reading path doesn't exist or is not a valid directory path, please enter another one")
            
        if(os.path.exists(writingPath)==False or os.path.isdir(writingPath)==False ):
            print("writing path doesn't exist or is not a valid directory path, please enter another one")
                
        self.readingPath=readingPath
        self.writingPath=writingPath
        
        self.elements=[el for el in os.listdir(self.readingPath) if re.match('^.*\.htm',el)]
        
        self.mDir= [el for el in os.listdir(self.readingPath) if el not in self.elements]
    
    def generateCsv(self,path):
        
        #
        #functions for retreiving csv from contact file
        #
        def table_tag(parent):
            table=parent.find('table')
            check=table.find_all('tr')

            title=[]
            argument=[]
            for tr in check[1:]:
                title.append(tr.td.text)
                argument.append(tr.find('span',attrs={'class':'meta'}))

            f=[]

            for el in argument:
                t=[]
                for li in el.ul:
                    if li.name == 'li':
                        t.append(li.text)
                    else:
                        t.append(li)
            
                l=''.join(str(e) for e in t)
                f.append(l)
    
            dFrame=pd.DataFrame({
                'title':title,
                'arg':f
                })
            return dFrame
        
        #
        #csv from timeline file
        #
        def meta_tag(parent):
            check= parent.find_all('div', attrs={'class':'meta'})

            time=[]
            next_child=[]
            for i in check:
                time=i.text
                next_child.append(i.next_sibling)
    
            finalDframe=pd.DataFrame(
            {
                'date_time':time,
                'operation':next_child
            })

            return finalDframe
        #
        #informations from security tag
        #
        def sec_tag(parent):
            titols=[]
            for tt in parent.find_all('h2'):
                titols.append(tt.text)
            elements=contents.find_all('ul')

            dic={}
            i=0
            for ul in elements:
                l=[]
                for li in ul:
                    t=li.find('p')
                    if(t!=None):
                        l.append(t.text)
                    else:
                        l.append(li)
                dic[titols[i]]=l
                i+=1


            FFrame=pd.DataFrame.from_dict(dic,orient='index')
            return (FFrame.T)
        
        def ul_tag(parent):
            titols=[]
            for tt in parent.find_all('h2'):
                titols.append(tt.text)
            elements=parent.find_all('ul')
    
            dic={}
            i=0
    
            for ul in elements:
                l=[]
                for li in ul:
                    l.append(li.text)
                dic[titols[i]]=l
                i+=1
   
            UFrame=pd.DataFrame.from_dict(dic,orient='index')
            return UFrame.T

        #take the directory path that contains all the html file->messages
        #take the directory path that contains all the html file->messages
        def parseMessagges(path,finalPath):
            el=  os.listdir(path)
            el = [w for w in el if re.match('^.*\.(html|htm)',w)]

            mFrame=pd.DataFrame(columns=['c_id','title','partic','time','user','message'])
            
            ix=0
            iD = ms1 = f = document = None
            for l in el:
                iD=l.split('.')[0]
                #print(iD)
                ms1 = path+'/'+l
                f = codecs.open(ms1, 'r', 'utf-8')
                document= BeautifulSoup(f.read(),'html5lib')
        
                thread = title = participants = None
        
                try:
                    thread=document.body.find('div', attrs={'class':'thread'})
                    title=thread.h3.text
                    participants=thread.h3.next_sibling
                    ms_el=thread.findAll('div',attrs={'class':'message'})
            
                    i=0
                    final_list=[]
            
                    for index in range(0,len(ms_el)):
                        i=thread.index(ms_el[index])
    
                        e=list(thread)[i+1].text
                        if e!='':
                            final_list.append(ms_el[index])
                            final_list.append(e)
                        elif i+2<len(list(thread)):
                            #check the next one
                            e=list(thread)[i+2]
                            if e.name=='p':
                                final_list.append(ms_el[index])
                                final_list.append(e.text)
                        
                    for index,s in enumerate(final_list):
                        if type(s) is not str:
                            #tag with info:
                            header=s.find('div',attrs={'class':'message_header'})
                            users=header.find('span',attrs={'class':'user'}).text
                            times=header.find('span',attrs={'class':'meta'}).text
                            mFrame.loc[ix]=[iD,title,participants,times,users,final_list[index+1]]
                        ix+=1
                
                    final_list.clear()
                    ms_el.clear()
    
                except AttributeError as e:
                    pass

            name='/messages.csv'
            return mFrame
            #mFrame.to_csv(finalPath+name,sep='\t')


        
        
        #main section
            
        name=path.split('.')[-2]
        f_name=name+'.csv'
        no_table=['videos','photos','pokes']
        if name not in no_table:
            print('generating table {}'.format(name))
        w_path=self.writingPath+'/'+f_name
        
        f_path=self.readingPath+'/'+path
    
        f = codecs.open(f_path, 'r', 'utf-8')
        document= BeautifulSoup(f.read(),'html5lib')
        contents=document.body.find('div', attrs={'class':'contents'})
        
        if name == 'messages':
            dFrame=parseMessages(name)
        elif name == 'contact_info':
            dFrame=table_tag(contents)
        elif name == 'timeline':
            dFrame=meta_tag(contents)
        elif name == 'security':
            dFrame = sec_tag(contents)
        elif name in no_table:
            return 
        else:
            dFrame=ul_tag(contents)
        
        dFrame.to_csv(w_path,sep='\t')
        
    def execute(self):
        t1=time()
        print('generating tables from facebook...')
        for el in self.elements:
            self.generateCsv(el) 
            
        t2=time()
        print('time elapsed {}'.format(t2-t1))
        
        #todo
        #self.parseMessages(self.mDir)
                

