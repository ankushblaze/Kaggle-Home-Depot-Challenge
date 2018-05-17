'''
Created on Apr 2, 2016
@author: Harsha Mysur
'''

import os
import pandas as pd
from pandas import *
import re
from collections import Counter
import math
import csv
import time

resultlist = []

# read all the files

data = pd.read_csv('train.csv',encoding="ISO-8859-1")
descdata = pd.read_csv('product_descriptions.csv',encoding="ISO-8859-1")
attrdata = pd.read_csv('attributes.csv',encoding="ISO-8859-1")
testdata = pd.read_csv('test.csv',encoding="ISO-8859-1")

# calculate the tf of all the product titles and calculate the cosine similarity between the search term and product titles

title_size = len(data)
data['sep'] = data.product_title.apply(lambda rec: rec.replace(',',' '))

data['clean'] = data.sep.apply(lambda rec: re.sub("[\s]", " ", rec.lower().strip()).split())

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

data['stemed'] = data.clean.apply(lambda words: [stemmer.stem(x) for x in words])
data['freq'] = data.stemed.apply(lambda words: Counter(words))
data['log_freq'] = data.freq.apply(lambda d: dict([(k,math.log(v) + 1) for k, v in d.items()]))

del data['clean']
del data['freq']
del data['sep']

data['srchtmclean'] = data.search_term.apply(lambda rec: re.sub("[\s]", " ", rec.lower().strip()).split())
data['srchtmstemed'] = data.srchtmclean.apply(lambda words: [stemmer.stem(x) for x in words])
data['srchtmfreq'] = data.srchtmstemed.apply(lambda words: Counter(words))
data['srchtmlog_freq'] = data.srchtmfreq.apply(lambda d: dict([(k,math.log(v) + 1) for k, v in d.items()]))

del data['srchtmclean']
del data['srchtmstemed']
del data['srchtmfreq']
del data['stemed']

def valuation_length():
	for i, row in data.iterrows():
		len = 0
		srchlen = 0
		flen = 0
		srchflen = 0
		elems = row['log_freq']	
		srchelems = row['srchtmlog_freq']	
		for key,value in elems.items():
			len = len + (value * value)
		flen = math.sqrt(len)
		for key,value in srchelems.items():
			srchlen = srchlen + (value * value)
		srchflen = math.sqrt(srchlen)
		data.set_value(i,'tfidflen',flen)
		data.set_value(i,'tflen',srchflen)
	return

valuation_length()

def valuation_titlesrch():
	for i, row in data.iterrows():
		val = 0
		elems = row['log_freq']	
		srchelems = row['srchtmlog_freq']	
		doclength = row['tfidflen']
		querylength = row['tflen']
	
		for key,value in srchelems.items():
			if key in elems:
				val = val + (value * elems.get(key))
		querydoc = val /(doclength * querylength)
		data.set_value(i,'titlesrch',querydoc)
	return

valuation_titlesrch()

# calculate the tf of all the product description and calculate the cosine similarity between the search term and its corresponding product descriptions

desc_size = len(descdata)
descdata['sep'] = descdata.product_description.apply(lambda rec: rec.replace(',',' '))
descdata['clean'] = descdata.sep.apply(lambda rec: re.sub("[\s]", " ", rec.lower().strip()).split())
descdata['stemed'] = descdata.clean.apply(lambda words: [stemmer.stem(x) for x in words])
descdata['freq'] = descdata.stemed.apply(lambda words: Counter(words))
descdata['log_freq'] = descdata.freq.apply(lambda d: dict([(k,math.log(v) + 1) for k, v in d.items()]))

del descdata['clean']
del descdata['freq']
del descdata['stemed']
del descdata['sep']

def valuation_desclength():
	for i, row in descdata.iterrows():
		len = 0
		flen = 0
		elems = row['log_freq']		
		for key,value in elems.items():
			len = len + (value * value)
		flen = math.sqrt(len)
		descdata.set_value(i,'tfidflen',flen)
	return

valuation_desclength()

descinfo = pd.merge(data, descdata, on='product_uid')
descresult = descinfo.drop(['product_title','product_description','search_term', 'relevance','titlesrch','log_freq_x','tfidflen_x'], axis=1)

def valuation_descsrch():
	for i, row in descresult.iterrows():
		val = 0
		elems = row['log_freq_y']	
		srchelems = row['srchtmlog_freq']	
		doclength = row['tfidflen_y']
		querylength = row['tflen']
		for key,value in srchelems.items():
			if key in elems:
				val = val + (value * elems.get(key))
		querydoc = val /(doclength * querylength)
		descresult.set_value(i,'descsrch',querydoc)
	return

valuation_descsrch()

# calculate the tf of all the product attributes and calculate the cosine similarity between the search term and its corresponding product attributes

attrdata['value'] = attrdata['value'].astype(str)
attrdata['name'] = attrdata['name'].astype(str)

atrdata = (attrdata.groupby('product_uid').agg(lambda x: ' '.join(set(x)))).reset_index()
atrdata['namevalue'] = atrdata[['name', 'value']].apply(lambda x: ' '.join(x), axis=1)

attr_size = len(atrdata)
atrdata['sep'] = atrdata.namevalue.apply(lambda rec: rec.replace(',',' '))
atrdata['clean'] = atrdata.sep.apply(lambda rec: re.sub("[\s]", " ", rec.lower().strip()).split())
atrdata['stemed'] = atrdata.clean.apply(lambda words: [stemmer.stem(x) for x in words])
atrdata['freq'] = atrdata.stemed.apply(lambda words: Counter(words))
atrdata['log_freq'] = atrdata.freq.apply(lambda d: dict([(k,math.log(v) + 1) for k, v in d.items()]))

del atrdata['clean']
del atrdata['freq']
del atrdata['stemed']
del atrdata['sep']

def valuation_attrlength():
	for i, row in atrdata.iterrows():
		len = 0
		flen = 0
		elems = row['log_freq']		
		for key,value in elems.items():
			len = len + (value * value)
		flen = math.sqrt(len)
		atrdata.set_value(i,'tfidflen',flen)
	return

valuation_attrlength()
atrinfo = pd.merge(data, atrdata, how='left', on='product_uid')
atrresult = atrinfo.drop(['product_title','search_term', 'relevance','titlesrch','log_freq_x','tfidflen_x'], axis=1)

def valuation_atrsrch():
	for i, row in atrresult.iterrows():
		val = 0
		elems = row['log_freq_y']	
		srchelems = row['srchtmlog_freq']	
		doclength = row['tfidflen_y']
		querylength = row['tflen']
		for key,value in srchelems.items():
			if(not (pd.isnull(elems))):
				if key in elems:
					val = val + (value * elems.get(key))
		querydoc = val /(doclength * querylength)
		atrresult.set_value(i,'atrsrch',querydoc)
	return

valuation_atrsrch()

itmresult = pd.merge(data, descresult, on='id')
result = pd.merge(itmresult, atrresult, how='left', on='id')


finalresult = result.drop(['product_uid_x','product_title', 'search_term','log_freq','tflen_x','tfidflen','srchtmlog_freq_x','product_uid_y', 'tflen_y','srchtmlog_freq_y','log_freq_y_x','tfidflen_y_x','product_uid','tflen','srchtmlog_freq','name','value','log_freq_y_y','tfidflen_y_y'], axis=1)

#calculate the cosine similarity of the test search term with the product title, description and attributes

test_size = len(testdata)
testdata['sep'] = testdata.product_title.apply(lambda rec: rec.replace(',',' '))
testdata['titleclean'] = testdata.sep.apply(lambda rec: re.sub("[\s]", " ", rec.lower().strip()).split())
testdata['titlestemed'] = testdata.titleclean.apply(lambda words: [stemmer.stem(x) for x in words])
testdata['titlefreq'] = testdata.titlestemed.apply(lambda words: Counter(words))
testdata['titlelog_freq'] = testdata.titlefreq.apply(lambda d: dict([(k,math.log(v) + 1) for k, v in d.items()]))

del testdata['titleclean']
del testdata['titlestemed']
del testdata['titlefreq']
del testdata['sep']

testdata['testsrchtmclean'] = testdata.search_term.apply(lambda rec: re.sub("[\s]", " ", rec.lower().strip()).split())
testdata['testsrchtmstemed'] = testdata.testsrchtmclean.apply(lambda words: [stemmer.stem(x) for x in words])
testdata['testsrchtmfreq'] = testdata.testsrchtmstemed.apply(lambda words: Counter(words))
testdata['testsrchtmlog_freq'] = testdata.testsrchtmfreq.apply(lambda d: dict([(k,math.log(v) + 1) for k, v in d.items()]))

del testdata['testsrchtmclean']
del testdata['testsrchtmstemed']
del testdata['testsrchtmfreq']

testdata['titlelflen'] = 0.0
testdata['srchlflen'] = 0.0

def valuation_title_length():
	for i, row in testdata.iterrows():
		len = 0
		srchlen = 0
		flen = 0
		srchflen = 0
		elems = row['titlelog_freq']	
		srchelems = row['testsrchtmlog_freq']	
		for key,value in elems.items():
			len = len + (value * value)
		flen = math.sqrt(len)
		for key,value in srchelems.items():
			srchlen = srchlen + (value * value)
		srchflen = math.sqrt(srchlen)
		testdata.set_value(i,'titlelflen',flen)
		testdata.set_value(i,'srchlflen',srchflen)
	return

valuation_title_length()

def valuation_test_titlesrch():
	for i, row in testdata.iterrows():
		val = 0
		elems = row['titlelog_freq']	
		srchelems = row['testsrchtmlog_freq']	
		doclength = row['titlelflen']
		querylength = row['srchlflen']
		for key,value in srchelems.items():
			if key in elems:
				val = val + (value * elems.get(key))
		querydoc = val /(doclength * querylength)
		testdata.set_value(i,'titlesrch',querydoc)
	return

valuation_test_titlesrch()

del testdata['titlelflen']
del testdata['titlelog_freq']

testtitledesc = pd.merge(testdata, descdata, on='product_uid')


def valuation_test_descsrch():
	for i, row in testtitledesc.iterrows():
		val = 0
		elems = row['log_freq']	
		srchelems = row['testsrchtmlog_freq']	
		doclength = row['tfidflen']
		querylength = row['srchlflen']
		for key,value in srchelems.items():
			if key in elems:
				val = val + (value * elems.get(key))
		querydoc = val /(doclength * querylength)
		testtitledesc.set_value(i,'descsrch',querydoc)
	return

valuation_test_descsrch()


testtitledescatr = pd.merge(testtitledesc, atrdata, how='left', on='product_uid')

def valuation_test_attrsrch():
	for i, row in testtitledescatr.iterrows():
		val = 0
		elems = row['log_freq_y']	
		srchelems = row['testsrchtmlog_freq']	
		doclength = row['tfidflen_y']
		querylength = row['srchlflen']
		for key,value in srchelems.items():
			if(not (pd.isnull(elems))):
				if key in elems:
					val = val + (value * elems.get(key))
		querydoc = val /(doclength * querylength)
		testtitledescatr.set_value(i,'attrsrch',querydoc)
	return

valuation_test_attrsrch()

# Group the training data based on the relevance it belongs to. According to the training data, we have 13 different relevance classes.
# For each group, calculate the avg of the calculated cosine similarity of train search term and product title, product description and product attributes

resultlist.append(['id','relevance'])

title = [0,0,0,0,0,0,0,0,0,0,0,0,0]
desc = [0,0,0,0,0,0,0,0,0,0,0,0,0]
attr = [0,0,0,0,0,0,0,0,0,0,0,0,0]
count = [0,0,0,0,0,0,0,0,0,0,0,0,0]


for i, row in finalresult.iterrows():

	if ((row['relevance']) == 1):
		title[0] = title[0] + row['titlesrch']
		desc[0] = desc[0] + row['descsrch']
		if (not (pd.isnull(row['atrsrch']))):
			attr[0] = attr[0] + row['atrsrch']
		count[0] = count[0] + 1
	elif ((row['relevance']) == 1.25):
		title[1] = title[1] + row['titlesrch']
		desc[1] = desc[1] + row['descsrch']
		if (not (pd.isnull(row['atrsrch']))):
			attr[1] = attr[1] + row['atrsrch']
		count[1] = count[1] + 1
	elif ((row['relevance']) == 1.33):
		title[2] = title[2] + row['titlesrch']
		desc[2] = desc[2] + row['descsrch']
		if (not (pd.isnull(row['atrsrch']))):
			attr[2] = attr[2] + row['atrsrch']
		count[2] = count[2] + 1
	elif ((row['relevance']) == 1.5):
		title[3] = title[3] + row['titlesrch']
		desc[3] = desc[3] + row['descsrch']
		if (not (pd.isnull(row['atrsrch']))):
			attr[3] = attr[3] + row['atrsrch']
		count[3] = count[3] + 1
	elif ((row['relevance']) == 1.67):
		title[4] = title[4] + row['titlesrch']
		desc[4] = desc[4] + row['descsrch']
		if (not (pd.isnull(row['atrsrch']))):
			attr[4] = attr[4] + row['atrsrch']
		count[4] = count[4] + 1
	elif ((row['relevance']) == 1.75):
		title[5] = title[5] + row['titlesrch']
		desc[5] = desc[5] + row['descsrch']
		if (not (pd.isnull(row['atrsrch']))):
			attr[5] = attr[5] + row['atrsrch']
		count[5] = count[5] + 1
	elif ((row['relevance']) == 2):
		title[6] = title[6] + row['titlesrch']
		desc[6] = desc[6] + row['descsrch']
		if (not (pd.isnull(row['atrsrch']))):
			attr[6] = attr[6] + row['atrsrch']
		count[6] = count[6] + 1
	elif ((row['relevance']) == 2.25):
		title[7] = title[7] + row['titlesrch']
		desc[7] = desc[7] + row['descsrch']
		if (not (pd.isnull(row['atrsrch']))):
			attr[7] = attr[7] + row['atrsrch']
		count[7] = count[7] + 1
	elif ((row['relevance']) == 2.33):
		title[8] = title[8] + row['titlesrch']
		desc[8] = desc[8] + row['descsrch']
		if (not (pd.isnull(row['atrsrch']))):
			attr[8] = attr[8] + row['atrsrch']
		count[8] = count[8] + 1
	elif ((row['relevance']) == 2.5):
		title[9] = title[9] + row['titlesrch']
		desc[9] = desc[9] + row['descsrch']
		if (not (pd.isnull(row['atrsrch']))):
			attr[9] = attr[9] + row['atrsrch']
		count[9] = count[9] + 1
	elif ((row['relevance']) == 2.67):
		title[10] = title[10] + row['titlesrch']
		desc[10] = desc[10] + row['descsrch']
		if (not (pd.isnull(row['atrsrch']))):
			attr[10] = attr[10] + row['atrsrch']
		count[10] = count[10] + 1
	elif ((row['relevance']) == 2.75):
		title[11] = title[11] + row['titlesrch']
		desc[11] = desc[11] + row['descsrch']
		if (not (pd.isnull(row['atrsrch']))):
			attr[11] = attr[11] + row['atrsrch']
		count[11] = count[11] + 1
	else:
		title[12] = title[12] + row['titlesrch']
		desc[12] = desc[12] + row['descsrch']
		if (not (pd.isnull(row['atrsrch']))):
			attr[12] = attr[12] + row['atrsrch']
		count[12] = count[12] + 1

avgt = [title[0]/count[0],title[1]/count[1],title[2]/count[2],title[3]/count[3],title[4]/count[4],title[5]/count[5],title[6]/count[6],title[7]/count[7],title[8]/count[8],title[9]/count[9],title[10]/count[10],title[11]/count[11],title[12]/count[12]]
avgd = [desc[0]/count[0],desc[1]/count[1],desc[2]/count[2],desc[3]/count[3],desc[4]/count[4],desc[5]/count[5],desc[6]/count[6],desc[7]/count[7],desc[8]/count[8],desc[9]/count[9],desc[10]/count[10],desc[11]/count[11],desc[12]/count[12]]
avga = [attr[0]/count[0],attr[1]/count[1],attr[2]/count[2],attr[3]/count[3],attr[4]/count[4],attr[5]/count[5],attr[6]/count[6],attr[7]/count[7],attr[8]/count[8],attr[9]/count[9],attr[10]/count[10],attr[11]/count[11],attr[12]/count[12]]

classes = ['1','1.25','1.33','1.5','1.67','1.75','2','2.25','2.33','2.5','2.67','2.75','3']
fdict = dict()

# For each test data pair of cosine similarity of search term and product title, product description and product attributes, find the nearest neighbour 
# to the calculated average points of different relevance. Assign the relevance of the point which is closest to the test cosine similarity pair.

def find_distance(val1,val2,val3,i):
	fdict[i] = math.sqrt(math.pow((val1 - avgt[i]), 2) + math.pow((val2 - avgd[i]), 2) + math.pow((val3 - avga[i]), 2))
	
for i, row in testtitledescatr.iterrows():

	val1 = row['titlesrch']	
	val2 = row['descsrch']
	if (not (pd.isnull(row['attrsrch']))):
		val3 = row['attrsrch']
	else:
		val3 = 99

	for i in range(0,13):
		find_distance(val1,val2,val3,i)


	minm = fdict.get(0)
	rel = classes[1]
	
	for key,value in fdict.items():
		if(value < minm):
			minm = value
			rel = classes[key]

	resultlist.append([str(row['id']),str(rel)])


with open('results.csv', 'w',newline='') as outcsv:
	writer = csv.writer(outcsv,delimiter=',')
	writer.writerows(resultlist)
