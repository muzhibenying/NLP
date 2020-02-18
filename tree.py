import xml.etree.ElementTree as ET 
import spacy
import numpy as np
import torch

nlp = spacy.load('en_core_web_sm')
xmlname = '/Users/xiaoying/Downloads/NLP/NLP数据/news/articles-training-byarticle-20181122.xml'
labelname = '/Users/xiaoying/Downloads/NLP/NLP数据/ground-truth/ground-truth-training-byarticle-20181122.xml'
#xmlname = '/Users/xiaoying/Downloads/NLP/NLP数据/articles-validation-bypublisher-20181122.xml'
#labelname = '/Users/xiaoying/Downloads/NLP/NLP数据/ground-truth-validation-bypublisher-20181122.xml'
articletree = ET.parse(xmlname)
labeltree = ET.parse(labelname)
articledict = {}
labeldict = {}
iddict = {}
articleroot = articletree.getroot()
labelroot = labeltree.getroot()
i = 0
for article in articleroot:
    id = article.attrib['id']
    print(id)
    wordvector = []
    for para in article:
        text = ET.tostring(para, method = 'text', encoding = 'unicode')
        with nlp.disable_pipes():
            if nlp(text).vector.size != 0:
                wordvector.append(nlp(text).vector)
    articledict[id] = wordvector
    iddict[i] = id
    i += 1
for label in labelroot:
    id = label.attrib['id']
    truth = label.attrib['hyperpartisan']
    labeldict[id] = truth
dictionary = {'sample': articledict, 'label': labeldict, 'id': iddict}
torch.save(dictionary, 'byarticleval.pki')
print(iddict)
