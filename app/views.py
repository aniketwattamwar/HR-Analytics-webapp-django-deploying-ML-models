from django.shortcuts import render
from django.http import HttpResponse
from wsgiref.util import FileWrapper
# Create your views here.
import os
import pickle
import pandas as pd

test_data_preprocessed = pd.read_csv('test_preprocessed.csv')
test_data_preprocessed = test_data_preprocessed.drop(['Unnamed: 0'],axis =1)
test_data_preprocessed = test_data_preprocessed.iloc[:,:].values

def home(request):
    return  render(request,"index.html")

def models(request):
        
    if 'gNB' in request.POST:
        gaussian = pickle.load(open('gNB.sav','rb'))
        y_pred = gaussian.predict(test_data_preprocessed)
        output = pd.DataFrame(y_pred)
        output.to_csv('gaussianNB.csv')
        
        filename = 'gaussianNB.csv'
        response = HttpResponse(open(filename, 'rb').read(), content_type='text/csv')               
        response['Content-Length'] = os.path.getsize(filename)
        response['Content-Disposition'] = 'attachment; filename=%s' % 'gaussianNB.csv'
        return response
    
    if 'multiNB' in request.POST:
        multi = pickle.load(open('classifier_multi_NB.sav','rb'))
        y_pred = multi.predict(test_data_preprocessed)
        output = pd.DataFrame(y_pred)
        output.to_csv('multi_NB.csv')
        
        filename = 'multi_NB.csv'
        response = HttpResponse(open(filename, 'rb').read(), content_type='text/csv')                
        response['Content-Length'] = os.path.getsize(filename)
        response['Content-Disposition'] = 'attachment; filename=%s' % 'multi_NB.csv'
        return response
    
    if 'rf' in request.POST:
        rf = pickle.load(open('random_forest.sav','rb'))
        y_pred = rf.predict(test_data_preprocessed)
        output = pd.DataFrame(y_pred)
        output.to_csv('rf.csv')
        
        filename = 'rf.csv'
        response = HttpResponse(open(filename, 'rb').read(), content_type='text/csv')             
        response['Content-Length'] = os.path.getsize(filename)
        response['Content-Disposition'] = 'attachment; filename=%s' % 'rf.csv'
        return response

