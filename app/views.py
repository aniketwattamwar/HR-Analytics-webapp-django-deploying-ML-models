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

valid_requests = {
    'gNB': ('gNB.sav', 'guassianNB.csv'),
    'multiNB': ('classifier_multi_NB.sav', 'multi_NB.csv'),
    'rf': ('multi_NB.csv', 'rf.csv')
}


def home(request):
    return  render(request,"index.html")


def models(request):
    length = os.path.getsize(valid_requests[request.POST])
    dispos = 'attachment; filename=%s' % f'{valid_requests[request.POST][1]}'

    pd.DataFrame(
        pickle.load(
            open(valid_requests[request.POST][0])
        ).predict(test_data_preprocessed)
    ).to_csv(valid_requests[request.POST])

    response = HttpResponse(open(
        valid_requests[request.POST], 'rb').read(),
        content_type="text/csv"
    )

    response['Content-Length'] = length
    response['Content-Disposition'] = dispos

    return response