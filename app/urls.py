# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 20:58:52 2020

@author: hp
"""

from django.urls import path

from . import views

urlpatterns = [
        
        path('',views.home, name = 'home'),
        path('download',views.models)
        ]

