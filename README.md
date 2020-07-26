# HR-Analytics-webapp-django-deploying-ML-models

This webapp will let you download prediction files for the HR Analytics Challenge taken from Analytics Vidhya. The templates folder has the index.html file and in the views.py
you will find the main logic of requests from the frontend to the django.

This is the link to the website: https://hr-analytics-webapp.herokuapp.com/

If you do not see the static files(like images) that is because on heroku you cannot deploy django static files directly. You have to use a third party library such as
whitenoise to do so. I have not yet implemented that.

