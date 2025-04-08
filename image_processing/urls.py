from django.urls import path
from . import views


app_name = 'image_processing'
urlpatterns = [
    path('', views.upload_images, name='upload_images'),
    path('results/', views.results_list, name='results_list'),
]

