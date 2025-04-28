from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_page, name='upload_page'),
    path('launch_classification/', views.launch_classification, name='launch_classification'),
    path('download/<str:file_type>/', views.download_file, name='download_file'),
]