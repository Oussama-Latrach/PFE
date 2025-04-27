# pointscloud_upload/views.py
from django.shortcuts import render

def upload_page(request):
    return render(request, 'pointscloud_upload/upload.html')
