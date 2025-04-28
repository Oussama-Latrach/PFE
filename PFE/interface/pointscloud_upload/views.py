import os
from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse, HttpResponse
from .dgcnn_inference import DGCNNInference


def upload_page(request):
    if request.method == 'POST' and request.FILES.get('pointcloud'):
        # Sauvegarde du fichier uploadé
        uploaded_file = request.FILES['pointcloud']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        uploaded_path = fs.path(filename)

        # Stockage du chemin dans la session
        request.session['uploaded_file_path'] = uploaded_path
        request.session['original_filename'] = filename

        return render(request, 'pointscloud_upload/upload.html', {
            'file_uploaded': True,
            'filename': filename
        })

    return render(request, 'pointscloud_upload/upload.html')


def launch_classification(request):
    if 'uploaded_file_path' not in request.session:
        return redirect('upload_page')

    uploaded_path = request.session['uploaded_file_path']
    original_filename = request.session.get('original_filename', '')

    try:
        inferencer = DGCNNInference()
        classified_data = inferencer.predict(uploaded_path)
        results = inferencer.save_results(classified_data, uploaded_path)

        # Nettoyage
        if os.path.exists(uploaded_path):
            os.remove(uploaded_path)

        # Préparation des données pour le template
        context = {
            'original_filename': original_filename,
            'results': results,
            'img_url': os.path.join(settings.MEDIA_URL, 'classification_results',
                                    f"{results['base_name']}_2d.png"),
            'ply_url': os.path.join(settings.MEDIA_URL, 'classification_results',
                                    f"{results['base_name']}_classified.ply"),
            'stats': results['stats']
        }

        return render(request, 'pointscloud_upload/results.html', context)

    except Exception as e:
        print(f"Error during classification: {e}")
        return redirect('upload_page')


def download_file(request, file_type):
    if 'results' not in request.session:
        return redirect('upload_page')

    results = request.session['results']
    file_path = results.get(f'{file_type}_path')

    if file_path and os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/octet-stream")
            response['Content-Disposition'] = f'inline; filename={os.path.basename(file_path)}'
            return response

    return redirect('upload_page')