import cv2
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
from django.conf import settings
from .models import ImageProcessingResult

@csrf_exempt
def upload_images(request):
    if request.method == 'POST':
        try:
            fs = FileSystemStorage(location=settings.MEDIA_ROOT)
            image1_file = request.FILES.get('image1')
            image2_file = request.FILES.get('image2')

            if not image1_file or not image2_file:
                return JsonResponse({'error': 'Please upload both images.'}, status=400)

            if not (image1_file.content_type.startswith('image/') and image2_file.content_type.startswith('image/')):
                return JsonResponse({'error': 'Please upload only image files.'}, status=400)

            # Save uploaded files
            image1_path = fs.save('uploads/image1.jpg', image1_file)
            image2_path = fs.save('uploads/image2.jpg', image2_file)
            img1 = cv2.imread(os.path.join(settings.MEDIA_ROOT, image1_path), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(os.path.join(settings.MEDIA_ROOT, image2_path), cv2.IMREAD_GRAYSCALE)
            img1_color = cv2.imread(os.path.join(settings.MEDIA_ROOT, image1_path))
            img2_color = cv2.imread(os.path.join(settings.MEDIA_ROOT, image2_path))

            if img1 is None or img2 is None:
                return JsonResponse({'error': 'Failed to process one or both images.'}, status=500)

            # Preprocess images for better detection
            img1 = cv2.equalizeHist(img1)
            img2 = cv2.equalizeHist(img2)

            # SIFT with improved parameters
            sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.03, edgeThreshold=10, sigma=1.6)
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)

            if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
                return JsonResponse({'error': 'Insufficient keypoints detected.'}, status=400)

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

            sift_output = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:50], None, 
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            sift_path = os.path.join(settings.MEDIA_ROOT, 'processed/sift_matches.jpg')
            os.makedirs(os.path.dirname(sift_path), exist_ok=True)
            cv2.imwrite(sift_path, sift_output)

            # RANSAC with fallback
            ransac_image = None
            if len(good_matches) >= 4:  # Minimum for homography
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0, maxIters=2000, confidence=0.995)
                matches_mask = mask.ravel().tolist()

                h, w = img1.shape
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                img2_color = cv2.polylines(img2_color, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

                ransac_output = cv2.drawMatches(img1_color, kp1, img2_color, kp2, good_matches, None,
                                              matchesMask=matches_mask, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                ransac_path = os.path.join(settings.MEDIA_ROOT, 'processed/ransac_transform.jpg')
                cv2.imwrite(ransac_path, ransac_output)
                ransac_image = 'processed/ransac_transform.jpg'
            else:
                # Fallback: Save SIFT output as RANSAC result with a note
                cv2.putText(sift_output, "RANSAC Skipped: Too Few Matches", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ransac_path = os.path.join(settings.MEDIA_ROOT, 'processed/ransac_transform.jpg')
                cv2.imwrite(ransac_path, sift_output)
                ransac_image = 'processed/ransac_transform.jpg'

            # Harris Corner Detection with dilation
            harris = cv2.cornerHarris(img1, blockSize=2, ksize=3, k=0.04)
            harris = cv2.dilate(harris, None)  # Enhance visibility
            img1_harris = img1_color.copy()
            threshold = 0.01 * harris.max()
            img1_harris[harris > threshold] = [0, 0, 255]  # Mark corners in red
            harris_path = os.path.join(settings.MEDIA_ROOT, 'processed/harris_corners.jpg')
            cv2.imwrite(harris_path, img1_harris)

            # Save to model
            result = ImageProcessingResult(
                name=f"Result_{ImageProcessingResult.objects.count() + 1}",  # Unique name
                sift_image='processed/sift_matches.jpg',
                ransac_image=ransac_image,
                harris_image='processed/harris_corners.jpg'
            )
            result.save()

            return render(request, 'image_processing/result.html', {
                'sift_image': f'{settings.MEDIA_URL}processed/sift_matches.jpg',
                'ransac_image': f'{settings.MEDIA_URL}{ransac_image}',
                'harris_image': f'{settings.MEDIA_URL}processed/harris_corners.jpg',
            })

        except Exception as e:
            return JsonResponse({'error': f'An error occurred: {str(e)}'}, status=500)

    return render(request, 'image_processing/upload.html')

def results_list(request):
    results = ImageProcessingResult.objects.all()
    return render(request, 'image_processing/results_list.html', {'results': results})

