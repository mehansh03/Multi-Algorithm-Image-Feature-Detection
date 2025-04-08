# image_processing/models.py
from django.db import models

class ImageProcessingResult(models.Model):
    name = models.CharField(max_length=100, default="unnamed")
    sift_image = models.ImageField(upload_to='processed/', null=True)
    ransac_image = models.ImageField(upload_to='processed/', null=True, blank=True)
    harris_image = models.ImageField(upload_to='processed/', null=True)
    

    class Meta:
        app_label = 'image_processing'  # Explicitly set app_label

    def __str__(self):
        return f"Result {self.id}"