import base64
import io
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from .serializers import ImageUploadSerializer
import os
from PIL import Image
from detection.detection import Detect
from django.http import JsonResponse


class ImageUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file_serializer = ImageUploadSerializer(data=request.data)
        if file_serializer.is_valid():
            file_instance = file_serializer.save()
            file_path = file_instance.image.path
            detect = Detect()
            img, number_plate = detect.run(file_path)
            image_base64 = numpy_array_to_base64(img)
            return Response(
                {"number_plate": number_plate, "image_base64": image_base64},
                status=status.HTTP_201_CREATED,
            )
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


def numpy_array_to_base64(image_np):
    # Ensure the image is in uint8 format and RGB mode
    image_np = image_np.astype(np.uint8)
    if image_np.shape[-1] != 3:
        raise ValueError("Input should be a RGB image")

    # Create a PIL Image object from the numpy array
    image = Image.fromarray(image_np)

    # Save PIL Image to byte stream
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)

    # Encode byte stream to base64
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Construct data URL
    base64_image = f"data:image/png;base64,{encoded_image}"

    return base64_image
