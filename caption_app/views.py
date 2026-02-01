from django.shortcuts import render
from django.core.files.storage import default_storage
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load model ONCE (important for performance)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def generate_caption(image_path):
    """
    Takes image path and returns caption text
    """
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def upload_image(request):
    caption = None
    image_url = None

    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        path = default_storage.save('uploads/' + image.name, image)
        image_url = default_storage.url(path)

        # Generate caption
        caption = generate_caption(default_storage.path(path))

    return render(request, 'upload.html', {
        'caption': caption,
        'image_url': image_url
    })
