from PIL import Image
from torchvision import transforms


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])


def preprocess_image(image: Image.Image):

    image = image.convert("L")

    tensor = transform(image)
    

    # tensor is already 0–1 because ToTensor() does this
    tensor = tensor.view(1, 784)
    return tensor