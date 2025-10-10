import torch
from torchvision import transforms
from matplotlib import pyplot as plt
from torch import nn
from typing import List, Tuple
from PIL import Image
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def pred_and_plot_image(
        model:nn.Module,
        image_path:str,
        class_names:List[str],
        image_size:Tuple[int,int]=(224,224),
        transform:transforms = None,
        device : torch.device = device,

):
    image = Image.open(image_path)
    if transform is not None:
        image_transform = transforms
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        model.to(device)

        model.eval()
        with torch.inference_mode():
            transformed_image = image_transform(image).unsqueeze(0).to(device)
            target_image_pred = model(transformed_image).to(device)
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
        target_image_pred_labels = torch.argmax(target_image_pred_probs, dim=1)
        plt.figure()
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Predictions: {class_names[target_image_pred_labels]}|Probabilities: {target_image_pred_probs.max():.3f}")
