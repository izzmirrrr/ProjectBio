import streamlit as st
import torch
from torchvision import models, transforms, datasets
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import random

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model(model_name):
    if model_name == "ResNet50":
        model = models.resnet50(pretrained=False)  
        model.fc = nn.Linear(model.fc.in_features, 4)  
        model.load_state_dict(torch.load("FixmodelRessNet50.pth", map_location=device))  
    elif model_name == "EfficientNet-B0":
        model = models.efficientnet_b0(pretrained=False)  
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)  
        model.load_state_dict(torch.load("EffiencyNet.pth", map_location=device))
    else:
        raise ValueError("Model not recognized.")
    model = model.to(device)
    model.eval() 
    return model

@st.cache_resource
def get_validation_loader():
    validation_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root="C:\\Users\\user\\Desktop\\OriginalDataset", transform=validation_pipeline)

    dataset_size = len(full_dataset)
    val_size = int(0.2 * dataset_size) 
    train_size = dataset_size - val_size  

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    return val_loader, val_dataset

# Calculate accuracy
def calculate_accuracy(model, dataloader):
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = correct / total * 100
    return accuracy, all_labels, all_predictions

# Streamlit UI
st.title("Alzheimer Disease Classification")
st.write("Choose the model:")

# Model selection
model_option = st.selectbox("Choose Model", ["ResNet50", "EfficientNet-B0"])

if st.button("Predict"):
    with st.spinner("Loading model and data..."):
        model = load_trained_model(model_option)
        val_loader, val_dataset = get_validation_loader()
        accuracy, all_labels, all_predictions = calculate_accuracy(model, val_loader)

    # Display accuracy
    st.success("Prediction Complete!")
    st.write(f"Model: **{model_option}**")
    st.write(f"Accuracy: **{accuracy:.2f}%**")

    # Display confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    class_labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap='viridis', ax=ax)
    st.pyplot(fig)

    # Display classification report
    report = classification_report(all_labels, all_predictions, target_names=class_labels, output_dict=False)
    st.text("Classification Report:")
    st.text(report)

    # Show sample predictions
    st.subheader("Sample Predictions")
    random_idx = random.sample(range(len(val_dataset)), 4)  # Pick 4 random indices
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for i, idx in enumerate(random_idx):
        img, true_label = val_dataset[idx]
        img = img.unsqueeze(0).to(device)  # Add batch dimension and move to device

        with torch.no_grad():
            outputs = model(img)
            _, predicted_label = torch.max(outputs, 1)

        # Undo normalization
        inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        img = inv_normalize(img[0].cpu())

        # Plot image
        ax = axes[i // 2, i % 2]
        ax.imshow(img.permute(1, 2, 0).numpy())  # Convert from (C, H, W) to (H, W, C)
        ax.axis('off')

        true_label_name = val_dataset.dataset.classes[true_label]
        predicted_label_name = val_dataset.dataset.classes[predicted_label.item()]
        ax.set_title(f"True: {true_label_name}\nPred: {predicted_label_name}")

    plt.tight_layout()
    st.pyplot(fig)
