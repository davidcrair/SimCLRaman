import ramanspy as rp
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.graph_objects as go

# load training and testing datasets
X_train, y_train = rp.datasets.bacteria("train", folder="bacteria_data")
X_train = X_train.spectral_data
X_test, y_test = rp.datasets.bacteria("test", folder="bacteria_data")
X_test = X_test.spectral_data

# load the names of the species and antibiotics corresponding to the 30 classes
y_labels, antibiotics_labels = rp.datasets.bacteria("labels")

label_bacteria_name_dict = {i: label for i, label in enumerate(y_labels)}
label_antibiotics_label_dict = {i: label for i, label in enumerate(antibiotics_labels)}

antibiotic_color_dict = {
    'Vancomycin': '#D7F1F6',
    'Ceftriaxone': '#C7D0F2',
    'Penicillin': '#DBA5E7',
    'Daptomycin': '#CC7AA5',
    'Meropenem': '#B26F57',
    'Ciprofloxacin': '#776E22',
    'TZP': '#376E24',
    'Caspofungin': '#122632'
}

label_color_dict = {
    i: antibiotic_color_dict[antibiotics_labels[i]]
    for i in range(len(label_bacteria_name_dict))
}

# function to evaluate model and get confusion matrix
def get_confusion_matrix(model, test_loader, device, num_classes, mapping_dict=None):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    if mapping_dict is not None:
        all_labels = [mapping_dict[label] for label in all_labels]
        all_preds = [mapping_dict[pred] for pred in all_preds]

    # print accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc:.4f}")
    
    # calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    conf_matrix_percent = confusion_matrix(all_labels, all_preds, normalize='true') * 100
    
    return conf_matrix, conf_matrix_percent

# function to plot confusion matrix
def plot_confusion_matrix(conf_matrix, conf_matrix_percent, class_names, zmin=0, zmax=100, text_mode='count'):
    # heatmap
    heatmap = go.Heatmap(
        z=conf_matrix_percent,
        x=class_names,
        y=class_names,
        colorscale='deep',
        zmin=zmin,
        zmax=zmax,
        text=conf_matrix if text_mode == 'count' else conf_matrix_percent,
        texttemplate="%{text:.0f}",
        textfont={"size": 10},
        hovertemplate='Ground Truth: %{y}<br>Predicted: %{x}<br>Percentage: %{z:.1f}%<extra></extra>',
        showscale=True
    )

    print(class_names)
    
    fig = go.Figure(data=[heatmap])
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="True Class",
        height=1000,
        xaxis=dict(
            side="top",
            tickangle=-45
        ),
        yaxis=dict(
            autorange="reversed" # make it be top-left to bottom-right diagonal to match nature paper
        ),
        # shapes=diagonal_shape,
        template='plotly_white'
    )
    
    fig.show()