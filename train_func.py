import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import time
import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def weighted_cross_entropy(preds, targets, weights):
    log_probs = torch.nn.functional.log_softmax(preds, dim=1)
    loss = -log_probs[range(len(targets)), targets] * weights
    return loss.mean()

def weighted_training(dataloader, original_model, num_epochs=10, lr=1e-3, patience=3,
          val_dataloader=None, amplify_factor=2.0, quant=False, config=None):
    model = copy.deepcopy(original_model)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    start_time = time.time()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        dtype = torch.float32

        for x, y in tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            if quant:
                x, y = x.to(device=device, dtype=dtype), y.to(device=device)
            else:
                x, y = x.to(device), y.to(device)

            preds = model(x)

            # Select samples per batch and amplify their gradients             
            with torch.no_grad():
                probs = torch.nn.functional.softmax(preds, dim=1)
                top2_vals, _ = probs.topk(2, dim=1)
                margins = top2_vals[:, 0] - top2_vals[:, 1]  # low margin = more confusing

            # Select k lowest-margin (most ambiguous) samples per batch
            k = int(0.2 * x.size(0))  # Amplify 20% of samples
            _, topk_indices = torch.topk(-margins, k)

            # Initialize weights as 1, amplify selected samples
            weights = torch.ones_like(y, dtype=torch.float32, device=device)
            weights[topk_indices] *= amplify_factor  

            loss = weighted_cross_entropy(preds, y, weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (preds.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        epoch_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        # Validation loss for early stopping
        if val_dataloader:
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for x_val, y_val in tqdm.tqdm(val_dataloader):
                    x_val, y_val = x_val.to(device=device, dtype=dtype), y_val.to(device)
                    val_preds = model(x_val)
                    val_loss += criterion(val_preds, y_val).item()
            model.train()

            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Validation Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    end_time = time.time()
    training_time = end_time - start_time

    if best_model_state:
        model.load_state_dict(best_model_state)

    print(f"Training time: {training_time:.2f}s")
    return training_time, model, total_loss, epoch_accuracy, epoch

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            if images.shape[1] == 1:  # If grayscale
                images = images.repeat(1, 3, 1, 1)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total

def test_model(test_loader, my_approach_trained_model):
    my_approach_trained_model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for samples, labels in tqdm.tqdm(test_loader, desc='Testing'):
            samples, labels = samples.to(device), labels.to(device)

            if samples.shape[1] == 1:  # If grayscale
                samples = samples.repeat(1, 3, 1, 1)
            
            # Forward pass through the trained model
            outputs = my_approach_trained_model(samples)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
      
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)

    print(f"Test Accuracy: {acc:.2f}%")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Sensitivity/Recall (macro): {recall:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

