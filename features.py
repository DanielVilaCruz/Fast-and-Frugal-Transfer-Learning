import torch
import tqdm

def extract_features(dataloader, backbone, device):
    backbone.eval()
    backbone.to(device)
        
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Extracting features"):
            images, labels = batch[:2]
            images = images.to(device, non_blocking=True) 
            features = backbone(images)

            if isinstance(features, tuple):
                features = features[0]
            
            if len(features.shape) > 2:
                features = features.flatten(start_dim=1)
            
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
    
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)
