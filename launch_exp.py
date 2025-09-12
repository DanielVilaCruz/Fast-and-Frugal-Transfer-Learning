import os
import time
import copy
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
from codecarbon import EmissionsTracker
from collections import defaultdict
import psutil
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Local imports
from data.data_loaders import get_data_pcam, get_data_breakhis, get_data_brain_cancer
from train_func import weighted_training, test_model
from models import IntermediateFeatureExtractor, get_head, get_full_model
from features import extract_features

# Configure logging
os.environ['CODECARBON_LOG_LEVEL'] = 'WARNING'
logging.getLogger("codecarbon").setLevel(logging.WARNING)

class ExperimentConfig:
    """Centralized configuration for the experiment"""
    def __init__(self):
        # Experiment parameters
        self.method = "ours"  # Options: "ours", "fine-tune"
        self.dataset = "pcam"  # Options: "cifar10", "cifar100", "places", "mri", "breakhis", "pcam"
        self.base_backbone = "densenet121"  # Options: "resnet18", "resnet50", "mobilenet_v3_large", "vit_b_16"
        self.model_name = self.base_backbone
        
        # Training parameters
        self.num_epochs = 10
        self.train_batch_size = 16 
        self.lr = 0.001
        self.patience = 10
        self.amplify_factor = 10.0
        self.n_real_samples = 0
        self.k_percent = 85
        self.num_runs = 1  # Number of training runs to perform
        self.unfreeze_layers = 2  # Number of layers to unfreeze for fine-tuning
        
        # Technical settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.quant = None  # torch.float16 
        self.img_size = 224
        self.use_mixed_precision = False
        self.simple_head = False

class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.setup_experiment()
        
    def setup_experiment(self):
        """Initialize data loaders, model and tracker"""
        self.load_data()
        self.init_feature_extractor()
        self.init_tracker()

    def load_data(self):
        
        """Load dataset based on configuration"""
        if self.config.dataset == "pcam":
            self.train_loader, self.val_loader, self.test_loader, self.class_names = get_data_pcam(
                max_dataset_size=self.config.n_real_samples,
                data_root='./data/pcam',
                train_batch_size=self.config.train_batch_size,
                test_batch_size=self.config.train_batch_size,
                data_size=(self.config.img_size, self.config.img_size)
            )

        elif self.config.dataset == "breakhis":
            self.train_loader, self.val_loader, self.test_loader, self.class_names = get_data_breakhis(
                      max_dataset_size=self.config.n_real_samples,
                      data_root='./data/break_his/BreaKHis_v1/BreaKHis_v1/histology_slides',
                      train_batch_size=256, 
                      test_batch_size=256,
                      data_size=(self.config.img_size, self.config.img_size),
                      train_ratio=0.7,
                      val_ratio=0.15,
                      mode="binary",
                      magnification="40X",
                      random_seed=42, 
                      quant=None)
            
        elif self.config.dataset == "brain_cancer":
            self.train_loader, self.val_loader, self.test_loader, self.class_names = get_data_brain_cancer(
                max_dataset_size=self.config.n_real_samples,
                data_root='./data/brain_cancer', 
                train_batch_size=self.config.train_batch_size,
                test_batch_size=self.config.train_batch_size,
                data_size=(self.config.img_size, self.config.img_size),
                train_ratio=0.7,
                val_ratio=0.15,
                random_seed=42
            )
       
    def init_feature_extractor(self):
        """Initialize feature extraction model"""
        self.feature_extractor = IntermediateFeatureExtractor(
            last_layer=self.config.last_layer, 
            base_model=self.config.base_backbone, 
            quant=self.config.quant
        ).eval()

    def count_layers(self, model):
        total_layers = 0
        batchnorm_layers = 0

        for module in model.modules():
            # Skip the root module (the model itself)
            if module is model:
                continue
            
            # Check if this is a leaf module (not a container)
            is_leaf = True
            for child in module.children():
                is_leaf = False
                break
            
            if is_leaf:
                total_layers += 1
                # Count BatchNorm layers
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    batchnorm_layers += 1
            
        return total_layers, batchnorm_layers

        
    def init_tracker(self):
        sample_size = self.config.n_real_samples if self.config.n_real_samples > 0 else "All"
        self.experiment_name = (
            f"{self.config.model_name}, {self.config.method}, {self.config.dataset}, "
            f"samples={sample_size}, k%={self.config.k_percent}, "
            f"lr={self.config.lr}, amplify_factor={self.config.amplify_factor}, "
            f"num_runs={self.config.num_runs}"
        )
        
        os.makedirs("exps", exist_ok=True)
        self.tracker = EmissionsTracker(
            project_name=self.experiment_name,
            output_file=f"exps/{self.experiment_name}.csv"
        )
    
    def adapt_batchnorm(self, model, data_loader, max_samples=2000, stop_threshold=0.01):
        model.to(self.config.device).train()
        running_mean = None
        
        print("Adaptively updating BN stats...")
        with torch.no_grad():
            for batch in tqdm(data_loader, total=min(len(data_loader), max_samples//data_loader.batch_size)):
                x = batch[0].to(self.config.device)
                _ = model(x)
                
                new_mean = torch.cat([
                    bn.running_mean.view(1, -1) 
                    for bn in model.modules() 
                    if isinstance(bn, nn.BatchNorm2d)
                ], dim=1)
                
                if running_mean is not None:
                    delta = torch.norm(new_mean - running_mean) / (torch.norm(running_mean) + 1e-6)
                    if delta < stop_threshold:
                        break
                running_mean = new_mean
        model.eval()
        return model
     
    def extract_features(self, perform_batchnorm=True):
        start_time = time.time()
        
        if not "vit" in self.config.base_backbone and perform_batchnorm:
            self.feature_extractor = self.adapt_batchnorm(
                self.feature_extractor, 
                self.train_loader
            )
        
        # Feature extraction
        train_feats, train_labels = extract_features(self.train_loader, self.feature_extractor, self.config.device)
        val_feats, val_labels = extract_features(self.val_loader, self.feature_extractor, self.config.device)
        test_feats, test_labels = extract_features(self.test_loader, self.feature_extractor, self.config.device)

        # Create datasets
        self.train_dataset = TensorDataset(train_feats, train_labels.squeeze())
        self.val_dataset = TensorDataset(val_feats, val_labels.squeeze())
        self.test_dataset = TensorDataset(test_feats, test_labels.squeeze())
        
        # Create dataloaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.train_batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.config.train_batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config.train_batch_size, shuffle=False)
        
        return time.time() - start_time
    
    def run_ours_method(self):
        all_results = []
        extraction_time = 0
        
        for run in range(self.config.num_runs):
            print(f"\n=== Starting run {run + 1}/{self.config.num_runs} ===")
            
            memory_stats = {
                "gpu_peak": 0,
                "cpu_peak": 0,
                "gpu_per_batch": [],
                "cpu_per_batch": []
            }
            
            # Feature extraction (only needed once if not changing data)
            if run == 0:
                extraction_time = self.extract_features()

            # Initialize classifier head
            train_feats, train_labels = next(iter(self.train_loader))
            self.head = get_head(train_feats, train_labels, simple=self.config.simple_head)

            # Training
            start_time = time.time()
            training_time, trained_model, _, _, _ = weighted_training(
                self.train_loader, 
                self.head, 
                num_epochs=self.config.num_epochs,
                lr=self.config.lr,
                patience=self.config.patience,
                amplify_factor=self.config.amplify_factor,
                val_dataloader=self.val_loader,
                quant=self.config.quant,
            )
            total_training_time = time.time() - start_time

            gpu_mem = self.get_gpu_memory()
            cpu_mem = self.get_cpu_memory()
            memory_stats["gpu_peak"] = max(memory_stats["gpu_peak"], gpu_mem["allocated"])
            memory_stats["cpu_peak"] = max(memory_stats["cpu_peak"], cpu_mem)
            memory_stats["gpu_per_batch"].append(gpu_mem["allocated"])
            memory_stats["cpu_per_batch"].append(cpu_mem)
            
            # Evaluation
            test_results = test_model(self.test_loader, trained_model)
            
            all_results.append({
                "metrics": test_results,
                "times": {
                    "extraction": extraction_time if run == 0 else 0,
                    "training": total_training_time
                },
                "memory": memory_stats
            })
        
        return self.aggregate_results(all_results)
    
    def run_finetune_method(self):
        all_results = []
        
        for run in range(self.config.num_runs):
            print(f"\n=== Starting run {run + 1}/{self.config.num_runs} ===")

            memory_stats = {
                "gpu_peak": 0,
                "cpu_peak": 0,
                "gpu_per_batch": [],
                "cpu_per_batch": []
            }

            # Initialize full model
            model = get_full_model(self.config.model_name, pretrained=True)
            
            num_classes = len(self.class_names) if hasattr(self, 'class_names') else \
                        len(torch.unique(self.train_loader.dataset.targets))
            
            # Modify final layer
            if "dense" in self.config.model_name:
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Linear(num_ftrs, num_classes)
            elif hasattr(model, 'fc'):  # For ResNet models
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, num_classes)
            elif "vit" in self.config.model_name:
                num_ftrs = model.heads.head.in_features
                model.heads.head = nn.Linear(num_ftrs, num_classes)
            else:
                model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
            
            model = model.to(self.config.device)

            input_res = (3, config.img_size, config.img_size)

            # Freeze layers
            for param in model.parameters():
                param.requires_grad = False
            
            # Unfreeze the classifier/fc layer
            if hasattr(model, 'classifier'):
                for param in model.classifier.parameters():
                    param.requires_grad = True
            elif hasattr(model, 'fc'):
                for param in model.fc.parameters():
                    param.requires_grad = True

            # Unfreeze last few layers
            def unfreeze_last_layers(model, num_layers=2):
                layers = list(model.children())
                for layer in layers[-(num_layers + 1):-1]:
                    for param in layer.parameters():
                        param.requires_grad = True
            unfreeze_last_layers(model, num_layers=self.config.unfreeze_layers)

            # Initialize training components
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
            scheduler = OneCycleLR(optimizer, max_lr=5e-3, 
                                steps_per_epoch=len(self.train_loader), 
                                epochs=self.config.num_epochs)
            
            if self.config.use_mixed_precision:
                scaler = torch.cuda.amp.GradScaler()
            else:
                scaler = None

            # Training 
            best_val_loss = float('inf')
            best_model_stmodel_state = None
            epochs_no_improve = 0
            start_time = time.time()
            
            for epoch in range(self.config.num_epochs):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                epoch_start = time.time()
                
                for inputs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}"):
                    inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)
                    
                    optimizer.zero_grad()
                    
                    if self.config.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    
                    scheduler.step()
                    
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                # Validation 
                val_loss = 0.0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in self.val_loader:
                        inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)
                        outputs = model(inputs)
                        val_loss += criterion(outputs, labels).item()
                
                avg_val_loss = val_loss / len(self.val_loader)
                
                # Early stopping 
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.config.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Load best model
            if best_model_state:
                model.load_state_dict(best_model_state)
            
            gpu_mem = self.get_gpu_memory()
            cpu_mem = self.get_cpu_memory()
            memory_stats["gpu_peak"] = max(memory_stats["gpu_peak"], gpu_mem["allocated"])
            memory_stats["cpu_peak"] = max(memory_stats["cpu_peak"], cpu_mem)
            memory_stats["gpu_per_batch"].append(gpu_mem["allocated"])
            memory_stats["cpu_per_batch"].append(cpu_mem)

            # Evaluation
            model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in self.test_loader:
                    inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            
            accuracy = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='macro', zero_division=0
            )
            
            all_results.append({
                "metrics": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                },
                "times": {
                    "training": time.time() - start_time,
                    "extraction": 0  # No feature extraction in fine-tuning
                },
                "memory": memory_stats
            })
        
        return self.aggregate_results(all_results)
    
    def aggregate_results(self, all_results):
        """Aggregate results from multiple runs"""
        aggregated = {
            "metrics": defaultdict(list),
            "times": defaultdict(list)
        }
        
        # Collect all metrics and times
        for result in all_results:
            for metric, value in result["metrics"].items():
                aggregated["metrics"][metric].append(value)
            for time_key, time_value in result["times"].items():
                aggregated["times"][time_key].append(time_value)
        
        # Calculate mean and std
        final_results = {
            "metrics": {},
            "times": {},
            "all_runs": all_results
        }
        
        for metric, values in aggregated["metrics"].items():
            final_results["metrics"][f"{metric}_mean"] = np.mean(values)
            final_results["metrics"][f"{metric}_std"] = np.std(values)
        
        for time_key, time_values in aggregated["times"].items():
            final_results["times"][f"{time_key}_mean"] = np.mean(time_values)
            final_results["times"][f"{time_key}_std"] = np.std(time_values)
            final_results["times"][f"{time_key}_total"] = np.sum(time_values)
        
        return final_results
    
    def run(self):
        self.tracker.start()
        start_time = time.time()
        
        if self.config.method == "ours":
            results = self.run_ours_method()
        elif self.config.method == "fine-tune":
            results = self.run_finetune_method()
        
        # Collect results
        emissions_data = self.tracker.stop()
        emissions_data = self.tracker.final_emissions_data
        total_time = time.time() - start_time
        
        # Print and save results
        self.print_results(results, emissions_data, total_time)
        self.append_experiment_to_json(results, emissions_data, total_time, output_path="results.json")
        self.save_results(results, emissions_data, total_time)
        
        return results
    
    def append_experiment_to_json(self, results, emissions_data, total_time, output_path="results.json"):

        def convert(o):
            if isinstance(o, (np.integer, np.int32, np.int64)):
                return int(o)
            elif isinstance(o, (np.floating, np.float32, np.float64)):
                return float(o)
            elif isinstance(o, np.ndarray):
                return o.tolist()
            elif isinstance(o, (list, tuple)):  
                return [convert(x) for x in o]
            elif isinstance(o, dict): 
                return {k: convert(v) for k, v in o.items()}
            else:
                return o
                
        results_converted = {k: convert(v) for k, v in results.items()}
        
        if hasattr(emissions_data, '__dict__'):
            emissions_dict = vars(emissions_data)
        else:
            emissions_dict = emissions_data
        emissions_converted = {k: convert(v) for k, v in emissions_dict.items()}
        
        exp_data = {
            "num_runs": convert(self.config.num_runs),
            "metrics": convert(results["metrics"]),
            "times": convert(results["times"]),
            "emissions": {
                "co2_kg": convert(emissions_data.emissions),
                "energy_kwh": convert(emissions_data.energy_consumed)
            },
            "total_duration_s": convert(total_time),
            "all_runs": convert(results["all_runs"])
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            with open(output_path, "r") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = {}
        else:
            existing_data = {}

        existing_data[self.experiment_name] = exp_data

        with open(output_path, "w") as f:
            json.dump(existing_data, f, indent=4)

    def save_results_to_json(self, results, emissions_data, total_time, output_path="results.json"):

        export_data = {
            "experiment_name": self.experiment_name,
            "num_runs": self.config.num_runs,
            "metrics": results["metrics"],
            "times": results["times"],
            "emissions": {
                "co2_kg": emissions_data.emissions,
                "energy_kwh": emissions_data.energy_consumed
            },
            "total_duration_s": total_time,
            "all_runs": results["all_runs"]
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=4)

        print(f"\nResults saved to {output_path.resolve()}")


    def print_results(self, results, emissions_data, total_time):
        print(f'\n=== Final Results for {self.experiment_name} ===')
        print(f'Number of runs: {self.config.num_runs}')
        print("\nMetrics:")
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            mean = results["metrics"][f"{metric}_mean"]
            std = results["metrics"][f"{metric}_std"]
            print(f'{metric.capitalize():<10}: {mean:.4f} ± {std:.4f}')
        
        print("\nTimes:")
        for time_key in  ['extraction', 'training']:
            
            print(f'Total Time ({time_key.capitalize()}) {results["times"][f"{time_key}_total"]:.2f} s')
            if time_key == "extraction":
                continue
            mean = results["times"][f"{time_key}_mean"]
            std = results["times"][f"{time_key}_std"]
            print(f'{time_key.capitalize() + " Time":<15}: {mean:.2f} ± {std:.2f} s')
        
        print("\nOther Statistics:")
        print(f'CO2 Emissions (kg): {emissions_data.emissions:.6f}')
        print(f'Energy Consumed (kWh): {emissions_data.energy_consumed:.6f}')
        print(f'Total Duration (s): {total_time:.2f}')
        
        print("\n=== Individual Run Results ===")
        for i, run in enumerate(results["all_runs"]):
            print(f"\nRun {i + 1}:")
            print(f"Accuracy: {run['metrics']['accuracy']:.4f}")
            print(f"Precision: {run['metrics']['precision']:.4f}")
            print(f"Recall: {run['metrics']['recall']:.4f}")
            print(f"F1: {run['metrics']['f1']:.4f}")
            print(f"Training Time: {run['times']['training']:.2f} s")

        print("\nMemory Usage:")
        print(f"Peak GPU Memory (MB): {results['all_runs'][0]['memory']['gpu_peak']:.2f}")
        print(f"Peak CPU RAM (MB): {results['all_runs'][0]['memory']['cpu_peak']:.2f}")
        print(f"GPU Memory per Batch (MB): {np.mean(results['all_runs'][0]['memory']['gpu_per_batch']):.2f} ± {np.std(results['all_runs'][0]['memory']['gpu_per_batch']):.2f}")
        print(f"CPU RAM per Batch (MB): {np.mean(results['all_runs'][0]['memory']['cpu_per_batch']):.2f} ± {np.std(results['all_runs'][0]['memory']['cpu_per_batch']):.2f}")

    def save_results(self, results, emissions_data, total_time):
        results_dict = {
            "Experiment": [self.experiment_name],
            "Number of Runs": [self.config.num_runs],
            "CO2 Emissions (kg)": [emissions_data.emissions],
            "Energy Consumed (kWh)": [emissions_data.energy_consumed],
            "Total Duration (s)": [total_time],
            "CPU Energy (kWh)": [emissions_data.cpu_energy],
            "GPU Energy (kWh)": [emissions_data.gpu_energy],
            "RAM Energy (kWh)": [emissions_data.ram_energy]
        }
        
        # Add metrics
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            results_dict[f"{metric}_mean"] = [results["metrics"][f"{metric}_mean"]]
            results_dict[f"{metric}_std"] = [results["metrics"][f"{metric}_std"]]
        
        # Add times
        for time_key in ['extraction', 'training']:
            results_dict[f"{time_key}_time_mean"] = [results["times"][f"{time_key}_mean"]]
            results_dict[f"{time_key}_time_std"] = [results["times"][f"{time_key}_std"]]
        
        # Add individual run results
        for i, run in enumerate(results["all_runs"]):
            results_dict[f"run_{i+1}_accuracy"] = [run["metrics"]["accuracy"]]
            results_dict[f"run_{i+1}_precision"] = [run["metrics"]["precision"]]
            results_dict[f"run_{i+1}_recall"] = [run["metrics"]["recall"]]
            results_dict[f"run_{i+1}_f1"] = [run["metrics"]["f1"]]
            results_dict[f"run_{i+1}_training_time"] = [run["times"]["training"]]
        
        results_df = pd.DataFrame(results_dict)
        
        os.makedirs("exps", exist_ok=True)
        results_df.to_csv(f"exps/{self.experiment_name}_results.csv", index=False)

    def get_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)     # MB
            return {"allocated": allocated, "reserved": reserved}
        return {"allocated": 0, "reserved": 0}

    def get_cpu_memory(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)  # MB

if __name__ == "__main__":

    experiment_configs = []

    with open("experiments.yaml", "r") as f:
        data = yaml.safe_load(f)
    experiment_configs = []

    for exp in data["experiments"]:
        config = ExperimentConfig()
        config.method = exp["method"]
        config.base_backbone = exp["base_backbone"]
        config.dataset = exp["dataset"]
        config.n_real_samples = exp["n_real_samples"]
        config.num_runs = exp["num_runs"]
        config.last_layer = exp.get("last_layer", 1)
        config.lr = exp.get("lr", 0.001)
        config.amplify_factor = exp.get("amplify_factor", 10.0)
        config.patience = exp.get("patience", 3)
        config.unfreeze_layers = exp.get("unfreeze_layers", 2)
        config.simple_head = exp.get("simple_head", False)

        experiment_configs.append(config)

    print(f"--- Starting {len(experiment_configs)} Experiments ---")
    for i, config in enumerate(experiment_configs):
        print(f"\nRunning Experiment {i+1}/{len(experiment_configs)}:")
        print(f"  Method: {config.method}")
        print(f"  Backbone: {config.base_backbone}")
        print(f"  Dataset: {config.dataset}")
        print(f"  Real Samples: {config.n_real_samples}")
        print(f"  Number of Runs: {config.num_runs}")
        
        runner = ExperimentRunner(config)
        runner.run()
    
    print("\n--- All Experiments Completed ---")