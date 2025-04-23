"""
TSAI Explainability via attribution map


Attribution Map Interpretation
These attribution maps visualize which parts of the ECG signals were most influential in the model's decisions. The heat-colored dots overlaid on the black signal line indicate the degree of importance:

Red/orange/yellow areas: These brighter colors represent regions where the model is paying the most attention - the features most influential in making predictions about cardiac scar tissue
Dark/black areas: These represent regions the model considers less important for its prediction

Example Key Patterns Observed

QRS Complex Focus: The model appears to be paying particularly close attention to the sharp peaks and troughs in the ECG, especially the QRS complexes (the large upward/downward deflections). This makes clinical sense since structural abnormalities like scars often affect these ventricular depolarization patterns.
ST Segment Attention: In several samples, there's significant attention (yellow coloration) in the areas following the QRS complex, corresponding to the ST segment. Changes in this region can indicate myocardial damage.
Pattern Consistency: There's consistency in how the model directs attention across different samples. For example:

Sample 3571 shows high attribution at each major deflection
Sample 376 focuses strongly on the initial spike and later complexes
Sample 3046 shows a regular pattern of attribution at each cardiac cycle


Selective Focus: The model isn't just universally highlighting all peaks - it's selectively emphasizing specific features in each ECG, suggesting it has learned meaningful patterns related to the scar target.

Clinical Relevance
This visualization suggests the model is focusing on clinically relevant features. Cardiac scars typically manifest in ECGs as changes in QRS morphology, abnormal Q waves, and ST-T wave changes - precisely the areas receiving high attribution scores.
The attribution maps provide explainability for the model's decisions, potentially helping validate its clinical application for detecting cardiac scar tissue from ECG signals.

"""

from tsai.inference import load_learner
from tsai.models.explainability import get_acts_and_grads, get_attribution_map
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch


def visualize_attribution_maps(model_path, X, y=None, sample_indices=None, n_samples=5, 
                             figsize=(10, 8), module_name='backbone', apply_relu=True, outpath=None):
    """
    Visualize attribution maps for ECG time series data using a trained tsai model.
    
    This function loads a trained model, computes attribution maps to highlight which
    parts of the input signal most influenced the model's prediction, and displays
    the results in a comprehensive visualization.
    
    Parameters:
    -----------
    model_path : str
        Path to the trained tsai model (.pkl file)
    X : np.ndarray
        Input data in the shape expected by the model (n_samples, n_channels, seq_length)
    y : np.ndarray, optional
        True labels for the input data, if available
    sample_indices : list or np.ndarray, optional
        Indices of specific samples to visualize. If None, randomly selects n_samples
    n_samples : int, default=5
        Number of samples to visualize if sample_indices is None
    figsize : tuple, default=(20, 15)
        Figure size for the visualization
    module_name : str, default='backbone'
        Name of the model module to extract activations and gradients from
    apply_relu : bool, default=True
        Whether to apply ReLU to attribution maps to highlight positive contributions
    outpath : str, optional 
        Path to save the generated figure. If None, the figure is not saved.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the visualizations
    """

    
    # Load the model
    try:
        model = load_learner(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        raise ValueError(f"Failed to load model: {str(e)}")
    
    # Get the module
    module = None
    for name, mod in model.model.named_modules():
        if module_name in name:
            module = mod
            print(f"Found module: {name}")
            break
    
    if module is None:
        # Try to find any available modules and suggest them
        available_modules = [name for name, _ in model.model.named_modules() if len(name) > 0]
        modules_str = ", ".join(available_modules[:10]) + ("..." if len(available_modules) > 10 else "")
        raise ValueError(f"Module '{module_name}' not found. Available modules include: {modules_str}")
    
    # Select sample indices
    if sample_indices is None:
        if n_samples > len(X):
            n_samples = len(X)
            print(f"Warning: n_samples was greater than the number of available samples. Set to {n_samples}")
        sample_indices = random.sample(range(len(X)), n_samples)
    
    # Make predictions for the selected samples
    X_selected = X[sample_indices]
    
    # Convert to torch tensor if it's a numpy array
    if isinstance(X_selected, np.ndarray):
        X_tensor = torch.tensor(X_selected, dtype=torch.float32)
    else:
        X_tensor = X_selected  # Assume it's already a tensor
    
    if y is not None:
        y_selected = y[sample_indices]
        if not isinstance(y_selected, torch.Tensor):
            y_tensor = torch.tensor(y_selected, dtype=torch.long)
        else:
            y_tensor = y_selected
    else:
        # Get predictions from the model
        with torch.no_grad():
            probs, _, preds = model.get_X_preds(X_tensor)
        y_selected = preds
    
    # Create figure
    fig, axes = plt.subplots(len(sample_indices), 2, figsize=figsize)
    if len(sample_indices) == 1:
        axes = np.array([axes])  # Ensure axes is always 2D
    
    # Process each sample
    for i, (idx, ax_row) in enumerate(zip(sample_indices, axes)):
        print(i, idx, ax_row)
        x_sample_np = X_selected[i:i+1]  # Keep batch dimension, numpy array
        x_sample = X_tensor[i:i+1]  # Tensor for model
        y_true = y_selected[i] if y is not None else None
        
        # Ensure model is in eval mode
        model.model.eval()
        
        try:
            # Convert y_true to tensor if needed
            if y_true is not None and not isinstance(y_true, torch.Tensor):
                y_tensor = torch.tensor([y_true], dtype=torch.long)
            else:
                y_tensor = y_true
            # Get activations and gradients
            acts, grads = get_acts_and_grads(model.model, [module], x_sample, y=None)
            
            # Compute attribution map
            attribution_map = get_attribution_map(model.model, [module], x_sample, y=None, 
                                                apply_relu=apply_relu)
            

                
            # Convert attribution map to numpy if it's a tensor
            if isinstance(attribution_map, torch.Tensor):
                attribution_map = attribution_map.cpu().detach().numpy()
                
            # Plot original signal
            ax_row[0].plot(x_sample_np[0, 0, :], 'b-', linewidth=1.5)
            ax_row[0].set_title(f"Sample {idx} - Original ECG Signal")
            ax_row[0].set_xlabel("Time")
            ax_row[0].set_ylabel("Amplitude")
            
            # Plot attribution map
            signal_len = x_sample_np.shape[-1]
            time = np.arange(signal_len)
            signal = x_sample_np[0, 0, :]

            # Handle different attribution map dimensions
            if attribution_map.ndim == 3:
                att_map = attribution_map[0, 0, :signal_len]
            elif attribution_map.ndim == 2:
                att_map = attribution_map[0, :signal_len]
            else:
                att_map = attribution_map[:signal_len]
            
            
            # Create a colormap based on attribution values
            ax_row[1].scatter(time, signal, c=att_map, cmap='hot', 
                            s=10, alpha=0.8, marker='o')
            ax_row[1].plot(signal, 'b-', linewidth=0.5, alpha=0.3)
            ax_row[1].set_title(f"Sample {idx} - Attribution Map")
            ax_row[1].set_xlabel("Time")
            ax_row[1].set_ylabel("Amplitude")
            
            # Get prediction information
            if hasattr(model, 'dls') and hasattr(model.dls, 'vocab'):
                vocab = model.dls.vocab
                class_names = vocab
                class_idx = int(preds[i]) if 'preds' in locals() else y_true
                if class_idx < len(class_names):
                    class_name = class_names[class_idx]
                    ax_row[0].set_title(f"Sample {idx} - Original ECG Signal (Class: {class_name})")
        
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            ax_row[0].text(0.5, 0.5, f"Error: {str(e)}", 
                          horizontalalignment='center', verticalalignment='center',
                          transform=ax_row[0].transAxes, color='red')
            ax_row[1].text(0.5, 0.5, "Failed to generate attribution map", 
                          horizontalalignment='center', verticalalignment='center',
                          transform=ax_row[1].transAxes, color='red')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle("Attribution Maps for ECG Signals", fontsize=16)
    outpath = os.path.join(outpath, 'attribution_maps.png') if outpath else None
    if outpath:
        plt.savefig(outpath, bbox_inches='tight')
        print(f"Figure saved to {outpath}")
    return fig


def analyze_ecg_with_attribution(model_path, X, y=None, point_numbers=None, wavefront=None, 
                                 target=None, module_name='backbone', apply_relu=True,
                                 top_k_timepoints=10, class_names=None, outpath=None):
    """
    Comprehensive analysis of ECG signals using attribution maps to identify key features
    influencing model predictions.
    
    Parameters:
    -----------
    model_path : str
        Path to the trained tsai model (.pkl file)
    X : np.ndarray
        Input data in the shape expected by the model (n_samples, n_channels, seq_length)
    y : np.ndarray, optional
        True labels for the input data
    point_numbers : list, optional
        Point numbers corresponding to each sample for identification
    wavefront : str, optional
        Wavefront type ('LVp', 'RVp', or 'SR')
    target : str, optional
        Target type for classification
    module_name : str, default='backbone'
        Name of the model module to extract activations and gradients from
    apply_relu : bool, default=True
        Whether to apply ReLU to attribution maps
    top_k_timepoints : int, default=10
        Number of most influential timepoints to identify
    class_names : list, optional
        Names of the output classes
    outpath : str, optional
        Path to save the generated figure. If None, the figure is not saved.
        
    Returns:
    --------
    dict
        Dictionary containing analysis results including:
        - Figure with attribution maps
        - Most influential timepoints for each sample
        - Average importance across samples
    """
    
    # Load the model
    model = load_learner(model_path)
    print(f"Analyzing {len(X)} ECG samples using model from {model_path}")
    
    # Extract model information from filename if available
    if wavefront is None or target is None:
        model_filename = model_path.split('/')[-1]
        parts = model_filename.split('_')
        if len(parts) >= 3:
            if target is None and len(parts) > 1:
                target = parts[1]
            if wavefront is None and len(parts) > 2:
                wavefront = parts[2]
    
    # Get the specified module
    module = None
    for name, mod in model.model.named_modules():
        if module_name in name:
            module = mod
            print(f"Using module: {name}")
            break
    
    if module is None:
        raise ValueError(f"Module '{module_name}' not found in model")
    
    # Convert X to torch tensor if it's not already
    if isinstance(X, np.ndarray):
        X_tensor = torch.tensor(X, dtype=torch.float32)
    else:
        X_tensor = X  # Assume it's already a tensor
    
    # Make predictions
    with torch.no_grad():
        probs, _, preds = model.get_X_preds(X_tensor)
    
    # Convert predictions to numpy if they're tensors
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    
    # Prepare for results
    n_samples = len(X)
    results = {
        'predictions': preds,
        'probabilities': probs,
        'top_timepoints': [],
        'attribution_maps': [],
        'point_numbers': point_numbers if point_numbers is not None else list(range(n_samples))
    }
    
    # Create figure for visualization
    n_rows = min(10, n_samples)  # Show maximum 10 samples in the figure
    fig, axes = plt.subplots(n_rows, 2, figsize=(20, 3 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    
    # Process all samples
    all_attributions = []
    
    # Ensure model is in eval mode
    model.model.eval()
    
    for i in range(n_samples):
        try:
            x_sample_np = X[i:i+1]  # Keep batch dimension for numpy
            x_sample = X_tensor[i:i+1]  # Keep batch dimension for tensor
            y_pred = preds[i] if len(preds) > 0 else None
            
            # Convert y_pred to tensor if needed
            if y_pred is not None and not isinstance(y_pred, torch.Tensor):
                y_tensor = torch.tensor([y_pred], dtype=torch.long)
            else:
                y_tensor = None  # Use model's prediction
            
            # Get attributions
            acts, grads = get_acts_and_grads(model.model, [module], x_sample, y=None)
            
            attribution_map = get_attribution_map(model.model, [module], x_sample, y=None, 
                                                apply_relu=apply_relu)
            
            # Convert attribution map to numpy if it's a tensor
            if isinstance(attribution_map, torch.Tensor):
                attribution_map = attribution_map.cpu().detach().numpy()
            
            # Handle different attribution map dimensions
            if attribution_map.ndim == 3:
                sample_attribution = attribution_map[0, 0, :]
            elif attribution_map.ndim == 2:
                sample_attribution = attribution_map[0, :]
            else:
                sample_attribution = attribution_map[:]
            all_attributions.append(sample_attribution)
            results['attribution_maps'].append(sample_attribution)
            
            # Find top-k most influential timepoints
            signal_len = min(x_sample_np.shape[-1], len(sample_attribution))
            top_k_indices = np.argsort(sample_attribution[:signal_len])[-top_k_timepoints:]
            results['top_timepoints'].append({
                'indices': top_k_indices.tolist(),
                'values': sample_attribution[top_k_indices].tolist()
            })
            
            # Plot for the first n_rows samples
            if i < n_rows:
                signal = x_sample_np[0, 0, :]
                time = np.arange(len(signal))
                
                # Original signal
                axes[i, 0].plot(signal, 'b-', linewidth=1.0)
                axes[i, 0].set_title(f"Sample {results['point_numbers'][i]} - Original Signal")
                
                # Attribution map
                mapped_attribution = sample_attribution[:len(signal)]
                scatter = axes[i, 1].scatter(time, signal, c=mapped_attribution, cmap='hot', 
                                            s=10, alpha=0.7)
                axes[i, 1].plot(signal, 'b-', linewidth=0.5, alpha=0.3)
                
                # Mark top important timepoints
                axes[i, 1].plot(top_k_indices, signal[top_k_indices], 'g*', markersize=10)
                
                # Add class information if available
                if class_names is not None and y_pred is not None:
                    class_name = class_names[int(y_pred)] if int(y_pred) < len(class_names) else f"Class {int(y_pred)}"
                    prob_value = probs[i, int(y_pred)] if probs.shape[1] > 1 else probs[i]
                    title = f"Sample {results['point_numbers'][i]} - Attribution Map - {class_name} ({prob_value:.2f})"
                else:
                    title = f"Sample {results['point_numbers'][i]} - Attribution Map"
                
                axes[i, 1].set_title(title)
                
        except Exception as e:
            print(f"Error processing sample {i}: {str(e)}")
            if i < n_rows:
                axes[i, 0].text(0.5, 0.5, f"Error: {str(e)}", 
                              horizontalalignment='center', verticalalignment='center',
                              transform=axes[i, 0].transAxes, color='red')
                axes[i, 1].text(0.5, 0.5, "Failed to generate attribution map", 
                              horizontalalignment='center', verticalalignment='center',
                              transform=axes[i, 1].transAxes, color='red')
            results['top_timepoints'].append({'indices': [], 'values': []})
            results['attribution_maps'].append(None)
    
    # Compute average attribution across all samples with valid attributions
    valid_attributions = [attr for attr in all_attributions if attr is not None]
    if len(valid_attributions) > 0:
        # Ensure all attributions have the same length for averaging
        min_length = min(map(len, valid_attributions))
        all_attributions_trimmed = [attr[:min_length] for attr in valid_attributions]
        avg_attribution = np.mean(all_attributions_trimmed, axis=0)
        results['average_attribution'] = avg_attribution.tolist()
        
        # Find top-k most influential timepoints on average
        top_k_avg_indices = np.argsort(avg_attribution)[-top_k_timepoints:]
        results['top_average_timepoints'] = {
            'indices': top_k_avg_indices.tolist(),
            'values': avg_attribution[top_k_avg_indices].tolist()
        }
    
    # Add colorbar if we have at least one successful plot
    if n_rows > 0 and 'scatter' in locals():
        plt.colorbar(scatter, ax=axes[:, 1], label="Attribution Strength")
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle("Attribution Maps for ECG Signals", fontsize=16)
    if outpath:
        plt.savefig(os.path.join(outpath, 'attribution_maps.png'), bbox_inches='tight')
        print(f"Figure saved to {outpath}/attribution_maps.png")
    results['figure'] = fig
    
    # Print summary
    print(f"\nAnalysis completed for {n_samples} ECG samples")
    if wavefront and target:
        print(f"Model target: {target}, Wavefront: {wavefront}")
    
    print("\nMost influential timepoints across all samples:")
    if 'top_average_timepoints' in results:
        for idx, val in zip(results['top_average_timepoints']['indices'], 
                           results['top_average_timepoints']['values']):
            print(f"  Timepoint {idx}: Attribution value {val:.4f}")
    
    return results


def analyze_fat_composition_attribution(inpath, out_dir=None):
    """
    Analyzes attribution maps for EndoIntra_SCARComposition in SR wavefront
    from the adiposity ECG project data.
    
    Parameters:
    -----------
    inpath : str
        Path to the FAT SUB PROJECT data folder
    out_dir : str, optional
        Directory to save results (defaults to inpath/attribution_maps)
    
    Returns:
    --------
    dict
        Analysis results containing attribution maps and key timepoints
    """
    from classifier_tsai import TSai

    # Setup output directory
    if out_dir is None:
        out_dir = os.path.join(inpath, 'attribution_maps')
    os.makedirs(out_dir, exist_ok=True)
    
    # Initialize TSai with the adiposity data
    target = 'EndoIntra_SCARComposition'
    wavefront = 'SR'
    tsai = TSai(
        inpath=inpath, 
        fname_csv='publishable_model_data_AdiposityElectrogram_master_merged_raw_unipolar_clean.parquet', 
        load_train_data=True, 
        target_type='fat'
    )
    
    # Convert data for the selected wavefront and target
    X, y = tsai.df_to_ts(wavefront, target)
    
    # We'll use a subset for the attribution analysis (first 10 samples)
    X_subset = X[:10]
    y_subset = y[:10]
    
    # Path to the trained model
    model_path = os.path.join(inpath, 'models', f'clf_raw_unipolar_{target}_{wavefront}_120epochs.pkl')
    
    # Get point numbers for better labeling
    point_numbers = tsai.df[tsai.df['WaveFront'] == wavefront]['Point_Number'].unique()[:10]
    
    # Define class names based on the dataset
    scar_composition_classes = [f'Composition-{i}' for i in range(3)]  
    
    # Run attribution map analysis
    results = analyze_ecg_with_attribution(
        model_path=model_path,
        X=X_subset,
        y=y_subset,
        point_numbers=point_numbers,
        wavefront=wavefront,
        target=target,
        class_names=scar_composition_classes,
        top_k_timepoints=20
    )
    
    # Save the attribution map figure
    results['figure'].savefig(
        os.path.join(out_dir, f'attribution_map_{wavefront}_{target}.png'), 
        dpi=300, 
        bbox_inches='tight'
    )
    
    # Create a summary of important timepoints
    print(f"Analysis for {target} with {wavefront} wavefront:")
    print("Top 5 most influential timepoints:")
    for i, (idx, val) in enumerate(zip(
        results['top_average_timepoints']['indices'], 
        results['top_average_timepoints']['values']
    )):
        if i >= 5:
            break
        print(f"  Timepoint {idx}: Attribution value {val:.4f}")
    
    return results