import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler

def demonstrate_optics_on_mall_csv(csv_path):
    """
    Apply OPTICS clustering on mall customers dataset
    Features: Age, Annual Income, Spending Score
    """
    
    print("="*60)
    print("OPTICS CLUSTERING DEMONSTRATION - MALL CUSTOMERS")
    print("="*60)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print("\n1. RAW DATA:")
    print(df.head(10).to_string(index=False))
    
    # Extract features
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
    print("\n2. FEATURE MATRIX:")
    print(f"Shape: {X.shape}")
    print(X)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\n3. STANDARDIZED FEATURES (Mean=0, Std=1):")
    print(X_scaled)
    print(f"Mean: {X_scaled.mean(axis=0)}")
    print(f"Std: {X_scaled.std(axis=0)}")
    
    # Apply OPTICS algorithm
    print("\n4. APPLYING OPTICS ALGORITHM:")
    optics_model = OPTICS(
        min_samples=2,
        xi=0.05,
        min_cluster_size=0.1,
        metric='euclidean'
    )
    clusters = optics_model.fit_predict(X_scaled)
    
    # Add cluster labels to DataFrame
    df['Cluster'] = clusters
    print("\n5. CLUSTERING RESULTS:")
    print(df.to_string(index=False))
    
    # Reachability distances
    reachability = optics_model.reachability_[optics_model.ordering_]
    
    # Visualize results
    visualize_optics_results(df, X_scaled, reachability, optics_model.ordering_)
    
    return df, optics_model

def visualize_optics_results(df, X_scaled, reachability, ordering):
    """
    Visualize OPTICS clustering results
    """
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Age vs Annual Income
    ax1 = fig.add_subplot(131)
    scatter = ax1.scatter(df['Age'], df['Annual Income (k$)'], 
                          c=df['Cluster'], cmap='viridis', s=200, edgecolors='black', linewidths=2)
    for idx, row in df.iterrows():
        ax1.annotate(row['CustomerID'], (row['Age'], row['Annual Income (k$)']), 
                     fontsize=10, ha='center', va='center', color='white', weight='bold')
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Annual Income (k$)')
    ax1.set_title('OPTICS: Age vs Income')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Cluster')
    
    # Plot 2: Annual Income vs Spending Score
    ax2 = fig.add_subplot(132)
    scatter2 = ax2.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], 
                           c=df['Cluster'], cmap='viridis', s=200, edgecolors='black', linewidths=2)
    for idx, row in df.iterrows():
        ax2.annotate(row['CustomerID'], (row['Annual Income (k$)'], row['Spending Score (1-100)']), 
                     fontsize=10, ha='center', va='center', color='white', weight='bold')
    ax2.set_xlabel('Annual Income (k$)')
    ax2.set_ylabel('Spending Score')
    ax2.set_title('OPTICS: Income vs Spending')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Cluster')
    
    # Plot 3: Reachability plot
    ax3 = fig.add_subplot(133)
    colors = ['g', 'r', 'b', 'y', 'c']
    for i, (order_idx, reach_dist) in enumerate(zip(ordering, reachability)):
        cluster = df.iloc[order_idx]['Cluster']
        color = colors[int(cluster) % len(colors)] if cluster != -1 else 'k'
        ax3.bar(i, reach_dist, color=color, width=0.8, edgecolor='black')
    ax3.set_xlabel('Order')
    ax3.set_ylabel('Reachability Distance')
    ax3.set_title('OPTICS Reachability Plot')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('optics_mall_customers.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved: optics_mall_customers.png")
    plt.show()

# Run on your mall CSV
if __name__ == "__main__":
    csv_path = "mall_customers.csv"  # Make sure the CSV is in the same folder
    df_result, model = demonstrate_optics_on_mall_csv(csv_path)
