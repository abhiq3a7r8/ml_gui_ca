"""
OPTICS Clustering Demonstration on 5 Students
Demonstrates the OPTICS algorithm step-by-step
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
import pandas as pd

def demonstrate_optics_5_students():
    """
    Demonstrate OPTICS algorithm on a group of 5 students
    Features: Age, Annual Income, Spending Score
    """
    
    print("="*60)
    print("OPTICS CLUSTERING DEMONSTRATION - 5 STUDENTS")
    print("="*60)
    
    # Sample data: 5 students with Age, Income, Spending Score
    students_data = {
    'Student_ID': ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12'],
    'Age': [22, 23, 24, 21, 23, 45, 50, 47, 48, 49, 30, 31],
    'Annual_Income': [25, 28, 26, 27, 29, 70, 75, 72, 73, 74, 40, 42],
    'Spending_Score': [80, 82, 78, 85, 79, 35, 40, 38, 36, 37, 60, 58]}

    
    df = pd.DataFrame(students_data)
    print("\n1. STUDENT DATA:")
    print(df.to_string(index=False))
    
    # Extract features
    X = df[['Age', 'Annual_Income', 'Spending_Score']].values
    
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
    print("   Parameters:")
    print("   - min_samples: 2 (minimum points in neighborhood)")
    print("   - xi: 0.05 (steepness threshold)")
    print("   - min_cluster_size: 0.1 (minimum cluster size)")
    
    optics_model = OPTICS(
        min_samples=2,
        xi=0.05,
        min_cluster_size=0.1,
        metric='euclidean'
    )
    
    clusters = optics_model.fit_predict(X_scaled)
    
    print("\n5. CLUSTERING RESULTS:")
    df['Cluster'] = clusters
    print(df.to_string(index=False))
    
    # Reachability distances
    reachability = optics_model.reachability_[optics_model.ordering_]
    
    print("\n6. REACHABILITY PLOT DATA:")
    print(f"Ordering: {optics_model.ordering_}")
    print(f"Reachability distances: {reachability}")
    
    # Calculate statistics
    unique_clusters = np.unique(clusters)
    print("\n7. CLUSTER STATISTICS:")
    for cluster_id in unique_clusters:
        cluster_points = df[df['Cluster'] == cluster_id]
        if cluster_id == -1:
            print(f"\n   Noise Points: {len(cluster_points)}")
        else:
            print(f"\n   Cluster {cluster_id}:")
            print(f"   - Number of points: {len(cluster_points)}")
            print(f"   - Avg Age: {cluster_points['Age'].mean():.2f}")
            print(f"   - Avg Income: {cluster_points['Annual_Income'].mean():.2f}")
            print(f"   - Avg Spending: {cluster_points['Spending_Score'].mean():.2f}")
    
    # Visualize results
    visualize_optics_results(df, X_scaled, reachability, optics_model.ordering_)
    
    return df, optics_model

def visualize_optics_results(df, X_scaled, reachability, ordering):
    """
    Visualize OPTICS clustering results
    """
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: 2D scatter (Age vs Income)
    ax1 = fig.add_subplot(131)
    scatter = ax1.scatter(
        df['Age'], 
        df['Annual_Income'], 
        c=df['Cluster'], 
        cmap='viridis', 
        s=200,
        edgecolors='black',
        linewidths=2
    )
    
    for idx, row in df.iterrows():
        ax1.annotate(
            row['Student_ID'], 
            (row['Age'], row['Annual_Income']),
            fontsize=10,
            ha='center',
            va='center',
            color='white',
            weight='bold'
        )
    
    ax1.set_xlabel('Age', fontsize=12)
    ax1.set_ylabel('Annual Income (k$)', fontsize=12)
    ax1.set_title('OPTICS Clustering: Age vs Income', fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Cluster')
    
    # Plot 2: 2D scatter (Income vs Spending)
    ax2 = fig.add_subplot(132)
    scatter2 = ax2.scatter(
        df['Annual_Income'], 
        df['Spending_Score'], 
        c=df['Cluster'], 
        cmap='viridis', 
        s=200,
        edgecolors='black',
        linewidths=2
    )
    
    for idx, row in df.iterrows():
        ax2.annotate(
            row['Student_ID'], 
            (row['Annual_Income'], row['Spending_Score']),
            fontsize=10,
            ha='center',
            va='center',
            color='white',
            weight='bold'
        )
    
    ax2.set_xlabel('Annual Income (k$)', fontsize=12)
    ax2.set_ylabel('Spending Score', fontsize=12)
    ax2.set_title('OPTICS Clustering: Income vs Spending', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Cluster')
    
    # Plot 3: Reachability plot
    ax3 = fig.add_subplot(133)
    colors = ['g', 'r', 'b', 'y', 'c']
    
    for i, (order_idx, reach_dist) in enumerate(zip(ordering, reachability)):
        cluster = df.iloc[order_idx]['Cluster']
        color = colors[int(cluster) % len(colors)] if cluster != -1 else 'k'
        ax3.bar(i, reach_dist, color=color, width=0.8, edgecolor='black')
    
    ax3.set_xlabel('Order', fontsize=12)
    ax3.set_ylabel('Reachability Distance', fontsize=12)
    ax3.set_title('OPTICS Reachability Plot', fontsize=14, weight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('optics_5_students_demo.png', dpi=300, bbox_inches='tight')
    print("\n8. VISUALIZATION SAVED: optics_5_students_demo.png")
    plt.show()

def explain_optics_algorithm():
    """
    Explain OPTICS algorithm concepts
    """
    print("\n" + "="*60)
    print("OPTICS ALGORITHM EXPLANATION")
    print("="*60)
    
    explanation = """
    OPTICS (Ordering Points To Identify the Clustering Structure)
    
    KEY CONCEPTS:
    
    1. CORE DISTANCE:
       - Minimum radius to have min_samples points in neighborhood
       - Measure of density around a point
    
    2. REACHABILITY DISTANCE:
       - Distance from one point to another
       - Max of (core distance, actual distance)
       - Lower values = points are closer/denser
    
    3. ORDERING:
       - Points are processed in a specific order
       - Creates a reachability plot showing cluster structure
    
    4. CLUSTER EXTRACTION:
       - Valleys in reachability plot = dense clusters
       - Peaks = boundaries between clusters
       - Noise points have high reachability
    
    ADVANTAGES:
    - No need to specify number of clusters
    - Handles varying density clusters
    - Produces hierarchical cluster structure
    - Robust to parameter choices
    
    PARAMETERS:
    - min_samples: Minimum points for a core point
    - xi: Steepness threshold for cluster extraction
    - min_cluster_size: Minimum fraction of points in cluster
    """
    
    print(explanation)

if __name__ == "__main__":
    # Run demonstration
    df_result, model = demonstrate_optics_5_students()
    
    # Explain algorithm
    explain_optics_algorithm()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*60)