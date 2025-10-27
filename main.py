"""
GUI Application for Comparing Clustering Algorithms
Includes OPTICS, DBSCAN, K-Means, and Hierarchical Clustering
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.cluster import OPTICS, DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import seaborn as sns

class ClusteringComparisonGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OPTICS Clustering Comparison Tool")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f0f0f0")
        
        self.df = None
        self.X = None
        self.X_scaled = None
        self.results = {}
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Title
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="🔍 OPTICS Clustering Comparison Tool",
            font=("Arial", 20, "bold"),
            fg="white",
            bg="#2c3e50"
        )
        title_label.pack(pady=10)
        
        # Main container
        main_container = tk.Frame(self.root, bg="#f0f0f0")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left panel - Controls
        left_panel = tk.Frame(main_container, bg="white", width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))
        left_panel.pack_propagate(False)
        
        self.create_control_panel(left_panel)
        
        # Right panel - Visualization
        right_panel = tk.Frame(main_container, bg="white")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_visualization_panel(right_panel)
        
    def create_control_panel(self, parent):
        """Create control panel with buttons and options"""
        
        # Data Loading Section
        data_frame = tk.LabelFrame(
            parent, 
            text="📁 Data Loading", 
            font=("Arial", 12, "bold"),
            bg="white",
            padx=10,
            pady=10
        )
        data_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(
            data_frame,
            text="Upload CSV Dataset",
            command=self.load_dataset,
            bg="#3498db",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=20,
            pady=5,
            cursor="hand2"
        ).pack(fill=tk.X, pady=5)
        
        self.data_label = tk.Label(
            data_frame,
            text="No dataset loaded",
            font=("Arial", 9),
            bg="white",
            fg="gray"
        )
        self.data_label.pack(pady=5)
        
        # Feature Selection
        feature_frame = tk.LabelFrame(
            parent,
            text="🎯 Feature Selection",
            font=("Arial", 12, "bold"),
            bg="white",
            padx=10,
            pady=10
        )
        feature_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(
            feature_frame,
            text="X-Axis Feature:",
            font=("Arial", 10),
            bg="white"
        ).pack(anchor=tk.W)
        
        self.x_feature = ttk.Combobox(feature_frame, state="readonly")
        self.x_feature.pack(fill=tk.X, pady=5)
        
        tk.Label(
            feature_frame,
            text="Y-Axis Feature:",
            font=("Arial", 10),
            bg="white"
        ).pack(anchor=tk.W)
        
        self.y_feature = ttk.Combobox(feature_frame, state="readonly")
        self.y_feature.pack(fill=tk.X, pady=5)
        
        # Algorithm Selection
        algo_frame = tk.LabelFrame(
            parent,
            text="⚙️ Algorithm Parameters",
            font=("Arial", 12, "bold"),
            bg="white",
            padx=10,
            pady=10
        )
        algo_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # OPTICS parameters
        tk.Label(algo_frame, text="OPTICS - min_samples:", bg="white").pack(anchor=tk.W)
        self.optics_min_samples = tk.Scale(
            algo_frame, 
            from_=2, 
            to=20, 
            orient=tk.HORIZONTAL,
            bg="white"
        )
        self.optics_min_samples.set(5)
        self.optics_min_samples.pack(fill=tk.X)
        
        # K-Means parameters
        tk.Label(algo_frame, text="K-Means - n_clusters:", bg="white").pack(anchor=tk.W)
        self.kmeans_clusters = tk.Scale(
            algo_frame,
            from_=2,
            to=10,
            orient=tk.HORIZONTAL,
            bg="white"
        )
        self.kmeans_clusters.set(5)
        self.kmeans_clusters.pack(fill=tk.X)
        
        # DBSCAN parameters
        tk.Label(algo_frame, text="DBSCAN - eps:", bg="white").pack(anchor=tk.W)
        self.dbscan_eps = tk.Scale(
            algo_frame,
            from_=0.1,
            to=2.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            bg="white"
        )
        self.dbscan_eps.set(0.5)
        self.dbscan_eps.pack(fill=tk.X)
        
        # Run Analysis Button
        tk.Button(
            parent,
            text="▶️ Run Comparison Analysis",
            command=self.run_comparison,
            bg="#27ae60",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=10,
            cursor="hand2"
        ).pack(fill=tk.X, padx=10, pady=20)
        
        # Results Display
        self.results_text = tk.Text(
            parent,
            height=10,
            font=("Courier", 9),
            wrap=tk.WORD,
            bg="#ecf0f1"
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def create_visualization_panel(self, parent):
        """Create visualization panel"""
        
        # Notebook for multiple plots
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.tab1 = tk.Frame(self.notebook, bg="white")
        self.tab2 = tk.Frame(self.notebook, bg="white")
        self.tab3 = tk.Frame(self.notebook, bg="white")
        
        self.notebook.add(self.tab1, text="Clustering Results")
        self.notebook.add(self.tab2, text="Reachability Plots")
        self.notebook.add(self.tab3, text="Performance Metrics")
        
    def load_dataset(self):
        """Load CSV dataset"""
        filename = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.df = pd.read_csv(filename)
                
                # Get numeric columns
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_cols) < 2:
                    messagebox.showerror("Error", "Dataset must have at least 2 numeric columns")
                    return
                
                # Update feature dropdowns
                self.x_feature['values'] = numeric_cols
                self.y_feature['values'] = numeric_cols
                
                if len(numeric_cols) >= 2:
                    self.x_feature.current(0)
                    self.y_feature.current(1)
                
                self.data_label.config(
                    text=f"✓ Loaded: {len(self.df)} rows, {len(numeric_cols)} numeric features",
                    fg="green"
                )
                
                messagebox.showinfo("Success", f"Dataset loaded successfully!\nRows: {len(self.df)}\nNumeric Features: {len(numeric_cols)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset:\n{str(e)}")
    
    def run_comparison(self):
        """Run clustering comparison analysis"""
        
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return
        
        if not self.x_feature.get() or not self.y_feature.get():
            messagebox.showwarning("Warning", "Please select X and Y features!")
            return
        
        try:
            # Prepare data
            x_col = self.x_feature.get()
            y_col = self.y_feature.get()
            
            self.X = self.df[[x_col, y_col]].values
            
            # Standardize
            scaler = StandardScaler()
            self.X_scaled = scaler.fit_transform(self.X)
            
            # Run algorithms
            self.results = {}
            
            # 1. OPTICS
            optics = OPTICS(min_samples=self.optics_min_samples.get())
            self.results['OPTICS'] = {
                'model': optics,
                'labels': optics.fit_predict(self.X_scaled),
                'name': 'OPTICS'
            }
            
            # 2. K-Means
            kmeans = KMeans(n_clusters=self.kmeans_clusters.get(), random_state=42, n_init=10)
            self.results['K-Means'] = {
                'model': kmeans,
                'labels': kmeans.fit_predict(self.X_scaled),
                'name': 'K-Means'
            }
            
            # 3. DBSCAN
            dbscan = DBSCAN(eps=self.dbscan_eps.get(), min_samples=5)
            self.results['DBSCAN'] = {
                'model': dbscan,
                'labels': dbscan.fit_predict(self.X_scaled),
                'name': 'DBSCAN'
            }
            
            # 4. Hierarchical
            hierarchical = AgglomerativeClustering(n_clusters=self.kmeans_clusters.get())
            self.results['Hierarchical'] = {
                'model': hierarchical,
                'labels': hierarchical.fit_predict(self.X_scaled),
                'name': 'Hierarchical'
            }
            
            # Calculate metrics
            self.calculate_metrics()
            
            # Visualize
            self.visualize_results(x_col, y_col)
            
            messagebox.showinfo("Success", "Comparison analysis completed!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed:\n{str(e)}")
    
    def calculate_metrics(self):
        """Calculate clustering metrics"""
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "CLUSTERING COMPARISON RESULTS\n")
        self.results_text.insert(tk.END, "="*50 + "\n\n")
        
        for algo_name, result in self.results.items():
            labels = result['labels']
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            self.results_text.insert(tk.END, f"{algo_name}:\n")
            self.results_text.insert(tk.END, f"  Clusters: {n_clusters}\n")
            self.results_text.insert(tk.END, f"  Noise points: {n_noise}\n")
            
            # Calculate metrics if we have valid clusters
            if n_clusters > 1 and n_clusters < len(self.X_scaled) and n_noise < len(self.X_scaled):
                # Filter out noise for metrics
                valid_mask = labels != -1
                if valid_mask.sum() > 1:
                    try:
                        silhouette = silhouette_score(self.X_scaled[valid_mask], labels[valid_mask])
                        davies_bouldin = davies_bouldin_score(self.X_scaled[valid_mask], labels[valid_mask])
                        calinski = calinski_harabasz_score(self.X_scaled[valid_mask], labels[valid_mask])
                        
                        self.results_text.insert(tk.END, f"  Silhouette Score: {silhouette:.4f}\n")
                        self.results_text.insert(tk.END, f"  Davies-Bouldin: {davies_bouldin:.4f}\n")
                        self.results_text.insert(tk.END, f"  Calinski-Harabasz: {calinski:.2f}\n")
                        
                        result['metrics'] = {
                            'silhouette': silhouette,
                            'davies_bouldin': davies_bouldin,
                            'calinski': calinski
                        }
                    except:
                        self.results_text.insert(tk.END, "  Metrics: N/A\n")
            
            self.results_text.insert(tk.END, "\n")
    
    def visualize_results(self, x_col, y_col):
        """Visualize clustering results"""
        
        # Clear previous plots
        for widget in self.tab1.winfo_children():
            widget.destroy()
        for widget in self.tab2.winfo_children():
            widget.destroy()
        for widget in self.tab3.winfo_children():
            widget.destroy()
        
        # Tab 1: Clustering scatter plots
        fig1 = plt.figure(figsize=(14, 10))
        
        for idx, (algo_name, result) in enumerate(self.results.items(), 1):
            ax = fig1.add_subplot(2, 2, idx)
            
            scatter = ax.scatter(
                self.X[:, 0],
                self.X[:, 1],
                c=result['labels'],
                cmap='viridis',
                s=50,
                alpha=0.6,
                edgecolors='black',
                linewidths=0.5
            )
            
            ax.set_xlabel(x_col, fontsize=10)
            ax.set_ylabel(y_col, fontsize=10)
            ax.set_title(f"{algo_name} Clustering", fontsize=12, weight='bold')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Cluster')
        
        plt.tight_layout()
        
        canvas1 = FigureCanvasTkAgg(fig1, self.tab1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab 2: Reachability plots (OPTICS and DBSCAN)
        fig2 = plt.figure(figsize=(14, 5))
        
        # OPTICS reachability
        ax1 = fig2.add_subplot(121)
        optics_model = self.results['OPTICS']['model']
        reachability = optics_model.reachability_[optics_model.ordering_]
        
        ax1.bar(range(len(reachability)), reachability, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Order', fontsize=10)
        ax1.set_ylabel('Reachability Distance', fontsize=10)
        ax1.set_title('OPTICS Reachability Plot', fontsize=12, weight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Cluster distribution
        ax2 = fig2.add_subplot(122)
        
        for algo_name, result in self.results.items():
            labels = result['labels']
            unique, counts = np.unique(labels, return_counts=True)
            ax2.bar(
                [f"{algo_name}\nC{l}" if l != -1 else f"{algo_name}\nNoise" for l in unique],
                counts,
                alpha=0.7,
                label=algo_name
            )
        
        ax2.set_xlabel('Algorithm - Cluster', fontsize=10)
        ax2.set_ylabel('Number of Points', fontsize=10)
        ax2.set_title('Cluster Size Distribution', fontsize=12, weight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        canvas2 = FigureCanvasTkAgg(fig2, self.tab2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab 3: Performance metrics comparison
        fig3 = plt.figure(figsize=(14, 5))
        
        metrics_data = {
            'Algorithm': [],
            'Silhouette': [],
            'Davies-Bouldin': [],
            'Calinski-Harabasz': []
        }
        
        for algo_name, result in self.results.items():
            if 'metrics' in result:
                metrics_data['Algorithm'].append(algo_name)
                metrics_data['Silhouette'].append(result['metrics']['silhouette'])
                metrics_data['Davies-Bouldin'].append(result['metrics']['davies_bouldin'])
                metrics_data['Calinski-Harabasz'].append(result['metrics']['calinski'])
        
        if metrics_data['Algorithm']:
            # Silhouette Score
            ax1 = fig3.add_subplot(131)
            ax1.bar(metrics_data['Algorithm'], metrics_data['Silhouette'], color='skyblue', edgecolor='black')
            ax1.set_ylabel('Score', fontsize=10)
            ax1.set_title('Silhouette Score\n(Higher is Better)', fontsize=11, weight='bold')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Davies-Bouldin Index
            ax2 = fig3.add_subplot(132)
            ax2.bar(metrics_data['Algorithm'], metrics_data['Davies-Bouldin'], color='salmon', edgecolor='black')
            ax2.set_ylabel('Score', fontsize=10)
            ax2.set_title('Davies-Bouldin Index\n(Lower is Better)', fontsize=11, weight='bold')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Calinski-Harabasz Score
            ax3 = fig3.add_subplot(133)
            ax3.bar(metrics_data['Algorithm'], metrics_data['Calinski-Harabasz'], color='lightgreen', edgecolor='black')
            ax3.set_ylabel('Score', fontsize=10)
            ax3.set_title('Calinski-Harabasz Score\n(Higher is Better)', fontsize=11, weight='bold')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        canvas3 = FigureCanvasTkAgg(fig3, self.tab3)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = ClusteringComparisonGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()