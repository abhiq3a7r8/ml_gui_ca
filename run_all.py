"""
Quick Runner Script - Run Both Parts of the Assignment
This script helps you run both parts sequentially
"""

import subprocess
import sys
import os

def print_banner(text):
    """Print a formatted banner"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def check_dependencies():
    """Check if all required packages are installed"""
    print_banner("Checking Dependencies")
    
    required_packages = [
        'numpy',
        'pandas',
        'sklearn',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} NOT installed")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n⚠️  Missing packages detected!")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✓ All dependencies satisfied!")
    return True

def check_files():
    """Check if all required files exist"""
    print_banner("Checking Project Files")
    
    required_files = [
        'optics_demo.py',
        'main.py',
        'Mall_Customers.csv'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} found")
        else:
            print(f"✗ {file} NOT found")
            missing_files.append(file)
    
    if missing_files:
        print("\n⚠️  Missing files detected!")
        print("Please ensure all files are in the project directory")
        return False
    
    print("\n✓ All files present!")
    return True

def run_demo():
    """Run the OPTICS demonstration on 5 students"""
    print_banner("Part 1: Running OPTICS Demo on 5 Students (15 marks)")
    
    try:
        result = subprocess.run([sys.executable, 'optics_demo.py'], 
                              capture_output=False, 
                              text=True)
        
        if result.returncode == 0:
            print("\n✓ Demo completed successfully!")
            return True
        else:
            print("\n✗ Demo failed!")
            return False
    except Exception as e:
        print(f"\n✗ Error running demo: {str(e)}")
        return False

def run_gui():
    """Run the GUI application"""
    print_banner("Part 2: Launching GUI Application (5 marks)")
    
    print("Instructions for GUI:")
    print("1. Click 'Upload CSV Dataset'")
    print("2. Select 'Mall_Customers.csv'")
    print("3. Choose X-Axis: 'Annual Income (k$)'")
    print("4. Choose Y-Axis: 'Spending Score (1-100)'")
    print("5. Adjust parameters if needed")
    print("6. Click 'Run Comparison Analysis'")
    print("7. View results in all 3 tabs\n")
    
    try:
        print("Launching GUI...")
        subprocess.run([sys.executable, 'main.py'])
        return True
    except Exception as e:
        print(f"\n✗ Error launching GUI: {str(e)}")
        return False

def main():
    """Main function"""
    print_banner("🔬 OPTICS Clustering Project - Complete Runner")
    
    print("This script will:")
    print("  1. Check all dependencies")
    print("  2. Check all required files")
    print("  3. Run OPTICS demo (Part 1 - 15 marks)")
    print("  4. Launch GUI application (Part 2 - 5 marks)")
    
    input("\nPress Enter to continue...")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Cannot proceed without dependencies!")
        return
    
    # Step 2: Check files
    if not check_files():
        print("\n❌ Cannot proceed without required files!")
        return
    
    # Step 3: Ask user what to run
    print_banner("Choose What to Run")
    print("1. Run Part 1 only (OPTICS Demo)")
    print("2. Run Part 2 only (GUI Application)")
    print("3. Run both sequentially")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        run_demo()
    elif choice == '2':
        run_gui()
    elif choice == '3':
        if run_demo():
            input("\nPart 1 completed. Press Enter to launch GUI (Part 2)...")
            run_gui()
    elif choice == '4':
        print("\nExiting...")
        return
    else:
        print("\n⚠️  Invalid choice!")
    
    print_banner("🎉 Project Execution Completed")
    print("Thank you for using the OPTICS Clustering Project!")
    print("\nProject Components:")
    print("  ✓ Part 1 (15 marks): OPTICS demonstration on 5 students")
    print("  ✓ Part 2 (5 marks): GUI for algorithm comparison")
    print("\nTotal: 20 marks")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)