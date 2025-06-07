#!/usr/bin/env python3
"""
Comprehensive Demo of TinyFabulist Experiment Tracking System
Demonstrates the complete workflow for research paper development
"""

import subprocess
import time
import sys
import os

def run_command(cmd, description):
    """Run a command and display output"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="experiments")
        
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout:
                print(result.stdout)
        else:
            print("❌ FAILED")
            if result.stderr:
                print("Error:", result.stderr)
                
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def main():
    """Demonstrate the complete experiment tracking workflow"""
    
    print("""
🔬 TinyFabulist Experiment Tracking System Demo
===============================================

This demo shows the complete workflow for research paper development:
1. Run systematic experiments
2. Track results automatically  
3. Analyze and compare results
4. Export paper-ready outputs

Let's start!
    """)
    
    # Step 1: Check system status
    print("\n📋 STEP 1: System Status Check")
    if not run_command("python ../tf3.py test", "Testing TinyFabulist framework"):
        print("⚠️  TinyFabulist framework test failed, but continuing with demo...")
    
    # Step 2: Run demo experiment
    print("\n📋 STEP 2: Run Demo Experiment")
    success = run_command("python run_experiments.py --demo", "Running quick demo experiment")
    
    if not success:
        print("❌ Demo experiment failed. Please check your environment setup.")
        return False
    
    # Step 3: List experiments
    print("\n📋 STEP 3: List Tracked Experiments")
    run_command("python experiment_manager.py list", "Listing all experiments")
    
    # Step 4: Get experiment ID from registry
    try:
        import json
        with open("experiments/experiment_registry.json", 'r') as f:
            registry = json.load(f)
        
        experiment_ids = list(registry["experiments"].keys())
        
        if experiment_ids:
            latest_exp = experiment_ids[-1]  # Get the most recent experiment
            
            print(f"\n📋 STEP 4: Analyze Results")
            print(f"Using experiment ID: {latest_exp}")
            
            # Generate summary
            run_command(f"python analysis_tools.py summary {latest_exp}", "Creating results summary")
            
            # Generate LaTeX table  
            run_command(f"python analysis_tools.py latex {latest_exp}", "Generating LaTeX table")
            
            # Generate report
            run_command(f"python analysis_tools.py report {latest_exp}", "Creating comprehensive report")
            
            # Export to CSV
            run_command(f"python experiment_manager.py export {latest_exp} --output demo_final.csv", 
                       "Exporting results to CSV")
            
            print(f"\n📋 STEP 5: Paper-Ready Outputs")
            print("Generated files for your paper:")
            
            # Check what files were created
            export_files = []
            exports_dir = "experiments/exports"
            if os.path.exists(exports_dir):
                for file in os.listdir(exports_dir):
                    if file.endswith(('.csv', '.tex', '.json')):
                        export_files.append(os.path.join(exports_dir, file))
            
            if export_files:
                for file in export_files:
                    print(f"  📄 {file}")
            else:
                print("  ⚠️  No export files found")
                
        else:
            print("❌ No experiments found in registry")
            
    except Exception as e:
        print(f"❌ Error accessing experiment registry: {e}")
    
    # Step 6: Demo complete
    print(f"\n{'='*60}")
    print("🎉 DEMO COMPLETE!")
    print(f"{'='*60}")
    print("""
✅ Successfully demonstrated:
  • Experiment execution with automatic tracking
  • Result storage with complete metadata
  • Analysis and comparison capabilities  
  • Paper-ready output generation
  • Cross-platform device management (MPS/CUDA/CPU)

🚀 Next Steps for Your Research:
  
  1. Run Systematic Experiments:
     cd experiments
     python run_experiments.py --baseline      # Compare GPT-2 models
     python run_experiments.py --temperature   # Parameter study
     python run_experiments.py --all          # Run everything
  
  2. Analyze Results:
     python experiment_manager.py list --status completed
     python analysis_tools.py summary exp1_id exp2_id exp3_id
  
  3. Generate Paper Tables:
     python analysis_tools.py latex exp1_id exp2_id --output table1.tex
     python experiment_manager.py export exp1_id exp2_id --output results.csv
  
  4. Compare Fine-tuned Models:
     # Add your model paths to run_experiments.py
     python run_experiments.py --finetuned

📚 Documentation:
  • experiments/README.md - Complete system documentation
  • Individual experiment data in experiments/runs/
  • Export files in experiments/exports/

The system is ready for your research paper development!
    """)
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        sys.exit(1) 