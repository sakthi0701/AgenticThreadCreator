import sys
import os

# Add current directory to path just in case
sys.path.append(os.getcwd())

print(f"Current working directory: {os.getcwd()}")
print(f"System path: {sys.path}")

try:
    print("Attempting to import dspy_module...")
    import dspy_module
    print(f"Successfully imported dspy_module: {dspy_module}")
    
    print("Attempting to import InferTemplate from dspy_module...")
    from dspy_module import EditThread
    print(f"Successfully imported EditThread: {EditThread}")
    
except ImportError as e:
    print(f"Caught ImportError: {e}")
except Exception as e:
    print(f"Caught Exception: {e}")
