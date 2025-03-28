import ast
import os
import sys
import pytest
from pathlib import Path
import pkg_resources

class ImportVisitor(ast.NodeVisitor):
    def __init__(self):
        self.imports = set()

    def visit_Import(self, node):
        for name in node.names:
            self.imports.add(name.name.split('.')[0])

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.add(node.module.split('.')[0])

def get_imports_from_file(file_path):
    """Extract imports from a Python file"""
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
        visitor = ImportVisitor()
        visitor.visit(tree)
        return visitor.imports

def get_project_imports():
    """Get all imports from project Python files"""
    imports = set()
    for root, _, files in os.walk('.'):
        # Skip venv, tests, and __pycache__ directories
        if any(skip in root for skip in ['venv', 'tests', '__pycache__']):
            continue
        
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                imports.update(get_imports_from_file(file_path))
    
    # Remove standard library modules
    stdlib_modules = set(sys.stdlib_module_names)
    return imports - stdlib_modules

def get_requirements():
    """Get all requirements from setup.py and requirements files"""
    requirements = set()
    
    # Get requirements from requirements files first
    req_files = ['requirements.txt']  # Changed to look for single requirements.txt
    for req_file in req_files:
        if os.path.exists(req_file):
            with open(req_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('-r'):
                        pkg = line.split('>=')[0].split('==')[0].strip('"\'')
                        requirements.add(pkg)
    
    return requirements

def test_all_imports_in_requirements():
    """Test that all imports used in the project are listed in requirements"""
    project_imports = get_project_imports()
    requirements = get_requirements()
    
    # Map common import names to package names
    package_mapping = {
        'yaml': 'PyYAML',
        'sklearn': 'scikit-learn',
        'PIL': 'Pillow',
        'git': 'GitPython',
        'torch': 'torch',
        'torchvision': 'torchvision',
    }
    
    # Convert import names to package names
    mapped_imports = {package_mapping.get(imp, imp) for imp in project_imports}
    
    # Find missing requirements
    missing = mapped_imports - requirements
    
    # Remove built-in packages and local modules that don't need to be in requirements
    builtin_packages = {
        'os', 'sys', 'pathlib', 'typing', 'datetime', 'json', 'enum',
        # Local project modules
        'data', 'models', 'utils', 'eval', 
        'dataset_loader', 'metrics_registry', 'dataset_registry',
        'config_validator', 'adversarial', 'experiment_tracker',
        # Setup related
        'setuptools'
    }
    missing = missing - builtin_packages
    
    if missing:
        pytest.fail(f"Found imports not listed in requirements: {missing}")

def test_requirements_consistency():
    """Test that requirements are consistent across setup.py and requirements files"""
    setup_requirements = set()
    with open('setup.py', 'r') as f:
        setup_content = f.read()
        setup_tree = ast.parse(setup_content)
        for node in ast.walk(setup_tree):
            if isinstance(node, ast.Call) and node.func.id == 'setup':
                for keyword in node.keywords:
                    if keyword.arg == 'install_requires':
                        for elt in keyword.value.elts:
                            pkg = elt.value.split('>=')[0].split('==')[0].strip('"\'')
                            setup_requirements.add(pkg)
    
    base_requirements = set()
    with open('requirements/base.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                pkg = line.split('>=')[0].split('==')[0].strip('"\'')
                base_requirements.add(pkg)
    
    assert setup_requirements == base_requirements, \
        "Requirements in setup.py don't match requirements/base.txt" 