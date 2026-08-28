import json

def fix_notebook_paths():
    nb_path = 'projects/week 5/week05_tiny_transformer.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Simple string replace to fix the nested path issue
    content = content.replace('projects/week 5/visualizations/', 'visualizations/')
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
if __name__ == '__main__':
    fix_notebook_paths()
    print("Notebook paths fixed!")
