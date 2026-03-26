import os

def create(project_name):
    folders= [
        "data/raw",
        "data/processed",
        "notebooks",
        "src",
        "src/utils",
        "src/preprocessing",
        "src/training",
        "models",
        "tests",
        "logs"
    ]
    files = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        "src/__init__.py"
        "src/app.py",
    ]
    os.makedirs(project_name,exist_ok=True)
    for folder in folders:
        path = os.path.join(project_name,folder)
        os.makedirs(path,exist_ok=True)
    for file in files:
        file_path =os.path.join(project_name,file)
        with open(file_path,"w") as f:
            f.write("")
    print("Project structure created")

if __name__ =="__main__":
    create("Crop_Recommendation")