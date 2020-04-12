# CS577(NLP) Project
Project Repository for NLP 


# 0. Layout

```bash
.
├── README.md
├── data

└── requirements.txt

```

# 1. Setup

## 1.1 Build a virtual environment

1. install virtual 
```bash
pip install virtualenv
```

2. install python virtual environment named 'ie590_project_venv'
```bash
python3 -m venv cs577_project_venv
```

3. activate the virtual environment module
```bash
source cs577_project/bin/activate
```

4. when finish working on the venv
```bash
deactivate
```


# 2. When making a pull request to Github

```diff
- Note: Use a branch named by your github account. Do not push to the master branch.
```

1. setup git on the working directory (you only do it once)
```bash
cd posture_venv # your virtual venv directory
mkdir posture 
cd posture # make sure you are in the directory which will be your working directory

git init
git remote add origin https://github.com/kmanchel/cs577_project.git
git pull origin master
```

2. create a branch (you only do it once)
```bash
git branch <your_github_account_name>
git checkout <your_github_account_name>
```

3. update codes

4. push your update to your branch in github

```bash
git add <updated_file_names>
git commit -m "<message>"
git push origin <your_github_account_name>
```

5. make a pull request in the Github repository website



