![header](https://capsule-render.vercel.app/api?type=rect&color=0080ff&height=180&section=header&text=Deep&nbsp;Knowledge&nbsp;Tracing(DKT)&%20render&fontSize=50&fontColor=FFFFFF)

# 목차
### [Team](#Team-1)
### [Skill](#Skill-1)
### [Project Overview](#Project-Overview-1)
### [Project Structure](#Project-Structure-1)
&nbsp;&nbsp;[Calendar](#Calendar-1)<br>
&nbsp;&nbsp;[Pipeline](#Pipeline-1)<br>
&nbsp;&nbsp;[1. Environment](#1-Environment-1)<br>
&nbsp;&nbsp;[2. Data](#2-Data-1)<br>
&nbsp;&nbsp;[3. Model](#3-Model-1)<br>
&nbsp;&nbsp;[4. Performance](#5-Performance-1)<br> 
### [Laboratory Report](#Laboratory-Report-1)

# Team
| **김세훈** | **문찬우** | **김시윤** | **배건우** | **이승준** |
| :------: |  :------: | :------: | :------: | :------: |
| [<img src="https://avatars.githubusercontent.com/u/8871767?v=4" height=150 width=150>](https://github.com/warpfence) | [<img src="https://avatars.githubusercontent.com/u/95879995?v=4" height=150 width=150> ](https://github.com/chanwoomoon) | [<img src="https://avatars.githubusercontent.com/u/68991530?v=4" height=150 width=150> ](https://github.com/tldbs5026) | [<img src="https://avatars.githubusercontent.com/u/83867930?v=4" height=150 width=150>](https://github.com/gunwoof) | [<img src="https://avatars.githubusercontent.com/u/133944361?v=4" height=150 width=150>](https://github.com/llseungjun) |
- 공통 : EDA & Feature engineering
- 김세훈 : Boosting모델 베이스라인 구축, T-Fixup모델 구현, K-Fold 적용, Data Augmentation 적용
- 문찬우 : Lastquery 모델링, Rnn, Gru, Tcn, 등 Sequence 모델링
- 김시윤 : LGBM 베이스라인 구축 및 최적화, lightgcn 모델링, Ensemble 진행
- 배건우 : Base environment 구축, Base pipeline 구축, Sweep 구현, Stacking ensemble 구현
- 이승준 : Saint, Saint + GRU, GRUATTN 모델링

# Skill 
### Language
  ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

### Library
  ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
  ![scikitlearn](https://img.shields.io/badge/scikitlearn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
  ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
  ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
  ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ff0000.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

### Communication
  ![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
  ![Github](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)
  ![Wandb](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)
  ![Slack](https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white)
  ![Notion](https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white)

### Environment
  ![NVIDIA-TeslaV100](https://img.shields.io/badge/NVIDIA-TeslaV100-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
  ![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
  ![Anaconda](https://img.shields.io/badge/Anaconda-44A833.svg?style=for-the-badge&logo=Anaconda&logoColor=white)

# Project Overview

### 초등학교부터 대학교까지 우리는 시험을 통해 지식을 평가해왔습니다. 그러나 시험에는 한계가 있고, 개인 맞춤형 피드백이 부족합니다. 이를 보완하기 위해 Deep Knowledge Tracing(DKT)가 등장했습니다. DKT는 우리의 지식 상태를 추적하고, 개인 맞춤형 학습을 위한 문제 추천 및 미래 성적 예측이 가능합니다.   
### 본 대회에서는 Iscream 데이터셋을 활용하여 DKT모델을 구축하여 주어진 마지막 문제를 맞출지 틀릴지 예측할 것입니다.  
 
![competition](https://github.com/boostcampaitech5/level2_dkt-recsys-09/assets/83867930/3a48942b-ef29-49a0-9fc0-f5dd65bcc78e) 

# Project Structure

### 일정
![image](https://github.com/boostcampaitech6/level2-dkt-recsys-04/assets/83867930/a1075fce-e913-45ff-b73a-2ce7d8ee511a)
### Pipeline
```bash
📦 code
    ├─ boosting # boosting model
    │  ├─ boosting
    │  ├─ lightgbm_siyun
    │  ├─ README.md
    │  ├─ main.py
    │  ├─ requirements copy.txt
    │  └─ requirements.txt
    ├─ dkt # sequence model
    │  ├─ asset
    │  ├─ dkt
    │  ├─ wandb
    │  ├─ README.md
    │  ├─ inference.py
    │  ├─ requirements.txt
    │  └─ train.py
    ├─ lightgcn # graph model
    │  ├─ readme.md
    │  └─ __init__.py
    ├─ .gitignore
    ├─ readme.md
    ├─ DKT_Recsys_팀_리포트(04조).pdf
```
![AI 파이프라인](https://github.com/boostcampaitech6/level2-dkt-recsys-04/assets/83867930/c1765101-edf3-419c-9cab-cf315835e7f2)
### 1. Environment
```
pandas==20.3
scikit-learn==1.3.2
tqdm==4.51.0
wandb==0.16.2
transformers==4.36.2
pytorch==1.12.1
torchvision==0.13.1
torchaudio==0.12.1
cudatoolkit==11.3
```
### 2. Data
**`userID`** : 사용자 별 고유번사로 총 7,422명의 사용자 데이터가 존재합니다.

**`assessmentItemID` :** 문항의 고유번호이며, 총 9,454개의 고유 문항이 있습니다.

**`testID` :** 시험지의 고유번호이며, 총 1,537개의 고유한 시험지가 있습니다.

**`answerCode` :** 사용자가 해당 문항을 맞췄는지 여부이며,  0은 틀릿 것, 1은 맞춘 것입니다. test 데이터의 경우 마지막 시퀀스의 answerCode가 -1로 예측해야 할 값입니다.

**`Timestamp` :** 사용자가 해당문항을 풀기 시작한 시점의 데이터입니다.

**`KnowleadgeTag` :** 문항 당 하나씩 배정되는 태그로, 일종의 중분류 역할을 합니다. 912개의 고유 태그가 존재합니다.

![feature](https://github.com/boostcampaitech6/level2-dkt-recsys-04/assets/68991530/198f77f4-ee69-4172-9033-0602a47cf6ba)
### 3. Model
  - **Boosting model**
    ![Boosting_Flow_Chart](https://github.com/boostcampaitech6/level2-dkt-recsys-04/assets/8871767/4031ba71-8ec2-4232-ab36-8fbc3e55f7bc)
  - **Sequence model**
    ![model_seq](https://github.com/boostcampaitech6/level2-dkt-recsys-04/assets/95879995/82b5668c-2b82-4038-8900-0ab418a64bad)

### 4. Performance

| **Model** | **LGBM-v1** | **Saint** | **Last-Query + GRU** | **LSTMATTN** | **GRUATTN** | **LGBM-v2** |
| :------: |  :------: | :------: | :------: | :------: | :------: | :------: |
| **Weight** | **0.67** | **0.084** | **0.064** | **0.064** | **0.059** | **0.059** | 

| **Public AUC** | **Public ACC** |
| :------: |  :------: | 
| 0.8156 | 0.7527 | 

![result](https://github.com/boostcampaitech6/level2-dkt-recsys-04/assets/68991530/ad3ecb4d-ce3d-4735-836e-318d1c998502)

# Laboratory Report
[DKT_Recsys_팀_리포트](https://github.com/boostcampaitech6/level2-dkt-recsys-04/blob/main/DKT_Recsys_%ED%8C%80_%EB%A6%AC%ED%8F%AC%ED%8A%B8(04%EC%A1%B0).pdf)


