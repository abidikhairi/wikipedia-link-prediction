## Competition Description (Kaggle)

Link prediction is an important task applied in networks. Given a pair of nodes ( u, v ) we need to predict if the edge between nodes u and v will be present or not. Link prediction is strongly related to recommendations, network reconstruction and network evolution. In this competition, we focus on the link prediction task applied to Wikipedia articles. In particular, given a sparsified subgraph of the Wikipedia network, we need to predict if a link exists between two Wikipedia pages u and v.

More specifically, you are given a ground-truth file which contains pairs of nodes corresponding to positive of negative samples. For example, if an edge exists between two nodes then the corresponding label is set to 1. Otherwise, the label is 0. From the original file, 20% of the information has been removed. This includes positive pairs (the edge exists) as well as negative pairs (the edge does not exist). Your mission is to correctly identify the positive and negative pairs.

Please, keep in mind that if a pair of nodes is not reported in the file, this does not imply that there is no edge between them. In other words, if a pair of nodes ( u, v ) does not appear in the file, no conclusion can be drawn with respect to their direct connection. Also, please note that the test dataset has been split such as 50% corresponds to the public leaderboard and the other 50% corresponds to the private leaderboard.

Good luck and have fun!

_The DSAA 2023 Competition Team_
---

<div align="center">    
 
# Your Project Name     

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## Description   
What it does   

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/YourGithubName/deep-learning-project-template

# install project   
cd deep-learning-project-template 
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python lit_classifier_main.py    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
