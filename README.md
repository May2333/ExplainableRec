# Explainable Rec

---
Implementation authors: Junjie Lai and Yue Wu.

This repository is an official PyTorch implementation of the TKDE 2023 paper ExplainableRec [Knowledge Enhanced Graph Neural Networks for Explainable Recommendation](https://ieeexplore.ieee.org/document/9681226).


### Required packages
The code has been tested running under Python 3.6, with the following packages installed (along with their dependencies):
* pytorch >= 1.0
* numpy >= 1.14.5
* nltk >= 3.5
* gensim >= 3.8.3
* tensorboardX
* wordfreq
* datasketch

### Files in the folder
* `dataset/`
    * `Home_and_Kitchen_5.json`: a dataset from Amazon 5-core. The original page (http://jmcauley.ucsd.edu/data/amazon) is inaccessible now, you can download it from [Home_and_Kitchen_5.json](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Home_and_Kitchen_5.json.gz) and put it in there.
	* `utils.py`: some preprocess methods and tools.
	* `text2url.py`: which normalizes natural-language text into the ConceptNet URI representation, copy from [conceptnet-numberbatch repository](https://github.com/commonsense/conceptnet-numberbatch).
	* `DataManager.py`: the script of Dataset and DataLoader.
	* `build_dict.py`: create the dict from dataset and provide tokenizor

* `saved/`
	* `conceptnet/`: you need to download `numberbatch-en.txt` from [conceptnet-numberbatch repository](https://github.com/commonsense/conceptnet-numberbatch) and put it in there.

* `modules/`: implementations of ExplainableRec.


### Training the code
> $ `cd ExplainableRec` \
> $ `python train.py --run_type train --data_path dataset/Home_and_Kitchen_5.json`

### Testing the code
> $ `cd ExplainableRec` \
> $ `python train.py --run_type test --data_path dataset/Home_and_Kitchen_5.json`


### Citing ExplainableRec
If you use our ExpalinableRec in your research, please cite:
> @ARTICLE{9681226, \
  author={Lyu, Ziyu and Wu, Yue and Lai, Junjie and Yang, Min and Li, Chengming and Zhou, Wei},\
  journal={IEEE Transactions on Knowledge and Data Engineering},\
  title={Knowledge Enhanced Graph Neural Networks for Explainable Recommendation},\
  year={2023},\
  volume={35},\
  number={5},\
  pages={4954-4968},\
  doi={10.1109/TKDE.2022.3142260}}