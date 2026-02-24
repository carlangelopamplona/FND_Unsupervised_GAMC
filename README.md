
<hr>

<h1> GAMC: An Unsupervised Method for Fake News Detection using Graph Autoencoder with Masking </h1>

GAMC is an unsupervised fake news detection technique using the graph autoencoder with masking and contrastive learning. The code related to the paper below：

Shu Yin, Peican Zhu, Lianwei Wu, Chao Gao, Zhen Wang, GAMC: An Unsupervised Method for Fake News Detection using Graph Autoencoder with Masking, Proceedings of the AAAI conference on artificial intelligence, 2024, 38(1): 347-355.



<h2>Dependencies </h2>

* Python >= 3.7
* [Pytorch](https://pytorch.org/) >= 1.9.0 
* [dgl](https://www.dgl.ai/) >= 0.7.2
* pyyaml == 5.4.1

<h2> Datasets </h2>

Due to file size upload limitations, datasets can be found at https://drive.google.com/drive/folders/1OslTX91kLEYIi2WBnwuFtXsVz5SS_XeR?usp=sharing.


<h2>Start </h2>

For the program start, you could run the script: 

```bash
python main_graph.py --dataset DATASETNAME --use_cfg
```
# Reference
If you make advantage of GAMC in your research, please cite the following in your manuscript:

```
@inproceedings{yin2024gamc,
  title={Gamc: an unsupervised method for fake news detection using graph autoencoder with masking},
  author={Yin, Shu and Zhu, Peican and Wu, Lianwei and Gao, Chao and Wang, Zhen},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={38},
  number={1},
  pages={347--355},
  year={2024}
}
```


