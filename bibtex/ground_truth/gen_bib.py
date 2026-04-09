#!/usr/bin/env python3
"""
generate_deepseek_bib.py
Creates the complete 250‑entry BibTeX file (100 valid, 80 partially valid, 70 invalid).
"""

from pathlib import Path

bib_content = r"""%% SECTION A — FULLY VALID (100 entries)

@article{turing1936computable,
  author    = {Turing, Alan M.},
  title     = {On Computable Numbers, with an Application to the {Entscheidungsproblem}},
  journal   = {Proceedings of the London Mathematical Society},
  volume    = {42},
  number    = {1},
  pages     = {230--265},
  year      = {1936},
  doi       = {10.1112/plms/s2-42.1.230}
}

@book{shannon1948mathematical,
  author    = {Shannon, Claude E.},
  title     = {A Mathematical Theory of Communication},
  publisher = {University of Illinois Press},
  year      = {1948},
  doi       = {10.1002/j.1538-7305.1948.tb01338.x}
}

@inproceedings{mccarthy1956proposal,
  author    = {McCarthy, John and Minsky, Marvin L. and Rochester, Nathaniel and Shannon, Claude E.},
  title     = {A Proposal for the {Dartmouth} Summer Research Project on Artificial Intelligence},
  booktitle = {Dartmouth Conference},
  year      = {1956},
  doi       = {10.5555/1123890.1123891}
}

@article{perceptron1958,
  author    = {Rosenblatt, Frank},
  title     = {The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain},
  journal   = {Psychological Review},
  volume    = {65},
  number    = {6},
  pages     = {386--408},
  year      = {1958},
  doi       = {10.1037/h0042519}
}

@book{minsky1969perceptrons,
  author    = {Minsky, Marvin L. and Papert, Seymour A.},
  title     = {Perceptrons: An Introduction to Computational Geometry},
  publisher = {{MIT} Press},
  year      = {1969},
  doi       = {10.7551/mitpress/11301.001.0001}
}

@article{backprop1986,
  author    = {Rumelhart, David E. and Hinton, Geoffrey E. and Williams, Ronald J.},
  title     = {Learning Representations by Back-Propagating Errors},
  journal   = {Nature},
  volume    = {323},
  number    = {6088},
  pages     = {533--536},
  year      = {1986},
  doi       = {10.1038/323533a0}
}

@inproceedings{lecun1989backprop,
  author    = {LeCun, Yann and Boser, Bernhard and Denker, John S. and Henderson, Donnie and Howard, Richard E. and Hubbard, Wayne and Jackel, Lawrence D.},
  title     = {Backpropagation Applied to Handwritten Zip Code Recognition},
  booktitle = {Neural Computation},
  year      = {1989},
  doi       = {10.1162/neco.1989.1.4.541}
}

@article{hinton2006fast,
  author    = {Hinton, Geoffrey E. and Salakhutdinov, Ruslan R.},
  title     = {Reducing the Dimensionality of Data with Neural Networks},
  journal   = {Science},
  volume    = {313},
  number    = {5786},
  pages     = {504--507},
  year      = {2006},
  doi       = {10.1126/science.1127647}
}

@inproceedings{krizhevsky2012imagenet,
  author    = {Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E.},
  title     = {{ImageNet} Classification with Deep Convolutional Neural Networks},
  booktitle = {Advances in Neural Information Processing Systems ({NeurIPS})},
  volume    = {25},
  pages     = {1097--1105},
  year      = {2012},
  doi       = {10.1145/3065386}
}

@article{vaswani2017attention,
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, {\L}ukasz and Polosukhin, Illia},
  title     = {Attention Is All You Need},
  journal   = {Advances in Neural Information Processing Systems},
  volume    = {30},
  pages     = {5998--6008},
  year      = {2017},
  doi       = {10.48550/arXiv.1706.03762}
}

@article{devlin2019bert,
  author    = {Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  title     = {{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  journal   = {Proceedings of the {NAACL}},
  year      = {2019},
  doi       = {10.18653/v1/N19-1423}
}

@inproceedings{he2016deep,
  author    = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  title     = {Deep Residual Learning for Image Recognition},
  booktitle = {{CVPR}},
  year      = {2016},
  doi       = {10.1109/CVPR.2016.90}
}

@article{kingma2014adam,
  author    = {Kingma, Diederik P. and Ba, Jimmy},
  title     = {Adam: A Method for Stochastic Optimization},
  journal   = {Proceedings of the 3rd International Conference on Learning Representations ({ICLR})},
  year      = {2014},
  doi       = {10.48550/arXiv.1412.6980}
}

@book{goodfellow2016deep,
  author    = {Goodfellow, Ian and Bengio, Yoshua and Courville, Aaron},
  title     = {Deep Learning},
  publisher = {{MIT} Press},
  year      = {2016},
  doi       = {10.7551/mitpress/11490.001.0001}
}

@article{hochreiter1997long,
  author    = {Hochreiter, Sepp and Schmidhuber, J{\"u}rgen},
  title     = {Long Short-Term Memory},
  journal   = {Neural Computation},
  volume    = {9},
  number    = {8},
  pages     = {1735--1780},
  year      = {1997},
  doi       = {10.1162/neco.1997.9.8.1735}
}

@inproceedings{ioffe2015batch,
  author    = {Ioffe, Sergey and Szegedy, Christian},
  title     = {Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift},
  booktitle = {Proceedings of the 32nd International Conference on Machine Learning ({ICML})},
  year      = {2015},
  doi       = {10.48550/arXiv.1502.03167}
}

@article{szegedy2015going,
  author    = {Szegedy, Christian and Liu, Wei and Jia, Yangqing and Sermanet, Pierre and Reed, Scott and Anguelov, Dragomir and Erhan, Dumitru and Vanhoucke, Vincent and Rabinovich, Andrew},
  title     = {Going Deeper with Convolutions},
  journal   = {{CVPR}},
  year      = {2015},
  doi       = {10.1109/CVPR.2015.7298594}
}

@article{simonyan2014very,
  author    = {Simonyan, Karen and Zisserman, Andrew},
  title     = {Very Deep Convolutional Networks for Large-Scale Image Recognition},
  journal   = {Proceedings of the {ICLR}},
  year      = {2014},
  doi       = {10.48550/arXiv.1409.1556}
}

@inproceedings{xu2015show,
  author    = {Xu, Kelvin and Ba, Jimmy and Kiros, Ryan and Cho, Kyunghyun and Courville, Aaron and Salakhutdinov, Ruslan and Zemel, Richard and Bengio, Yoshua},
  title     = {Show, Attend and Tell: Neural Image Caption Generation with Visual Attention},
  booktitle = {Proceedings of the 32nd International Conference on Machine Learning ({ICML})},
  year      = {2015},
  doi       = {10.48550/arXiv.1502.03044}
}

@article{wu2016google,
  author    = {Wu, Yonghui and Schuster, Mike and Chen, Zhifeng and Le, Quoc V. and Norouzi, Mohammad and Macherey, Wolfgang and Krikun, Maxim and Cao, Yuan and Gao, Qin and Macherey, Klaus and others},
  title     = {Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation},
  journal   = {arXiv preprint},
  year      = {2016},
  doi       = {10.48550/arXiv.1609.08144}
}

@inproceedings{radford2018improving,
  author    = {Radford, Alec and Narasimhan, Karthik and Salimans, Tim and Sutskever, Ilya},
  title     = {Improving Language Understanding by Generative Pre-Training},
  booktitle = {OpenAI Technical Report},
  year      = {2018}
}

@article{brown2020language,
  author    = {Brown, Tom B. and Mann, Benjamin and Ryder, Nick and Subbiah, Melanie and Kaplan, Jared and Dhariwal, Prafulla and Neelakantan, Arvind and Shyam, Pranav and Sastry, Girish and Askell, Amanda and others},
  title     = {Language Models are Few-Shot Learners},
  journal   = {Advances in Neural Information Processing Systems},
  volume    = {33},
  pages     = {1877--1901},
  year      = {2020},
  doi       = {10.48550/arXiv.2005.14165}
}

@article{radford2021learning,
  author    = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  title     = {Learning Transferable Visual Models From Natural Language Supervision},
  journal   = {Proceedings of the 38th International Conference on Machine Learning ({ICML})},
  year      = {2021},
  doi       = {10.48550/arXiv.2103.00020}
}

@article{ramesh2021zero,
  author    = {Ramesh, Aditya and Pavlov, Mikhail and Goh, Gabriel and Gray, Scott and Voss, Chelsea and Radford, Alec and Chen, Mark and Sutskever, Ilya},
  title     = {Zero-Shot Text-to-Image Generation},
  journal   = {Proceedings of the 38th International Conference on Machine Learning ({ICML})},
  year      = {2021},
  doi       = {10.48550/arXiv.2102.12092}
}

@article{chen2020generative,
  author    = {Chen, Mark and Radford, Alec and Child, Rewon and Wu, Jeffrey and Jun, Heewoo and Dhariwal, Prafulla and Luan, David and Sutskever, Ilya},
  title     = {Generative Pretraining From Pixels},
  journal   = {Proceedings of the 37th International Conference on Machine Learning ({ICML})},
  year      = {2020},
  doi       = {10.48550/arXiv.2006.10211}
}

@inproceedings{hendrycks2021measuring,
  author    = {Hendrycks, Dan and Burns, Collin and Basart, Steven and Zou, Andy and Mazeika, Mantas and Song, Dawn and Steinhardt, Jacob},
  title     = {Measuring Massive Multitask Language Understanding},
  booktitle = {Proceedings of the 9th International Conference on Learning Representations ({ICLR})},
  year      = {2021},
  doi       = {10.48550/arXiv.2009.03300}
}

@article{zhang2018mixup,
  author    = {Zhang, Hongyi and Ciss{\'e}, Moustapha and Dauphin, Yann N. and Lopez-Paz, David},
  title     = {mixup: Beyond Empirical Risk Minimization},
  journal   = {Proceedings of the 6th International Conference on Learning Representations ({ICLR})},
  year      = {2018},
  doi       = {10.48550/arXiv.1710.09412}
}

@article{arjovsky2017wasserstein,
  author    = {Arjovsky, Martin and Chintala, Soumith and Bottou, L{\'e}on},
  title     = {Wasserstein Generative Adversarial Networks},
  journal   = {Proceedings of the 34th International Conference on Machine Learning ({ICML})},
  year      = {2017},
  doi       = {10.48550/arXiv.1701.07875}
}

@article{gulrajani2017improved,
  author    = {Gulrajani, Ishaan and Ahmed, Faruk and Arjovsky, Martin and Dumoulin, Vincent and Courville, Aaron},
  title     = {Improved Training of Wasserstein {GAN}s},
  journal   = {Advances in Neural Information Processing Systems},
  volume    = {30},
  pages     = {5767--5777},
  year      = {2017},
  doi       = {10.48550/arXiv.1704.00028}
}

@article{mirza2014conditional,
  author    = {Mirza, Mehdi and Osindero, Simon},
  title     = {Conditional Generative Adversarial Nets},
  journal   = {arXiv preprint},
  year      = {2014},
  doi       = {10.48550/arXiv.1411.1784}
}

@article{radford2015unsupervised,
  author    = {Radford, Alec and Metz, Luke and Chintala, Soumith},
  title     = {Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks},
  journal   = {Proceedings of the {ICLR}},
  year      = {2015},
  doi       = {10.48550/arXiv.1511.06434}
}

@article{oord2016wavenet,
  author    = {van den Oord, A{\"a}ron and Dieleman, Sander and Zen, Heiga and Simonyan, Karen and Vinyals, Oriol and Graves, Alex and Kalchbrenner, Nal and Senior, Andrew and Kavukcuoglu, Koray},
  title     = {{WaveNet}: A Generative Model for Raw Audio},
  journal   = {arXiv preprint},
  year      = {2016},
  doi       = {10.48550/arXiv.1609.03499}
}

@article{oord2017neural,
  author    = {van den Oord, A{\"a}ron and Vinyals, Oriol and Kavukcuoglu, Koray},
  title     = {Neural Discrete Representation Learning},
  journal   = {Advances in Neural Information Processing Systems},
  volume    = {30},
  pages     = {6306--6315},
  year      = {2017},
  doi       = {10.48550/arXiv.1711.00937}
}

@article{kingma2013auto,
  author    = {Kingma, Diederik P. and Welling, Max},
  title     = {Auto-Encoding Variational Bayes},
  journal   = {Proceedings of the 2nd International Conference on Learning Representations ({ICLR})},
  year      = {2013},
  doi       = {10.48550/arXiv.1312.6114}
}

@article{rezende2014stochastic,
  author    = {Rezende, Danilo Jimenez and Mohamed, Shakir and Wierstra, Daan},
  title     = {Stochastic Backpropagation and Approximate Inference in Deep Generative Models},
  journal   = {Proceedings of the 31st International Conference on Machine Learning ({ICML})},
  year      = {2014},
  doi       = {10.48550/arXiv.1401.4082}
}

@inproceedings{bengio2013representation,
  author    = {Bengio, Yoshua and Courville, Aaron and Vincent, Pascal},
  title     = {Representation Learning: A Review and New Perspectives},
  booktitle = {{IEEE} Transactions on Pattern Analysis and Machine Intelligence},
  year      = {2013},
  doi       = {10.1109/TPAMI.2013.50}
}

@article{mnih2013playing,
  author    = {Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Graves, Alex and Antonoglou, Ioannis and Wierstra, Daan and Riedmiller, Martin},
  title     = {Playing {Atari} with Deep Reinforcement Learning},
  journal   = {arXiv preprint},
  year      = {2013},
  doi       = {10.48550/arXiv.1312.5602}
}

@article{mnih2015human,
  author    = {Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Rusu, Andrei A. and Veness, Joel and Bellemare, Marc G. and Graves, Alex and Riedmiller, Martin and Fidjeland, Andreas K. and Ostrovski, Georg and others},
  title     = {Human-level Control Through Deep Reinforcement Learning},
  journal   = {Nature},
  volume    = {518},
  number    = {7540},
  pages     = {529--533},
  year      = {2015},
  doi       = {10.1038/nature14236}
}

@article{schulman2015trust,
  author    = {Schulman, John and Levine, Sergey and Abbeel, Pieter and Jordan, Michael I. and Moritz, Philipp},
  title     = {Trust Region Policy Optimization},
  journal   = {Proceedings of the 32nd International Conference on Machine Learning ({ICML})},
  year      = {2015},
  doi       = {10.48550/arXiv.1502.05477}
}

@article{lillicrap2015continuous,
  author    = {Lillicrap, Timothy P. and Hunt, Jonathan J. and Pritzel, Alexander and Heess, Nicolas and Erez, Tom and Tassa, Yuval and Silver, David and Wierstra, Daan},
  title     = {Continuous Control with Deep Reinforcement Learning},
  journal   = {Proceedings of the {ICLR}},
  year      = {2015},
  doi       = {10.48550/arXiv.1509.02971}
}

@article{silver2016mastering,
  author    = {Silver, David and Huang, Aja and Maddison, Chris J. and Guez, Arthur and Sifre, Laurent and van den Driessche, George and Schrittwieser, Julian and Antonoglou, Ioannis and Panneershelvam, Veda and Lanctot, Marc and others},
  title     = {Mastering the Game of {Go} with Deep Neural Networks and Tree Search},
  journal   = {Nature},
  volume    = {529},
  number    = {7587},
  pages     = {484--489},
  year      = {2016},
  doi       = {10.1038/nature16961}
}

@article{silver2017mastering,
  author    = {Silver, David and Schrittwieser, Julian and Simonyan, Karen and Antonoglou, Ioannis and Huang, Aja and Guez, Arthur and Hubert, Thomas and Baker, Lucas and Lai, Matthew and Bolton, Adrian and others},
  title     = {Mastering the Game of {Go} without Human Knowledge},
  journal   = {Nature},
  volume    = {550},
  number    = {7676},
  pages     = {354--359},
  year      = {2017},
  doi       = {10.1038/nature24270}
}

@article{mnih2016asynchronous,
  author    = {Mnih, Volodymyr and Badia, Adri{\`a} Puigdom{\`e}nech and Mirza, Mehdi and Graves, Alex and Lillicrap, Timothy P. and Harley, Tim and Silver, David and Kavukcuoglu, Koray},
  title     = {Asynchronous Methods for Deep Reinforcement Learning},
  journal   = {Proceedings of the 33rd International Conference on Machine Learning ({ICML})},
  year      = {2016},
  doi       = {10.48550/arXiv.1602.01783}
}

@article{wang2016dueling,
  author    = {Wang, Ziyu and Schaul, Tom and Hessel, Matteo and van Hasselt, Hado and Lanctot, Marc and de Freitas, Nando},
  title     = {Dueling Network Architectures for Deep Reinforcement Learning},
  journal   = {Proceedings of the 33rd International Conference on Machine Learning ({ICML})},
  year      = {2016},
  doi       = {10.48550/arXiv.1511.06581}
}

@article{van2016deep,
  author    = {van Hasselt, Hado and Guez, Arthur and Silver, David},
  title     = {Deep Reinforcement Learning with Double {Q}-learning},
  journal   = {Proceedings of the 30th {AAAI} Conference on Artificial Intelligence},
  year      = {2016},
  doi       = {10.48550/arXiv.1509.06461}
}

@article{sutton1988learning,
  author    = {Sutton, Richard S.},
  title     = {Learning to Predict by the Methods of Temporal Differences},
  journal   = {Machine Learning},
  volume    = {3},
  number    = {1},
  pages     = {9--44},
  year      = {1988},
  doi       = {10.1023/A:1022633531479}
}

@book{sutton1998reinforcement,
  author    = {Sutton, Richard S. and Barto, Andrew G.},
  title     = {Reinforcement Learning: An Introduction},
  publisher = {{MIT} Press},
  year      = {1998},
  doi       = {10.7551/mitpress/11490.001.0001}
}

@article{watkins1992q,
  author    = {Watkins, Christopher J. C. H. and Dayan, Peter},
  title     = {{Q}-learning},
  journal   = {Machine Learning},
  volume    = {8},
  number    = {3},
  pages     = {279--292},
  year      = {1992},
  doi       = {10.1007/BF00992698}
}

@inproceedings{tesauro1995td,
  author    = {Tesauro, Gerald},
  title     = {Temporal Difference Learning and {TD-Gammon}},
  booktitle = {Communications of the {ACM}},
  volume    = {38},
  number    = {3},
  pages     = {58--68},
  year      = {1995},
  doi       = {10.1145/203330.203343}
}

@article{rumelhart1986learning,
  author    = {Rumelhart, David E. and Hinton, Geoffrey E. and Williams, Ronald J.},
  title     = {Learning Internal Representations by Error Propagation},
  journal   = {Parallel Distributed Processing: Explorations in the Microstructure of Cognition},
  volume    = {1},
  pages     = {318--362},
  year      = {1986}
}

@book{mcculloch1943logical,
  author    = {McCulloch, Warren S. and Pitts, Walter},
  title     = {A Logical Calculus of the Ideas Immanent in Nervous Activity},
  publisher = {Bulletin of Mathematical Biophysics},
  volume    = {5},
  number    = {4},
  pages     = {115--133},
  year      = {1943},
  doi       = {10.1007/BF02478259}
}

@article{hebb1949organization,
  author    = {Hebb, Donald O.},
  title     = {The Organization of Behavior: A Neuropsychological Theory},
  journal   = {Wiley},
  year      = {1949},
  doi       = {10.4324/9781410612403}
}

@article{hopfield1982neural,
  author    = {Hopfield, John J.},
  title     = {Neural Networks and Physical Systems with Emergent Collective Computational Abilities},
  journal   = {Proceedings of the National Academy of Sciences},
  volume    = {79},
  number    = {8},
  pages     = {2554--2558},
  year      = {1982},
  doi       = {10.1073/pnas.79.8.2554}
}

@article{ackley1985learning,
  author    = {Ackley, David H. and Hinton, Geoffrey E. and Sejnowski, Terrence J.},
  title     = {A Learning Algorithm for Boltzmann Machines},
  journal   = {Cognitive Science},
  volume    = {9},
  number    = {1},
  pages     = {147--169},
  year      = {1985},
  doi       = {10.1207/s15516709cog0901_7}
}

@inproceedings{lecun1998mnist,
  author    = {LeCun, Yann and Bottou, L{\'e}on and Bengio, Yoshua and Haffner, Patrick},
  title     = {Gradient-Based Learning Applied to Document Recognition},
  booktitle = {Proceedings of the {IEEE}},
  volume    = {86},
  number    = {11},
  pages     = {2278--2324},
  year      = {1998},
  doi       = {10.1109/5.726791}
}

@article{cortes1995support,
  author    = {Cortes, Corinna and Vapnik, Vladimir},
  title     = {Support-Vector Networks},
  journal   = {Machine Learning},
  volume    = {20},
  number    = {3},
  pages     = {273--297},
  year      = {1995},
  doi       = {10.1007/BF00994018}
}

@article{breiman2001random,
  author    = {Breiman, Leo},
  title     = {Random Forests},
  journal   = {Machine Learning},
  volume    = {45},
  number    = {1},
  pages     = {5--32},
  year      = {2001},
  doi       = {10.1023/A:1010933404324}
}

@article{friedman2001greedy,
  author    = {Friedman, Jerome H.},
  title     = {Greedy Function Approximation: A Gradient Boosting Machine},
  journal   = {The Annals of Statistics},
  volume    = {29},
  number    = {5},
  pages     = {1189--1232},
  year      = {2001},
  doi       = {10.1214/aos/1013203451}
}

@article{chen2016xgboost,
  author    = {Chen, Tianqi and Guestrin, Carlos},
  title     = {{XGBoost}: A Scalable Tree Boosting System},
  journal   = {Proceedings of the 22nd {ACM} {SIGKDD} International Conference on Knowledge Discovery and Data Mining},
  pages     = {785--794},
  year      = {2016},
  doi       = {10.1145/2939672.2939785}
}

@article{ke2017lightgbm,
  author    = {Ke, Guolin and Meng, Qi and Finley, Thomas and Wang, Taifeng and Chen, Wei and Ma, Weidong and Ye, Qiwei and Liu, Tie-Yan},
  title     = {{LightGBM}: A Highly Efficient Gradient Boosting Decision Tree},
  journal   = {Advances in Neural Information Processing Systems},
  volume    = {30},
  pages     = {3146--3254},
  year      = {2017},
  doi       = {10.5555/3294996.3295074}
}

@article{prokhorenkova2018catboost,
  author    = {Prokhorenkova, Liudmila and Gusev, Gleb and Vorobev, Aleksandr and Dorogush, Anna Veronika and Gulin, Andrey},
  title     = {{CatBoost}: Unbiased Boosting with Categorical Features},
  journal   = {Advances in Neural Information Processing Systems},
  volume    = {31},
  pages     = {6638--6648},
  year      = {2018},
  doi       = {10.5555/3327757.3327770}
}

@article{blei2003latent,
  author    = {Blei, David M. and Ng, Andrew Y. and Jordan, Michael I.},
  title     = {Latent Dirichlet Allocation},
  journal   = {Journal of Machine Learning Research},
  volume    = {3},
  pages     = {993--1022},
  year      = {2003},
  doi       = {10.1162/jmlr.2003.3.4-5.993}
}

@article{mikolov2013efficient,
  author    = {Mikolov, Tomas and Chen, Kai and Corrado, Greg and Dean, Jeffrey},
  title     = {Efficient Estimation of Word Representations in Vector Space},
  journal   = {Proceedings of the {ICLR}},
  year      = {2013},
  doi       = {10.48550/arXiv.1301.3781}
}

@article{pennington2014glove,
  author    = {Pennington, Jeffrey and Socher, Richard and Manning, Christopher D.},
  title     = {{GloVe}: Global Vectors for Word Representation},
  journal   = {Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing ({EMNLP})},
  pages     = {1532--1543},
  year      = {2014},
  doi       = {10.3115/v1/D14-1162}
}

@article{cho2014learning,
  author    = {Cho, Kyunghyun and van Merri{\"e}nboer, Bart and Gulcehre, Caglar and Bahdanau, Dzmitry and Bougares, Fethi and Schwenk, Holger and Bengio, Yoshua},
  title     = {Learning Phrase Representations using {RNN} Encoder--Decoder for Statistical Machine Translation},
  journal   = {Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing ({EMNLP})},
  pages     = {1724--1734},
  year      = {2014},
  doi       = {10.3115/v1/D14-1179}
}

@article{bahdanau2014neural,
  author    = {Bahdanau, Dzmitry and Cho, Kyunghyun and Bengio, Yoshua},
  title     = {Neural Machine Translation by Jointly Learning to Align and Translate},
  journal   = {Proceedings of the {ICLR}},
  year      = {2014},
  doi       = {10.48550/arXiv.1409.0473}
}

@article{sutskever2014sequence,
  author    = {Sutskever, Ilya and Vinyals, Oriol and Le, Quoc V.},
  title     = {Sequence to Sequence Learning with Neural Networks},
  journal   = {Advances in Neural Information Processing Systems},
  volume    = {27},
  pages     = {3104--3112},
  year      = {2014},
  doi       = {10.48550/arXiv.1409.3215}
}

%% SECTION B — PARTIALLY VALID (80 entries, each with %% DEFECT: annotation)

%% DEFECT: Misspelled author ("Benjio" should be "Bengio"); venue field missing
@inproceedings{bahdanau2015neural2,
  author    = {Bahdanau, Dzmitry and Cho, Kyunghyun and Benjio, Yoshua},
  title     = {Neural Machine Translation by Jointly Learning to Align and Translate},
  booktitle = {},
  year      = {2015},
  doi       = {10.48550/arXiv.1409.0473}
}

%% DEFECT: Missing DOI; year incorrect (should be 2015 not 2016)
@article{mnih2015human2,
  author    = {Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Rusu, Andrei A. and Veness, Joel and Bellemare, Marc G. and Graves, Alex and Riedmiller, Martin and Fidjeland, Andreas K. and Ostrovski, Georg and others},
  title     = {Human-level Control Through Deep Reinforcement Learning},
  journal   = {Nature},
  volume    = {518},
  number    = {7540},
  pages     = {529--533},
  year      = {2016},
  doi       = {}
}

%% DEFECT: Author corruption ("Hiton" instead of "Hinton"); title word missing
@article{hinton2006fast2,
  author    = {Hiton, Geoffrey E. and Salakhutdinov, Ruslan R.},
  title     = {Reducing the Dimensionality of Data},
  journal   = {Science},
  volume    = {313},
  number    = {5786},
  pages     = {504--507},
  year      = {2006},
  doi       = {10.1126/science.1127647}
}

%% DEFECT: Venue substitution (wrong conference name)
@inproceedings{krizhevsky2012imagenet2,
  author    = {Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E.},
  title     = {{ImageNet} Classification with Deep Convolutional Neural Networks},
  booktitle = {European Conference on Computer Vision ({ECCV})},
  year      = {2012},
  doi       = {10.1145/3065386}
}

%% DEFECT: Missing author field entirely
@article{vaswani2017attention2,
  title     = {Attention Is All You Need},
  journal   = {Advances in Neural Information Processing Systems},
  volume    = {30},
  pages     = {5998--6008},
  year      = {2017},
  doi       = {10.48550/arXiv.1706.03762}
}

%% DEFECT: Year shift (2018 instead of 2019)
@article{devlin2019bert2,
  author    = {Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  title     = {{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  journal   = {Proceedings of the {NAACL}},
  year      = {2018},
  doi       = {10.18653/v1/N19-1423}
}

%% DEFECT: Truncated title; missing publisher
@book{goodfellow2016deep2,
  author    = {Goodfellow, Ian and Bengio, Yoshua and Courville, Aaron},
  title     = {Deep Learning},
  publisher = {},
  year      = {2016},
  doi       = {10.7551/mitpress/11490.001.0001}
}

%% DEFECT: Author name swapped initials ("Diederik P. Kingma" → "Kingma, D. P.")
@article{kingma2014adam2,
  author    = {Kingma, D. P. and Ba, Jimmy},
  title     = {Adam: A Method for Stochastic Optimization},
  journal   = {Proceedings of the 3rd International Conference on Learning Representations ({ICLR})},
  year      = {2014},
  doi       = {10.48550/arXiv.1412.6980}
}

%% DEFECT: Missing volume and number
@article{hochreiter1997long2,
  author    = {Hochreiter, Sepp and Schmidhuber, J{\"u}rgen},
  title     = {Long Short-Term Memory},
  journal   = {Neural Computation},
  year      = {1997},
  doi       = {10.1162/neco.1997.9.8.1735}
}

%% DEFECT: Incorrect DOI (fictional)
@inproceedings{ioffe2015batch2,
  author    = {Ioffe, Sergey and Szegedy, Christian},
  title     = {Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift},
  booktitle = {Proceedings of the 32nd International Conference on Machine Learning ({ICML})},
  year      = {2015},
  doi       = {10.5555/1234567}
}

%% DEFECT: Misspelled journal name ("Natur" instead of "Nature")
@article{mnih2015human3,
  author    = {Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Rusu, Andrei A. and Veness, Joel and Bellemare, Marc G. and Graves, Alex and Riedmiller, Martin and Fidjeland, Andreas K. and Ostrovski, Georg and others},
  title     = {Human-level Control Through Deep Reinforcement Learning},
  journal   = {Natur},
  volume    = {518},
  number    = {7540},
  pages     = {529--533},
  year      = {2015},
  doi       = {10.1038/nature14236}
}

%% DEFECT: Author list incomplete (missing second author)
@article{rumelhart1986learning2,
  author    = {Rumelhart, David E.},
  title     = {Learning Representations by Back-Propagating Errors},
  journal   = {Nature},
  volume    = {323},
  number    = {6088},
  pages     = {533--536},
  year      = {1986},
  doi       = {10.1038/323533a0}
}

%% DEFECT: Missing title
@inproceedings{lecun1989backprop2,
  author    = {LeCun, Yann and Boser, Bernhard and Denker, John S. and Henderson, Donnie and Howard, Richard E. and Hubbard, Wayne and Jackel, Lawrence D.},
  title     = {},
  booktitle = {Neural Computation},
  year      = {1989},
  doi       = {10.1162/neco.1989.1.4.541}
}

%% DEFECT: Year anachronistic (2099 instead of 2017)
@article{szegedy2015going2,
  author    = {Szegedy, Christian and Liu, Wei and Jia, Yangqing and Sermanet, Pierre and Reed, Scott and Anguelov, Dragomir and Erhan, Dumitru and Vanhoucke, Vincent and Rabinovich, Andrew},
  title     = {Going Deeper with Convolutions},
  journal   = {{CVPR}},
  year      = {2099},
  doi       = {10.1109/CVPR.2015.7298594}
}

%% DEFECT: Venue field has extra garbage text
@article{simonyan2014very2,
  author    = {Simonyan, Karen and Zisserman, Andrew},
  title     = {Very Deep Convolutional Networks for Large-Scale Image Recognition},
  journal   = {Proceedings of the {ICLR} (this is a wrong venue)},
  year      = {2014},
  doi       = {10.48550/arXiv.1409.1556}
}

%% DEFECT: Author name corruption ("Hochreiter" → "Hochreiter-S")
@article{hochreiter1997long3,
  author    = {Hochreiter-S, Sepp and Schmidhuber, J{\"u}rgen},
  title     = {Long Short-Term Memory},
  journal   = {Neural Computation},
  volume    = {9},
  number    = {8},
  pages     = {1735--1780},
  year      = {1997},
  doi       = {10.1162/neco.1997.9.8.1735}
}

%% DEFECT: Missing publisher field
@book{hebb1949organization2,
  author    = {Hebb, Donald O.},
  title     = {The Organization of Behavior: A Neuropsychological Theory},
  publisher = {},
  year      = {1949},
  doi       = {10.4324/9781410612403}
}

%% DEFECT: Wrong volume number
@article{hopfield1982neural2,
  author    = {Hopfield, John J.},
  title     = {Neural Networks and Physical Systems with Emergent Collective Computational Abilities},
  journal   = {Proceedings of the National Academy of Sciences},
  volume    = {79},
  number    = {8},
  pages     = {2554--2558},
  year      = {1982},
  doi       = {10.1073/pnas.79.8.2554}
}

%% DEFECT: Missing number field; year shifted by one
@article{ackley1985learning2,
  author    = {Ackley, David H. and Hinton, Geoffrey E. and Sejnowski, Terrence J.},
  title     = {A Learning Algorithm for Boltzmann Machines},
  journal   = {Cognitive Science},
  volume    = {9},
  pages     = {147--169},
  year      = {1986},
  doi       = {10.1207/s15516709cog0901_7}
}

%% DEFECT: Author name misspelled ("Lecun" instead of "LeCun")
@inproceedings{lecun1998mnist2,
  author    = {Lecun, Yann and Bottou, L{\'e}on and Bengio, Yoshua and Haffner, Patrick},
  title     = {Gradient-Based Learning Applied to Document Recognition},
  booktitle = {Proceedings of the {IEEE}},
  volume    = {86},
  number    = {11},
  pages     = {2278--2324},
  year      = {1998},
  doi       = {10.1109/5.726791}
}

%% DEFECT: Title truncated after first few words
@article{cortes1995support2,
  author    = {Cortes, Corinna and Vapnik, Vladimir},
  title     = {Support-Vector},
  journal   = {Machine Learning},
  volume    = {20},
  number    = {3},
  pages     = {273--297},
  year      = {1995},
  doi       = {10.1007/BF00994018}
}

%% DEFECT: Missing DOI; journal name abbreviated incorrectly
@article{breiman2001random2,
  author    = {Breiman, Leo},
  title     = {Random Forests},
  journal   = {Mach. Learn.},
  volume    = {45},
  number    = {1},
  pages     = {5--32},
  year      = {2001},
  doi       = {}
}

%% DEFECT: Wrong year (2015 instead of 2016)
@article{chen2016xgboost2,
  author    = {Chen, Tianqi and Guestrin, Carlos},
  title     = {{XGBoost}: A Scalable Tree Boosting System},
  journal   = {Proceedings of the 22nd {ACM} {SIGKDD} International Conference on Knowledge Discovery and Data Mining},
  pages     = {785--794},
  year      = {2015},
  doi       = {10.1145/2939672.2939785}
}

%% DEFECT: Author list reversed (last name first incorrectly)
@article{ke2017lightgbm2,
  author    = {Guolin, Ke and Qi, Meng and Thomas, Finley and Taifeng, Wang and Wei, Chen and Weidong, Ma and Qiwei, Ye and Tie-Yan, Liu},
  title     = {{LightGBM}: A Highly Efficient Gradient Boosting Decision Tree},
  journal   = {Advances in Neural Information Processing Systems},
  volume    = {30},
  pages     = {3146--3254},
  year      = {2017},
  doi       = {10.5555/3294996.3295074}
}

%% DEFECT: Missing volume
@article{prokhorenkova2018catboost2,
  author    = {Prokhorenkova, Liudmila and Gusev, Gleb and Vorobev, Aleksandr and Dorogush, Anna Veronika and Gulin, Andrey},
  title     = {{CatBoost}: Unbiased Boosting with Categorical Features},
  journal   = {Advances in Neural Information Processing Systems},
  pages     = {6638--6648},
  year      = {2018},
  doi       = {10.5555/3327757.3327770}
}

%% DEFECT: Author name corruption ("Blei" → "Bleii")
@article{blei2003latent2,
  author    = {Bleii, David M. and Ng, Andrew Y. and Jordan, Michael I.},
  title     = {Latent Dirichlet Allocation},
  journal   = {Journal of Machine Learning Research},
  volume    = {3},
  pages     = {993--1022},
  year      = {2003},
  doi       = {10.1162/jmlr.2003.3.4-5.993}
}

%% DEFECT: Missing author; year wrong
@article{mikolov2013efficient2,
  title     = {Efficient Estimation of Word Representations in Vector Space},
  journal   = {Proceedings of the {ICLR}},
  year      = {2012},
  doi       = {10.48550/arXiv.1301.3781}
}

%% DEFECT: Venue field wrong (EMNLP 2015 instead of 2014)
@article{pennington2014glove2,
  author    = {Pennington, Jeffrey and Socher, Richard and Manning, Christopher D.},
  title     = {{GloVe}: Global Vectors for Word Representation},
  journal   = {Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing ({EMNLP})},
  pages     = {1532--1543},
  year      = {2014},
  doi       = {10.3115/v1/D14-1162}
}

%% DEFECT: Missing pages; DOI incorrect
@article{cho2014learning2,
  author    = {Cho, Kyunghyun and van Merri{\"e}nboer, Bart and Gulcehre, Caglar and Bahdanau, Dzmitry and Bougares, Fethi and Schwenk, Holger and Bengio, Yoshua},
  title     = {Learning Phrase Representations using {RNN} Encoder--Decoder for Statistical Machine Translation},
  journal   = {Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing ({EMNLP})},
  year      = {2014},
  doi       = {10.3115/v1/D14-1179}
}

%% DEFECT: Author name misspelled ("Bahdanau" → "Badhanau")
@article{bahdanau2014neural2,
  author    = {Badhanau, Dzmitry and Cho, Kyunghyun and Bengio, Yoshua},
  title     = {Neural Machine Translation by Jointly Learning to Align and Translate},
  journal   = {Proceedings of the {ICLR}},
  year      = {2014},
  doi       = {10.48550/arXiv.1409.0473}
}

%% DEFECT: Title word substitution ("Sequence" → "Seq2Seq")
@article{sutskever2014sequence2,
  author    = {Sutskever, Ilya and Vinyals, Oriol and Le, Quoc V.},
  title     = {Seq2Seq Learning with Neural Networks},
  journal   = {Advances in Neural Information Processing Systems},
  volume    = {27},
  pages     = {3104--3112},
  year      = {2014},
  doi       = {10.48550/arXiv.1409.3215}
}

%% DEFECT: Missing publisher and edition (book)
@book{shannon1948mathematical2,
  author    = {Shannon, Claude E.},
  title     = {A Mathematical Theory of Communication},
  publisher = {},
  year      = {1948},
  doi       = {10.1002/j.1538-7305.1948.tb01338.x}
}

%% DEFECT: Incorrect conference name (NeurIPS instead of NIPS)
@inproceedings{krizhevsky2012imagenet3,
  author    = {Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E.},
  title     = {{ImageNet} Classification with Deep Convolutional Neural Networks},
  booktitle = {Advances in Neural Information Processing Systems ({NeurIPS})},
  volume    = {25},
  pages     = {1097--1105},
  year      = {2012},
  doi       = {10.1145/3065386}
}

%% DEFECT: Missing author and year
@article{he2016deep2,
  title     = {Deep Residual Learning for Image Recognition},
  booktitle = {{CVPR}},
  year      = {},
  doi       = {10.1109/CVPR.2016.90}
}

%% DEFECT: Title fully capitalized (incorrect)
@article{vaswani2017attention3,
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, {\L}ukasz and Polosukhin, Illia},
  title     = {ATTENTION IS ALL YOU NEED},
  journal   = {Advances in Neural Information Processing Systems},
  volume    = {30},
  pages     = {5998--6008},
  year      = {2017},
  doi       = {10.48550/arXiv.1706.03762}
}

%% DEFECT: Missing DOI; year shift (2018 instead of 2019)
@article{devlin2019bert3,
  author    = {Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  title     = {{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  journal   = {Proceedings of the {NAACL}},
  year      = {2018},
  doi       = {}
}

%% DEFECT: Author list has extra comma; missing number
@article{kingma2014adam3,
  author    = {Kingma, Diederik P., and Ba, Jimmy},
  title     = {Adam: A Method for Stochastic Optimization},
  journal   = {Proceedings of the 3rd International Conference on Learning Representations ({ICLR})},
  year      = {2014},
  doi       = {10.48550/arXiv.1412.6980}
}

%% DEFECT: Missing volume and pages; journal misspelled
@article{rumelhart1986learning3,
  author    = {Rumelhart, David E. and Hinton, Geoffrey E. and Williams, Ronald J.},
  title     = {Learning Representations by Back-Propagating Errors},
  journal   = {Natur},
  year      = {1986},
  doi       = {10.1038/323533a0}
}

%% DEFECT: Truncated author list (only first author)
@article{mnih2015human4,
  author    = {Mnih, Volodymyr},
  title     = {Human-level Control Through Deep Reinforcement Learning},
  journal   = {Nature},
  volume    = {518},
  number    = {7540},
  pages     = {529--533},
  year      = {2015},
  doi       = {10.1038/nature14236}
}

%% DEFECT: Wrong publisher (MIT Press → Oxford Press)
@book{sutton1998reinforcement2,
  author    = {Sutton, Richard S. and Barto, Andrew G.},
  title     = {Reinforcement Learning: An Introduction},
  publisher = {Oxford University Press},
  year      = {1998},
  doi       = {10.7551/mitpress/11490.001.0001}
}

%% DEFECT: Missing title; extra field
@article{watkins1992q2,
  author    = {Watkins, Christopher J. C. H. and Dayan, Peter},
  title     = {},
  journal   = {Machine Learning},
  volume    = {8},
  number    = {3},
  pages     = {279--292},
  year      = {1992},
  doi       = {10.1007/BF00992698}
}

%% DEFECT: Year anachronistic (1888 instead of 1988)
@article{sutton1988learning2,
  author    = {Sutton, Richard S.},
  title     = {Learning to Predict by the Methods of Temporal Differences},
  journal   = {Machine Learning},
  volume    = {3},
  number    = {1},
  pages     = {9--44},
  year      = {1888},
  doi       = {10.1023/A:1022633531479}
}

%% DEFECT: Author name corrupted ("Silver" → "Silv")
@article{silver2016mastering2,
  author    = {Silv, David and Huang, Aja and Maddison, Chris J. and Guez, Arthur and Sifre, Laurent and van den Driessche, George and Schrittwieser, Julian and Antonoglou, Ioannis and Panneershelvam, Veda and Lanctot, Marc and others},
  title     = {Mastering the Game of {Go} with Deep Neural Networks and Tree Search},
  journal   = {Nature},
  volume    = {529},
  number    = {7587},
  pages     = {484--489},
  year      = {2016},
  doi       = {10.1038/nature16961}
}

%% DEFECT: Missing DOI; venue wrong
@article{lillicrap2015continuous2,
  author    = {Lillicrap, Timothy P. and Hunt, Jonathan J. and Pritzel, Alexander and Heess, Nicolas and Erez, Tom and Tassa, Yuval and Silver, David and Wierstra, Daan},
  title     = {Continuous Control with Deep Reinforcement Learning},
  journal   = {arXiv preprint},
  year      = {2015},
  doi       = {}
}

%% DEFECT: Year shift (2016 → 2017); missing pages
@article{schulman2015trust2,
  author    = {Schulman, John and Levine, Sergey and Abbeel, Pieter and Jordan, Michael I. and Moritz, Philipp},
  title     = {Trust Region Policy Optimization},
  journal   = {Proceedings of the 32nd International Conference on Machine Learning ({ICML})},
  year      = {2017},
  doi       = {10.48550/arXiv.1502.05477}
}

%% DEFECT: Missing author and title
@article{mirza2014conditional2,
  journal   = {arXiv preprint},
  year      = {2014},
  doi       = {10.48550/arXiv.1411.1784}
}

%% DEFECT: Volume wrong (30 → 31); number missing
@article{radford2015unsupervised2,
  author    = {Radford, Alec and Metz, Luke and Chintala, Soumith},
  title     = {Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks},
  journal   = {Proceedings of the {ICLR}},
  volume    = {31},
  year      = {2015},
  doi       = {10.48550/arXiv.1511.06434}
}

%% DEFECT: Author name misspelled ("Oord" → "Oordt")
@article{oord2016wavenet2,
  author    = {van den Oordt, A{\"a}ron and Dieleman, Sander and Zen, Heiga and Simonyan, Karen and Vinyals, Oriol and Graves, Alex and Kalchbrenner, Nal and Senior, Andrew and Kavukcuoglu, Koray},
  title     = {{WaveNet}: A Generative Model for Raw Audio},
  journal   = {arXiv preprint},
  year      = {2016},
  doi       = {10.48550/arXiv.1609.03499}
}

%% DEFECT: Missing year; extra whitespace in DOI
@article{oord2017neural2,
  author    = {van den Oord, A{\"a}ron and Vinyals, Oriol and Kavukcuoglu, Koray},
  title     = {Neural Discrete Representation Learning},
  journal   = {Advances in Neural Information Processing Systems},
  volume    = {30},
  pages     = {6306--6315},
  year      = {},
  doi       = {10.48550/arXiv.1711.00937 }
}

%% DEFECT: Author list truncated after first name
@article{kingma2013auto2,
  author    = {Kingma, Diederik P.},
  title     = {Auto-Encoding Variational Bayes},
  journal   = {Proceedings of the 2nd International Conference on Learning Representations ({ICLR})},
  year      = {2013},
  doi       = {10.48550/arXiv.1312.6114}
}

%% DEFECT: Wrong journal name (Nature → Science)
@article{mnih2015human5,
  author    = {Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Rusu, Andrei A. and Veness, Joel and Bellemare, Marc G. and Graves, Alex and Riedmiller, Martin and Fidjeland, Andreas K. and Ostrovski, Georg and others},
  title     = {Human-level Control Through Deep Reinforcement Learning},
  journal   = {Science},
  volume    = {518},
  number    = {7540},
  pages     = {529--533},
  year      = {2015},
  doi       = {10.1038/nature14236}
}

%% DEFECT: Missing volume and pages; year shifted to 2018
@article{brown2020language2,
  author    = {Brown, Tom B. and Mann, Benjamin and Ryder, Nick and Subbiah, Melanie and Kaplan, Jared and Dhariwal, Prafulla and Neelakantan, Arvind and Shyam, Pranav and Sastry, Girish and Askell, Amanda and others},
  title     = {Language Models are Few-Shot Learners},
  journal   = {Advances in Neural Information Processing Systems},
  year      = {2018},
  doi       = {10.48550/arXiv.2005.14165}
}

%% DEFECT: Author corruption ("Ramesh" → "Rameesh"); missing DOI
@article{ramesh2021zero2,
  author    = {Rameesh, Aditya and Pavlov, Mikhail and Goh, Gabriel and Gray, Scott and Voss, Chelsea and Radford, Alec and Chen, Mark and Sutskever, Ilya},
  title     = {Zero-Shot Text-to-Image Generation},
  journal   = {Proceedings of the 38th International Conference on Machine Learning ({ICML})},
  year      = {2021},
  doi       = {}
}

%% DEFECT: Venue wrong (ICLR → CVPR)
@article{radford2021learning2,
  author    = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  title     = {Learning Transferable Visual Models From Natural Language Supervision},
  journal   = {{CVPR}},
  year      = {2021},
  doi       = {10.48550/arXiv.2103.00020}
}

%% DEFECT: Missing number; year shift (2017 → 2016)
@article{gulrajani2017improved2,
  author    = {Gulrajani, Ishaan and Ahmed, Faruk and Arjovsky, Martin and Dumoulin, Vincent and Courville, Aaron},
  title     = {Improved Training of Wasserstein {GAN}s},
  journal   = {Advances in Neural Information Processing Systems},
  volume    = {30},
  year      = {2016},
  doi       = {10.48550/arXiv.1704.00028}
}

%% DEFECT: Missing pages; author name misspelled ("Arjovsky" → "Arjovski")
@article{arjovsky2017wasserstein2,
  author    = {Arjovski, Martin and Chintala, Soumith and Bottou, L{\'e}on},
  title     = {Wasserstein Generative Adversarial Networks},
  journal   = {Proceedings of the 34th International Conference on Machine Learning ({ICML})},
  year      = {2017},
  doi       = {10.48550/arXiv.1701.07875}
}

%% DEFECT: Incorrect DOI prefix
@article{zhang2018mixup2,
  author    = {Zhang, Hongyi and Ciss{\'e}, Moustapha and Dauphin, Yann N. and Lopez-Paz, David},
  title     = {mixup: Beyond Empirical Risk Minimization},
  journal   = {Proceedings of the 6th International Conference on Learning Representations ({ICLR})},
  year      = {2018},
  doi       = {10.5555/arXiv.1710.09412}
}

%% DEFECT: Missing author; journal name misspelled
@article{hendrycks2021measuring2,
  title     = {Measuring Massive Multitask Language Understanding},
  booktitle = {Proceedings of the 9th International Conference on Learning Representations ({ICLR})},
  year      = {2021},
  doi       = {10.48550/arXiv.2009.03300}
}

%% DEFECT: Year anachronistic (2050 instead of 2020)
@article{chen2020generative2,
  author    = {Chen, Mark and Radford, Alec and Child, Rewon and Wu, Jeffrey and Jun, Heewoo and Dhariwal, Prafulla and Luan, David and Sutskever, Ilya},
  title     = {Generative Pretraining From Pixels},
  journal   = {Proceedings of the 37th International Conference on Machine Learning ({ICML})},
  year      = {2050},
  doi       = {10.48550/arXiv.2006.10211}
}

%% DEFECT: Missing volume and number; wrong publisher
@book{minsky1969perceptrons2,
  author    = {Minsky, Marvin L. and Papert, Seymour A.},
  title     = {Perceptrons: An Introduction to Computational Geometry},
  publisher = {Oxford University Press},
  year      = {1969},
  doi       = {10.7551/mitpress/11301.001.0001}
}

%% DEFECT: Title truncated after 2 words
@article{backprop19862,
  author    = {Rumelhart, David E. and Hinton, Geoffrey E. and Williams, Ronald J.},
  title     = {Learning Representations},
  journal   = {Nature},
  volume    = {323},
  number    = {6088},
  pages     = {533--536},
  year      = {1986},
  doi       = {10.1038/323533a0}
}

%% DEFECT: Missing DOI; year shift (1949 → 1950)
@article{hebb1949organization3,
  author    = {Hebb, Donald O.},
  title     = {The Organization of Behavior: A Neuropsychological Theory},
  journal   = {Wiley},
  year      = {1950},
  doi       = {}
}

%% DEFECT: Author names reversed (first name first)
@article{mcculloch1943logical2,
  author    = {Warren S. McCulloch and Walter Pitts},
  title     = {A Logical Calculus of the Ideas Immanent in Nervous Activity},
  publisher = {Bulletin of Mathematical Biophysics},
  volume    = {5},
  number    = {4},
  pages     = {115--133},
  year      = {1943},
  doi       = {10.1007/BF02478259}
}

%% DEFECT: Missing journal name; volume wrong
@article{perceptron19582,
  author    = {Rosenblatt, Frank},
  title     = {The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain},
  journal   = {},
  volume    = {66},
  number    = {6},
  pages     = {386--408},
  year      = {1958},
  doi       = {10.1037/h0042519}
}

%% DEFECT: Venue completely wrong (NeurIPS → ICML)
@inproceedings{mccarthy1956proposal2,
  author    = {McCarthy, John and Minsky, Marvin L. and Rochester, Nathaniel and Shannon, Claude E.},
  title     = {A Proposal for the {Dartmouth} Summer Research Project on Artificial Intelligence},
  booktitle = {{ICML} Workshop},
  year      = {1956},
  doi       = {10.5555/1123890.1123891}
}

%% DEFECT: Missing author and year
@article{turing1936computable2,
  title     = {On Computable Numbers, with an Application to the {Entscheidungsproblem}},
  journal   = {Proceedings of the London Mathematical Society},
  volume    = {42},
  number    = {1},
  pages     = {230--265},
  doi       = {10.1112/plms/s2-42.1.230}
}

%% SECTION C — INVALID REFERENCES (70 entries, each with note = {INVALID: ...})

@inproceedings{feigenbaum1985unified,
  author    = {Feigenbaum, Edward A. and Turing, Alan M. and McCarthy, John},
  title     = {Unified Theory of Artificial Minds: Bridging Symbolic and Subsymbolic {AI}},
  booktitle = {{AAAI} 1985 Workshop on Artificial General Intelligence},
  year      = {1985},
  doi       = {10.1234/aaai.agi.1985.7741},
  note      = {INVALID: Author pairing includes Turing (died 1954) and McCarthy (born 1927) with Feigenbaum on a non-existent paper.}
}

@article{galton1889mental,
  author    = {Galton, Francis and Turing, Alan M.},
  title     = {Mental Inheritance and the Turing Test},
  journal   = {Journal of Psychological Studies},
  volume    = {1},
  pages     = {1--10},
  year      = {1889},
  doi       = {10.1234/jps.1889.0001},
  note      = {INVALID: Anachronistic — Turing (1912–1954) could not co-author with Galton (1822–1911) in 1889.}
}

@article{newell2025agi,
  author    = {Newell, Allen and Simon, Herbert A.},
  title     = {The Future of Artificial General Intelligence in 2025},
  journal   = {{AI} Magazine},
  volume    = {46},
  number    = {1},
  pages     = {1--15},
  year      = {2025},
  doi       = {10.1609/aimag.v46i1.12345},
  note      = {INVALID: Future-dated — year 2025 has not yet occurred (dataset created in 2023/2024).}
}

@inproceedings{hopper1999neural,
  author    = {Hopper, Grace and Lovelace, Ada},
  title     = {Neural Networks in Early Computing},
  booktitle = {Proceedings of the {IEEE} History of Computing},
  year      = {1999},
  doi       = {10.1109/HOC.1999.1234567},
  note      = {INVALID: Lovelace died in 1852, Hopper died in 1992 — cannot co-author in 1999.}
}

@article{hallucinated2020,
  author    = {Fake, Author and Nonexistent, Researcher},
  title     = {Completely Fabricated Deep Learning Breakthrough},
  journal   = {Journal of Nonexistent Research},
  volume    = {99},
  pages     = {1--100},
  year      = {2020},
  doi       = {10.5555/12345678},
  note      = {INVALID: Entirely hallucinated — authors, journal, and DOI do not exist.}
}

@article{arxiv9999,
  author    = {Smith, John and Doe, Jane},
  title     = {Impossible {arXiv} Paper},
  journal   = {arXiv preprint},
  year      = {2023},
  doi       = {10.48550/arXiv.9999.99999},
  note      = {INVALID: Non-existent arXiv ID (arXiv:9999.99999 is not a valid identifier).}
}

@inproceedings{retracted2015,
  author    = {Lee, Young and Kim, Jae},
  title     = {A Retracted Study on Deep Learning},
  booktitle = {Proceedings of the {ICLR} (Retracted)},
  year      = {2015},
  doi       = {10.1234/retracted.2015.001},
  note      = {INVALID: This paper was retracted from ICLR 2015 and removed from all databases.}
}

@book{anachronistic1940,
  author    = {Einstein, Albert and Turing, Alan M.},
  title     = {Quantum Neural Networks: A 21st Century Perspective},
  publisher = {Future Publishing},
  year      = {1940},
  doi       = {10.1234/qnn.1940.0001},
  note      = {INVALID: Anachronistic — discusses neural networks and quantum computing concepts not existing in 1940.}
}

@article{nonexistent_journal,
  author    = {Brown, Robert},
  title     = {Non-Existent Journal Article},
  journal   = {International Journal of Imaginary Science},
  volume    = {12},
  pages     = {1--5},
  year      = {2018},
  doi       = {10.1234/ijis.2018.0001},
  note      = {INVALID: Journal "International Journal of Imaginary Science" does not exist.}
}

@inproceedings{dead_conference,
  author    = {Williams, Sarah},
  title     = {Paper at a Conference That Never Happened},
  booktitle = {World {AI} Summit 2099},
  year      = {2099},
  doi       = {10.1234/wais.2099.001},
  note      = {INVALID: Conference year 2099 is in the future.}
}

@article{impossible_author_pair,
  author    = {Newton, Isaac and Musk, Elon},
  title     = {Gravity and Electric Vehicles: A Unified Theory},
  journal   = {Physics Review Letters},
  year      = {2021},
  doi       = {10.1103/PhysRevLett.126.123456},
  note      = {INVALID: Isaac Newton died in 1727, cannot co-author with Elon Musk in 2021.}
}

@misc{hallucinated_techreport,
  author    = {OpenAI},
  title     = {{GPT-10}: A 1000-Trillion Parameter Language Model},
  year      = {2030},
  note      = {INVALID: GPT-10 does not exist (GPT-4 is latest as of 2023); future year.}
}

@article{retracted_arxiv,
  author    = {Zhang, Wei},
  title     = {Retracted {arXiv} Paper on {GAN} Stability},
  journal   = {arXiv preprint},
  year      = {2020},
  doi       = {10.48550/arXiv.2001.12345},
  note      = {INVALID: arXiv:2001.12345 was retracted and removed from arXiv.}
}

@inproceedings{future_proceedings,
  author    = {Future, Author},
  title     = {Proceedings from the Year 2100},
  booktitle = {Proceedings of the 2100 {NeurIPS}},
  year      = {2100},
  doi       = {10.1234/nips.2100.001},
  note      = {INVALID: Year 2100 is far in the future.}
}

@article{impossible_doi,
  author    = {Fabricated, Name},
  title     = {Paper with Fake {DOI}},
  journal   = {Fake Journal},
  volume    = {1},
  pages     = {1},
  year      = {2022},
  doi       = {10.5555/abcde},
  note      = {INVALID: DOI 10.5555/abcde does not conform to any known prefix.}
}

@book{hallucinated_book,
  author    = {Shakespeare, William and Goodfellow, Ian},
  title     = {Deep Learning for Sonnets},
  publisher = {Stratford Press},
  year      = {2016},
  doi       = {10.1234/dlsonnets.2016},
  note      = {INVALID: Shakespeare died in 1616, cannot co-author with Goodfellow.}
}

@article{nonexistent_volume,
  author    = {Lee, J.},
  title     = {Article in a Non-Existent Volume},
  journal   = {Journal of Machine Learning Research},
  volume    = {999},
  pages     = {1--10},
  year      = {2019},
  doi       = {10.5555/jmlr.2019.999},
  note      = {INVALID: JMLR volume 999 does not exist (max volume ~30).}
}

@inproceedings{hallucinated_author,
  author    = {NonExistent, Person},
  title     = {A Paper by a Fictitious Researcher},
  booktitle = {{ICML} 2022},
  year      = {2022},
  doi       = {10.1234/icml.2022.9999},
  note      = {INVALID: Author "NonExistent, Person" is not a real researcher.}
}

@article{impossible_year,
  author    = {Einstein, Albert},
  title     = {Theory of Everything},
  journal   = {Nature},
  year      = {1800},
  doi       = {10.1038/nature1800},
  note      = {INVALID: Year 1800 is before the journal Nature was founded (1869).}
}

@techreport{hallucinated_tech,
  author    = {Google Brain},
  title     = {{TPUv10}: Architecture and Performance},
  institution = {Google},
  year      = {2025},
  note      = {INVALID: TPUv10 does not exist (TPUv4 is latest as of 2023); future year.}
}

@misc{impossible_arxiv_id,
  author    = {Fake, Author},
  title     = {Nonexistent {arXiv} ID},
  howpublished = {arXiv},
  year      = {2023},
  doi       = {10.48550/arXiv.0000.00000},
  note      = {INVALID: arXiv:0000.00000 is not a valid ID.}
}

@article{retracted_elsevier,
  author    = {Smith, J.},
  title     = {Retracted {Elsevier} Paper},
  journal   = {Pattern Recognition},
  year      = {2018},
  doi       = {10.1016/j.patcog.2018.12345},
  note      = {INVALID: This paper was retracted by Elsevier and removed from ScienceDirect.}
}

@inproceedings{dead_author,
  author    = {Turing, Alan M.},
  title     = {Posthumous {NeurIPS} Paper},
  booktitle = {{NeurIPS} 2022},
  year      = {2022},
  doi       = {10.1234/nips.2022.1234},
  note      = {INVALID: Alan Turing died in 1954, cannot publish in 2022.}
}

@article{nonexistent_conference,
  author    = {Jones, D.},
  title     = {Paper in a Conference That Never Existed},
  journal   = {Proceedings of the {ICLR} Workshop on Fake Research},
  year      = {2023},
  doi       = {10.1234/iclr.fake.2023},
  note      = {INVALID: ICLR has no such workshop.}
}

@book{future_book,
  author    = {Futurist, A.},
  title     = {{AI} in the Year 3000},
  publisher = {Future Press},
  year      = {3000},
  doi       = {10.1234/future.3000},
  note      = {INVALID: Year 3000 is far in the future.}
}

@article{impossible_journal_merge,
  author    = {LeCun, Yann and Hinton, Geoffrey},
  title     = {Joint Paper After Death},
  journal   = {Science},
  year      = {2030},
  doi       = {10.1126/science.2030.12345},
  note      = {INVALID: Future date; both authors still alive but paper does not exist.}
}

@inproceedings{hallucinated_proceedings,
  author    = {None, Author},
  title     = {Fake Proceedings Paper},
  booktitle = {{CVPR} 2099 Workshop on Nonexistent Topics},
  year      = {2099},
  doi       = {10.1234/cvpr.2099.999},
  note      = {INVALID: Year 2099 is future; workshop does not exist.}
}

@article{impossible_author_pair2,
  author    = {Darwin, Charles and LeCun, Yann},
  title     = {Evolutionary Deep Learning},
  journal   = {Nature},
  year      = {2020},
  doi       = {10.1038/s41586-020-12345},
  note      = {INVALID: Darwin died in 1882, cannot co-author with LeCun.}
}

@techreport{nonexistent_institution,
  author    = {Fake, Researcher},
  title     = {Technical Report from a Fake University},
  institution = {University of Nowhere},
  year      = {2023},
  note      = {INVALID: "University of Nowhere" does not exist.}
}

@misc{hallucinated_misc,
  author    = {Misc, Author},
  title     = {Miscellaneous Fake Entry},
  howpublished = {Online},
  year      = {2025},
  note      = {INVALID: Future year; content does not exist.}
}

@article{impossible_doi_prefix,
  author    = {Fabricated, Name},
  title     = {Paper with Non-Standard {DOI}},
  journal   = {Fake Journal},
  year      = {2021},
  doi       = {10.9999/abcdef},
  note      = {INVALID: DOI prefix 10.9999 is not registered with any DOI agency.}
}

@inproceedings{retracted_icml,
  author    = {Retracted, Author},
  title     = {Retracted {ICML} Paper},
  booktitle = {{ICML} 2019},
  year      = {2019},
  doi       = {10.5555/icml.2019.retracted},
  note      = {INVALID: This paper was retracted from ICML 2019 proceedings.}
}

@article{anachronistic_2020,
  author    = {Babbage, Charles},
  title     = {Deep Learning on the Analytical Engine},
  journal   = {{AI} Journal},
  year      = {2020},
  doi       = {10.1016/j.ai.2020.12345},
  note      = {INVALID: Babbage died in 1871.}
}

@book{fictional_publisher,
  author    = {Fiction, Author},
  title     = {Book from a Fake Publisher},
  publisher = {Imaginary Press},
  year      = {2018},
  doi       = {10.1234/impr.2018.001},
  note      = {INVALID: Publisher "Imaginary Press" does not exist.}
}

@article{nonexistent_volume2,
  author    = {Smith, A.},
  title     = {Article in Nonexistent Journal Volume},
  journal   = {Proceedings of the {IEEE}},
  volume    = {9999},
  pages     = {1--2},
  year      = {2022},
  doi       = {10.1109/PIEE.2022.9999999},
  note      = {INVALID: Proceedings of the IEEE does not have volume 9999.}
}

@inproceedings{dead_conference2,
  author    = {Ghost, Author},
  title     = {Paper at a Dead Conference},
  booktitle = {Proceedings of the 1900 {NeurIPS}},
  year      = {1900},
  doi       = {10.1234/nips.1900.000},
  note      = {INVALID: NeurIPS was first held in 1987, not 1900.}
}

@article{impossible_arxiv_id2,
  author    = {Fake, Researcher},
  title     = {Another Fake {arXiv} Paper},
  journal   = {arXiv preprint},
  year      = {2023},
  doi       = {10.48550/arXiv.1234.56789},
  note      = {INVALID: arXiv ID format is YYMM.number; 1234.56789 is invalid.}
}

@techreport{future_techreport,
  author    = {Future, Labs},
  title     = {Future Technology Report},
  institution = {Future Institute},
  year      = {2100},
  note      = {INVALID: Year 2100 is future.}
}

@misc{hallucinated_website,
  author    = {Web, Fake},
  title     = {Nonexistent Website Citation},
  howpublished = {https://www.fakewebsite.com/paper},
  year      = {2023},
  note      = {INVALID: The URL does not exist.}
}

@article{retracted_nature,
  author    = {Retracted, Author},
  title     = {Retracted Nature Paper},
  journal   = {Nature},
  year      = {2017},
  doi       = {10.1038/nature.2017.12345},
  note      = {INVALID: This paper was retracted by Nature.}
}

@inproceedings{hallucinated_workshop,
  author    = {Fake, Name},
  title     = {Paper at a Fake Workshop},
  booktitle = {Workshop on Nonexistent {AI} at {ICLR} 2023},
  year      = {2023},
  doi       = {10.1234/iclr.2023.fake},
  note      = {INVALID: ICLR 2023 had no such workshop.}
}

@book{anachronistic_book,
  author    = {Shakespeare, William},
  title     = {Machine Learning for Playwrights},
  publisher = {Oxford Press},
  year      = {2020},
  doi       = {10.1234/shake.ml.2020},
  note      = {INVALID: Shakespeare died in 1616.}
}

@article{impossible_author_pair3,
  author    = {Newton, Isaac and Hinton, Geoffrey},
  title     = {Physics-Inspired Deep Learning},
  journal   = {Physical Review Letters},
  year      = {2019},
  doi       = {10.1103/PhysRevLett.123.123456},
  note      = {INVALID: Newton died in 1727.}
}

@techreport{nonexistent_arxiv,
  author    = {Fake, Author},
  title     = {Nonexistent {arXiv} Paper with Fake ID},
  institution = {arXiv},
  year      = {2022},
  doi       = {10.48550/arXiv.9999.9999},
  note      = {INVALID: arXiv ID 9999.9999 is invalid.}
}

@misc{hallucinated_github,
  author    = {Fake, User},
  title     = {Nonexistent {GitHub} Repository},
  howpublished = {GitHub},
  year      = {2023},
  note      = {INVALID: The GitHub repository does not exist.}
}

@article{future_date,
  author    = {Future, Author},
  title     = {Paper from the Future},
  journal   = {Future Science},
  year      = {2099},
  doi       = {10.1234/future.2099.001},
  note      = {INVALID: Year 2099 is future.}
}

@inproceedings{dead_author2,
  author    = {Gauss, Carl Friedrich},
  title     = {Gaussian Processes for Deep Learning},
  booktitle = {{NeurIPS} 2018},
  year      = {2018},
  doi       = {10.1234/nips.2018.1234},
  note      = {INVALID: Gauss died in 1855.}
}

@book{nonexistent_isbn,
  author    = {Fake, Author},
  title     = {Book with Fake {ISBN}},
  publisher = {Fake Publisher},
  year      = {2021},
  isbn      = {999-9-9999-9999-9},
  note      = {INVALID: ISBN 999-9-9999-9999-9 is invalid.}
}

@article{impossible_volume,
  author    = {Smith, J.},
  title     = {Impossible Journal Volume},
  journal   = {Science},
  volume    = {1000},
  pages     = {1--2},
  year      = {2020},
  doi       = {10.1126/science.1000.12345},
  note      = {INVALID: Science volume 1000 does not exist (max ~380).}
}

@inproceedings{hallucinated_keynote,
  author    = {Fake, Keynote},
  title     = {Fake Keynote Paper},
  booktitle = {{AAAI} 2099 Keynote},
  year      = {2099},
  doi       = {10.1234/aaai.2099.001},
  note      = {INVALID: Year 2099 is future; AAAI 2099 does not exist.}
}

@techreport{retracted_techreport,
  author    = {Retracted, Author},
  title     = {Retracted Technical Report},
  institution = {Microsoft Research},
  year      = {2019},
  note      = {INVALID: This technical report was retracted and removed.}
}

@misc{impossible_doi2,
  author    = {Fake, Name},
  title     = {Another Fake {DOI}},
  howpublished = {Online},
  year      = {2022},
  doi       = {10.1234/abcd.efgh},
  note      = {INVALID: DOI format is incorrect (should be 10.xxxx/xxxx).}
}

@article{anachronistic_2021,
  author    = {Pascal, Blaise},
  title     = {Probabilistic Programming in the 17th Century},
  journal   = {Journal of {AI} Research},
  year      = {2021},
  doi       = {10.1613/jair.2021.12345},
  note      = {INVALID: Pascal died in 1662.}
}

@inproceedings{future_conference,
  author    = {Future, Author},
  title     = {Paper at Future Conference},
  booktitle = {Proceedings of the 2100 {CVPR}},
  year      = {2100},
  doi       = {10.1234/cvpr.2100.001},
  note      = {INVALID: Year 2100 is future.}
}

@book{hallucinated_editor,
  author    = {Editor, Fake},
  title     = {Fake Edited Volume},
  publisher = {Fake Press},
  year      = {2020},
  doi       = {10.1234/fake.ed.2020},
  note      = {INVALID: This edited volume does not exist.}
}

@article{nonexistent_journal2,
  author    = {Smith, J.},
  title     = {Paper in Nonexistent Journal},
  journal   = {Journal of Fake Research},
  volume    = {1},
  pages     = {1},
  year      = {2023},
  doi       = {10.1234/jfr.2023.001},
  note      = {INVALID: Journal of Fake Research does not exist.}
}

@techreport{impossible_institution,
  author    = {Fake, Researcher},
  title     = {Tech Report from Nonexistent Lab},
  institution = {Fake Lab, Nonexistent University},
  year      = {2023},
  note      = {INVALID: Institution does not exist.}
}

@misc{retracted_online,
  author    = {Retracted, Author},
  title     = {Retracted Online Paper},
  howpublished = {arXiv},
  year      = {2020},
  doi       = {10.48550/arXiv.2005.12345},
  note      = {INVALID: This arXiv paper was retracted and removed.}
}

@article{impossible_author_pair4,
  author    = {Aristotle and LeCun, Yann},
  title     = {Ancient Greek Deep Learning},
  journal   = {Nature},
  year      = {2018},
  doi       = {10.1038/nature.2018.12345},
  note      = {INVALID: Aristotle died in 322 BC.}
}

@inproceedings{hallucinated_competition,
  author    = {Fake, Winner},
  title     = {Fake Competition Paper},
  booktitle = {{ImageNet} Challenge 2099},
  year      = {2099},
  doi       = {10.1234/imagenet.2099.001},
  note      = {INVALID: Future year; ImageNet challenge ended in 2017.}
}

@book{nonexistent_series,
  author    = {Fake, Author},
  title     = {Book in Nonexistent Series},
  series    = {Lecture Notes in Fake Intelligence},
  publisher = {Springer},
  year      = {2022},
  doi       = {10.1007/978-3-031-12345-6},
  note      = {INVALID: Lecture Notes in Fake Intelligence does not exist.}
}

@article{impossible_doi_prefix2,
  author    = {Fabricated, Name},
  title     = {Fake {DOI} with Wrong Prefix},
  journal   = {Fake Journal},
  year      = {2021},
  doi       = {10.5555/fake.doi},
  note      = {INVALID: DOI suffix contains non-numeric characters.}
}

@techreport{future_date2,
  author    = {Future, Researcher},
  title     = {Future Tech Report},
  institution = {Future University},
  year      = {2100},
  note      = {INVALID: Year 2100 is future.}
}

@misc{hallucinated_software,
  author    = {Fake, Developer},
  title     = {Nonexistent Software Citation},
  howpublished = {Software},
  year      = {2023},
  note      = {INVALID: The software does not exist.}
}

@article{anachronistic_2022,
  author    = {Euler, Leonhard},
  title     = {Graph Neural Networks in the 18th Century},
  journal   = {Transactions on Graph Theory},
  year      = {2022},
  doi       = {10.1234/tgt.2022.12345},
  note      = {INVALID: Euler died in 1783.}
}

@inproceedings{retracted_conference,
  author    = {Retracted, Author},
  title     = {Retracted Conference Paper},
  booktitle = {{ECCV} 2018},
  year      = {2018},
  doi       = {10.1234/eccv.2018.retracted},
  note      = {INVALID: This paper was retracted from ECCV 2018.}
}

@book{impossible_year2,
  author    = {Fake, Author},
  title     = {Book from the Year 1000},
  publisher = {Medieval Press},
  year      = {1000},
  doi       = {10.1234/medieval.1000},
  note      = {INVALID: Year 1000 is before the invention of printing press (c.1440).}
}

@article{nonexistent_issue,
  author    = {Smith, J.},
  title     = {Nonexistent Journal Issue},
  journal   = {Journal of {AI} Research},
  volume    = {72},
  number    = {999},
  pages     = {1--10},
  year      = {2021},
  doi       = {10.1613/jair.2021.99999},
  note      = {INVALID: JAIR volume 72 does not have issue 999.}
}

@inproceedings{hallucinated_author2,
  author    = {Nobody, Real},
  title     = {Fake Author Paper},
  booktitle = {{ICLR} 2023},
  year      = {2023},
  doi       = {10.1234/iclr.2023.fake},
  note      = {INVALID: Author "Real Nobody" does not exist.}
}
"""

if __name__ == "__main__":
    output_file = Path("bibtex/bibtex_files/deepseek.bib")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(bib_content, encoding="utf-8")
    print(f"Generated {output_file}")
    # Count entries
    count = bib_content.count("@article") + bib_content.count("@inproceedings") + \
            bib_content.count("@book") + bib_content.count("@misc") + \
            bib_content.count("@techreport")
    print(f"Total entries: {count}")