# CattleInfra
# An efficient infrared cattle face segmentation model
# Coming Soon (After Paper Acceptted)

# If the paper contributes your work, please cite our paper!
@article{SHU2024108614,
title = {Automated collection of facial temperatures in dairy cows via improved UNet},
journal = {Computers and Electronics in Agriculture},
volume = {220},
pages = {108614},
year = {2024},
issn = {0168-1699},
doi = {https://doi.org/10.1016/j.compag.2024.108614},
url = {https://www.sciencedirect.com/science/article/pii/S016816992400005X},
author = {Hang Shu and Kaiwen Wang and Leifeng Guo and Jérôme Bindelle and Wensheng Wang},
keywords = {Precision livestock farming, Animal welfare, Deep learning, Body surface temperature, Heat stress},
abstract = {In cattle, facial temperatures captured by infrared thermography provide useful information from physiological aspects for researchers and local practitioners. Traditional temperature collection requires massive manual operations on relevant software. Therefore, this paper aimed to propose a tool for automated temperature collection from cattle facial landmarks (i.e., eyes, muzzle, nostrils, ears, and horns). An improved UNet was designed by replacing the traditional convolutional layers in the decoder with Ghost modules and adding Efficient Channel Attention (ECA) modules. The improved model was trained on our open-source cattle infrared image dataset. The results show that Ghost modules reduced computational complexity and ECA modules further improved segmentation performance. The improved UNet outperformed other comparable models on the testing set, with the highest mean Intersection of Union of 80.76% and a slightly slower but still good inference speed of 32.7 frames per second. Further agreement analysis reveals small to negligible differences between the temperatures obtained automatically in the areas of eyes and ears and the ground truth. Collectively, this study demonstrates the capacity of the proposed method for automated facial temperature collection in cattle infrared images. Further modelling and correction with data collected in more complex conditions are required before it can be integrated into on-farm monitoring of animal health and welfare.}
}
