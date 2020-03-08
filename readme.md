# DeepSeed: A Deep Learning Methodology for Automated Soybean Seed Damage Classification

### Abstract
A soybean crop health is proportional to the vigor of the original seeds. Traditionally, the health classification of seeds is made manually by a specialized human. This is an important step to ensure the consistency and yield of each crop. This method is highly subjective and adds time and cost to production. To improve the efficacy and the efficiency of this process, we propose a methodology to automatically define the soybean seed vigor based on its damages. To do so, we compared five state-of-the-art pre-trained models (VGG-16, VGG-19, Inception-V3, Xception-V1, and Inception-ResNet-V2) fine-tuned to classify soybean seeds according to their type of damage and severity degree. Our methodology also took into account the effect of the learning rate optimizer for each model and is capable to answer the well-suited combination. According to our experiments, the best combination of model and optimizer yielded a classification accuracy of 92%. This level of accuracy obtained through our automated classification improves the entire planting and harvesting process by making it more cost effective and faster. Furthermore, the method proposed here has the potential to be used with other types of seeds.

### Citation
If you think our work is meaningfull in your reasearch, please cite us:

```
@inproceedings{DBLP:conf/mldm/CruzSB19,
  author    = {Luigi Freitas Cruz and
               Priscila Tiemi Maeda Saito and
               Pedro Henrique Bugatti},
  editor    = {Petra Perner},
  title     = {DeepSeed: {A} Deep Learning Methodology for Automated Soybean Seed
               Damage Classification},
  booktitle = {Machine Learning and Data Mining in Pattern Recognition, 15th International
               Conference on Machine Learning and Data Mining, {MLDM} 2019, New York,
               NY, USA, July 20-25, 2019, Proceedings, Volume {II}},
  pages     = {890--900},
  publisher = {ibai publishing},
  year      = {2019},
  timestamp = {Mon, 16 Dec 2019 11:07:59 +0100},
  biburl    = {https://dblp.org/rec/conf/mldm/CruzSB19.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

### Acknowledgement
This research has been supported by grants from CNPq (grants #431668/2016-7, #422811/2016-5), CAPES, Araucária Foundation, SETI and UTFPR.
