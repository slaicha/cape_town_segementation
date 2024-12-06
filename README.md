# Data processing

## California Dataset
To generate image mask pais of the california dataset, I followed the instructions in this repo https://github.com/A-Stangeland/SolarDetection. The main module is `data_generation_california.py`

These are their instructions (edited):

The data generator class to use depends on the data source, as US and French satellite data have different sizes, and thus require different approaches. 
The US dataset is composed of images that are 5000 by 5000 pixels in sixe and usage of the ```DatasetGenerator``` class is created with images of this size in mind.

To generate a dataset of image samples and their corresponding binary mask ('0' if there is no panel, '1' if there is) the following arguments can be specified: 

* ```image_path```: Path to the satellite images (str)
* ```json_path```: Path to JSON file containing the panel polygons (str)
* ```dataset_path```: Path to where the dataset will be created (str)
* ```gers```: Set to `true` if generating from Gers data (Bool)
* ```image_size```: Generated image sample size, samples will be ```image_size``` by ```image_size``` pixels in size (int)
* ```shuffle```: Shuffle after generating samples (Bool)
* ```test_split```: Ratio of samples in the test set (Float)

These arguments can be modified in the `datagen_config_califonia.json` file.

To generate a dataset from satellite images and a polygon file, run the `data_generation.py` script by executing the following line:

```python data_generation.py```


### Data Augmentation
Data augmentation is used in the project to get the most out of the accessible dataset. The methods used are vertical and horizontal flips of image samples and its corresponding binary mask. 
The image sample has a 50% chance of being flipped along each axis, essentially making it possible to generate four total training samples from each training sample in the original dataset. The ground truth label is flipped along with the sample. 

### Sample California image mask pair
<p align="center">
  <img src="data/sample_california_dataset/i_0.png" alt="image" width="45%" style="display:inline-block;"/>
  <img src="data/sample_california_dataset/m_0.png" alt="mask" width="45%" style="display:inline-block;"/>
</p>



## Cape Town Dataset
To generate image masks pairs of the Cape Town images, we first needed to preprocess the annotations and create a comprehensive geodataframe that contains all the relevant information of the polygon annotations. This geodataframe follows the same structure as the one provided with the California dataset. You can find all the annotations processing details and steps in the  `data_processing_cape_town.ipynb` notebook. The final geodataframe can be found in the main directory under the name `annotations_final.geojson`. Sample image mask pairs are in the folder sample_cape_town_dataset.

### Sample Cape Town image mask pair 

<p align="center">
  <img src="data/sample_cape_town_dataset/i_W07C_4_8_4.png" alt="image" width="45%" style="display:inline-block;"/>
  <img src="data/sample_cape_town_dataset/m_W07C_4_8_4.png" alt="mask" width="45%" style="display:inline-block;"/>
</p>



# Training 
The segmentation model used for training is the U-Net. The details of the training on the California dataset and the results can be found in the `segmentation.ipynb` notebook.

