# 🚀Scan to Model Pipeline
3D scan data such as PCD (point cloud data) to model (ply) conversion.</br>
The scan to model pipeline is an open source tool that automatically generates mesh model files (PLY) by filtering and clustering data in LAS, a point cloud format. Using this, you can automate the extraction of buildings, ground, and trees from PCD. It is structured as a pipeline, so you can easily adjust parameters for each data processing step. This open source can be used to generate statistical numerical, low LoD level model files of objects from large number of PCD files.</br>

- PCD filtering using color map, CSF(W. Zhang etc)
- Clustering such as house, tree
- Footprint extraction
- Spreadsheet including ID, area, height etc
- Pipeline config file support (refer to pipeline config section)

## Revision history
- 0.1: draft version. basic functions supports. It's still in its early stages, but if you have the will, you can improve the parts that are lacking.

## Examples
![Pipeline Overview](https://github.com/mac999/scan_to_model_pipeline/blob/main/scan_to_model_pipeline.gif)
![Pipeline Overview](https://github.com/mac999/scan_to_model_pipeline/blob/main/image1.PNG)
![Pipeline Overview](https://github.com/mac999/scan_to_model_pipeline/blob/main/image2.PNG)
![Pipeline Overview](https://github.com/mac999/scan_to_model_pipeline/blob/main/image3.PNG)
![Pipeline Overview](https://github.com/mac999/scan_to_model_pipeline/blob/main/image4.PNG)
![Pipeline Overview](https://github.com/mac999/scan_to_model_pipeline/blob/main/image5.PNG)

# Install
To install the required dependencies, you can use:
```bash
pip install -r requirements.txt
```

# Config 
## pipeline 
The pipeline consists of stages such as clustring with CSF, footprint and LoD generation. Each stage has parameters depending on stage's algorithm. Refer to the below example.  
```
[
	{            
		"name": "csf",
		"config": {
			"cloth_resolution": 1.0, 
			"rigidness": 3,
			"time_step": 0.65, 
			"class_threshold": 0.5, 
			"interations": 500 
		}
	}, 
	{
		"name": "cluster",
		"config": {
			"eps": 3.0, 
			"min_samples": 10,
			"remove_samples": 100,
			"random_color": false 
		}
	}, 
	{
		"name": "footprint",
		"config": {
			"alpha_shape_factor": 0.2, 
			"simplify_tolerance": 0.1
		}
	}, 
	{
		"name": "LoD",
		"config": {
			"ground": "$ground"
		}
	},
	{
		"name": "sheet",
		"config": {
			"merge": true
		}
	}
]
```

## color map 
Color maps can be used for PCD clustering, but the current version is not complete.
```
{
    "name": "color map filter",
    "version": "1.0.0",
    "description": "segmentation using color map",
    "author": "",
    "email": "",
    "segment": [
        {
            "name": "house",
            "RGB1": [95, 108, 118],
            "RGB2": [167, 171, 189]
        }, 
        {
            "name": "tree",
            "RGB1": [30, 50, 30],
            "RGB2": [140, 150, 60]
        }, 
        {
            "name": "other"
        }
    ]
}
```

# Acknowledge
W. Zhang, J. Qi*, P. Wan, H. Wang, D. Xie, X. Wang, and G. Yan, “An Easy-to-Use Airborne LiDAR Data Filtering Method Based on Cloth Simulation,” Remote Sens., vol. 8, no. 6, p. 501, 2016. (http://www.mdpi.com/2072-4292/8/6/501/htm)

# License
[MIT License](https://pitt.libguides.com/openlicensing/MIT)
