# title: PCD to Segments
# description: PCD to Segments
# author: Taewook Kang
# date: 2023.11.12
# revision history
#   0.1: initial implementation
#   0.15: add pipeline architecture
# function: clustring. filtering. make footprints. make LoD1. make spreadsheet
# license: MIT license
# reference:
#   Deng, D., 2020, September. DBSCAN clustering algorithm based on density. In 2020 7th international forum on electrical engineering and automation (IFEEA) (pp. 949-953). IEEE.
#   Zhang, W., Qi, J., Wan, P., Wang, H., Xie, D., Wang, X. and Yan, G., 2016. An easy-to-use airborne LiDAR data filtering method based on cloth simulation. Remote sensing, 8(6), p.501.
#   Beckmann, N., Kriegel, H.P., Schneider, R. and Seeger, B., 1990, May. The R*-tree: An efficient and robust access method for points and rectangles. In Proceedings of the 1990 ACM SIGMOD international conference on Management of data (pp. 322-331).
#   http://ramm.bnu.edu.cn/projects/CSF/document/
# 
import argparse, math, numpy as np, os, sys, json, shutil, re, traceback, random, contextlib
import numpy as np
import laspy, CSF
from tqdm import tqdm
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from sklearn.cluster import DBSCAN

view_log = True

def dump(outputs):
	for output in outputs:
		print(output)

def disable_stdout():
	f = open('nul', 'w')
	sys.stdout = f

def enable_stdout():
	sys.stdout = sys.__stdout__

def filtering_csf(dataset, config):
	if len(dataset) == 0:
		return None

	cloth_resolution = config['cloth_resolution']
	rigidness = config['rigidness']
	time_step = config['time_step']
	class_threshold = config['class_threshold']
	interations = config['interations']

	outputs = []
	for item in tqdm(dataset, desc='filtering_csf'):
		input_fname = item['input']
		output_fname = item['output']
		if item['active'] == False:
			continue

		inFile = laspy.read(input_fname) # read a las file
		points = inFile.points
		xyz = np.vstack((inFile.x, inFile.y, inFile.z)).transpose() # extract x, y, z and put into a list

		csf = CSF.CSF()

		# prameter settings
		csf.params.bSloopSmooth = False
		csf.params.cloth_resolution = cloth_resolution
		csf.params.rigidness = rigidness
		csf.params.time_step = time_step
		csf.params.class_threshold = class_threshold
		csf.params.interations = interations

		csf.setPointCloud(xyz)
		ground = CSF.VecInt()  # a list to indicate the index of ground points after calculation
		non_ground = CSF.VecInt() # a list to indicate the index of non-ground points after calculation

		with contextlib.redirect_stdout(None):
			csf.do_filtering(ground, non_ground) # do actual filtering.

		output_seg_gnd_fname = output_fname.format(segment='ground', cloth_resolution=f'{csf.params.cloth_resolution:.2f}')
		outFile = laspy.LasData(inFile.header)
		outFile.points = points[np.array(ground)] # extract ground points, and save it to a las file.
		outFile.write(output_seg_gnd_fname)

		output_seg_non_gnd_fname = output_fname.format(segment='non_ground', cloth_resolution=f'{csf.params.cloth_resolution:.2f}')
		outFile = laspy.LasData(inFile.header)
		outFile.points = points[np.array(non_ground)] # extract ground points, and save it to a las file.
		outFile.write(output_seg_non_gnd_fname)

		output = {"name": "ground", "output": output_seg_gnd_fname}
		outputs.append(output)
		output = {"name": "non_ground", "output": output_seg_non_gnd_fname}
		outputs.append(output)
		
	return outputs

def filtering_color(dataset, config):
	if len(dataset) == 0:
		return None

	filter_fname = config['filter']
	colormap = None
	with open(filter_fname) as json_file:
		colormap = json.load(json_file)

	outputs = []
	for item in tqdm(dataset, desc='filtering_color'):
		input_fname = item['input']
		output_fname = item['output']
		if item['active'] == False:
			continue

		inFile = laspy.read(input_fname) # read a las file
		points = inFile.points
		red = np.right_shift(inFile.red, 8).astype(np.uint8) # rgb = np.vstack((inFile.red, inFile.green, inFile.blue)).transpose() # extract x, y, z and put into a list
		green = np.right_shift(inFile.green, 8).astype(np.uint8) # https://github.com/strawlab/python-pcl/issues/171
		blue = np.right_shift(inFile.blue, 8).astype(np.uint8)

		outputs = []
		for cm in colormap['segment']:
			name = cm['name']
			if 'RGB1' not in cm or 'RGB2' not in cm:
				continue
			min_rgb = cm['RGB1']
			max_rgb = cm['RGB2']

			# Create a mask based on RGB values
			mask = ((red >= min_rgb[0]) & (red <= max_rgb[0]) &
					(green >= min_rgb[1]) & (green <= max_rgb[1]) &
					(blue >= min_rgb[2]) & (blue <= max_rgb[2]))

			# Filter points based on the mask
			filtered_points = points[mask]

			output_seg_fname = output_fname.format(segment=name)
			outFile = laspy.LasData(inFile.header)
			outFile.points = filtered_points 
			outFile.write(output_seg_fname)

			seg = {'name': name, 'output': output_seg_fname}
			outputs.append(seg)

	return outputs

def make_clusters(dataset, config):
	if len(dataset) == 0:
		return None
	
	eps = config['eps']
	min_samples = config['min_samples']
	remove_samples = config['remove_samples']
	random_color = config['random_color']

	outputs = []
	for item in tqdm(dataset, desc='make_clusters'):
		name = item['name']
		input_fname = item['input']
		output_fname = item['output']
		active = item['active']
		if active == False:	
			continue

		inFile = laspy.read(input_fname)
		points = inFile.points
		xyz = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()

		# make clusters
		cluster = None
		try:
			cluster = DBSCAN(eps=eps, min_samples=min_samples, algorithm='kd_tree', n_jobs=-1).fit(xyz) # TBD. too slow. https://copyprogramming.com/howto/dbscan-sklearn-is-very-slow, https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
		except Exception as e:
			print(e)

			outFile = laspy.LasData(inFile.header)
			outFile.points = cluster_points
			output_cluster_fname = output_fname.format(segment=f'other')
			outFile.write(output_cluster_fname)
			seg = {'name': f'{name}_other', 'output': output_cluster_fname}
			outputs.append(seg)
			continue
		labels = cluster.labels_
		n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
		print(f'number of clusters: {n_clusters}')

		# save clusters to las file
		outFile = laspy.LasData(inFile.header)
		for i in tqdm(range(n_clusters), desc='save clusters'):
			cluster_points = points[labels == i]
			if len(cluster_points) <= remove_samples:
				continue
			if inFile.header.point_format.id == 1: # https://laspy.readthedocs.io/en/latest/examples.html#creating-a-new-lasdata
				header = laspy.LasHeader(point_format=2, version="1.2") # https://laspy.readthedocs.io/en/latest/intro.html#point-format-2
				outFile = laspy.LasData(header)
				
				outFile.points.X = cluster_points.X
				outFile.points.Y = cluster_points.Y
				outFile.points.Z = cluster_points.Z
				outFile.points.x = cluster_points.x
				outFile.points.y = cluster_points.y
				outFile.points.z = cluster_points.z
			else:
				outFile.points = cluster_points
			
			if random_color:	# change red color
				outFile.points.red = [random.randint(0, 255) ] * len(outFile.points)
				outFile.points.green = [random.randint(0, 255) ] * len(outFile.points)
				outFile.points.blue = [random.randint(0, 255) ] * len(outFile.points)
			output_cluster_fname = output_fname.format(segment=f'{i}')
			outFile.write(output_cluster_fname)

			seg = {'name': f'{name}_{i}', 'output': output_cluster_fname}
			outputs.append(seg)
		 		
	return outputs

def make_footprints(dataset, config):
	if len(dataset) == 0:
		return None

	alpha_shape_factor = config['alpha_shape_factor']
	simplify_tolerance = config['simplify_tolerance']

	# iterative tracing algorithm points vectorize. https://www.researchgate.net/figure/Schematic-representation-of-the-iterative-tracing-algorithm-Each-red-point-corresponds_fig2_327763519
	from alphashape import alphashape
	from shapely.geometry import Point, Polygon
	from rtree import index

	outputs = []
	for i, item in tqdm(enumerate(dataset), desc='make_footprints'):
		name = item['name']
		input_fname = item['input']
		output_fname = item['output']
		if os.path.exists(input_fname) == False:
			continue

		inFile = laspy.read(input_fname)
		points = inFile.points
		xyz = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()

		# projection to 2D and get convex hull index
		xy = np.vstack((inFile.x, inFile.y)).transpose()
		alpha_shape = alphashape(xy, alpha=alpha_shape_factor) # https://pypi.org/project/alphashape/
		hull_points = []
		if alpha_shape.geom_type == 'MultiPolygon':	# because of the alpha_shape.geom_type is MultiPolygon, we need to find the largest polygon
			for polygon in alpha_shape.geoms:
				if len(hull_points) < len(polygon.exterior.coords):
					hull_points = polygon.exterior.coords # hull_points.extend(polygon.exterior.coords)
			hull_points = np.array(hull_points)
		else: 
			hull_points = np.array(alpha_shape.exterior.coords)

		polygon = Polygon(hull_points)
		simplified_polygon = polygon.simplify(tolerance=simplify_tolerance)
		simplified_hull_points = np.array(simplified_polygon.exterior.coords)
		if len(simplified_hull_points) < 2:
			continue
		if np.array_equal(simplified_hull_points[0], simplified_hull_points[len(simplified_hull_points) - 1]):
			simplified_hull_points = simplified_hull_points[:-1]

		idx = index.Index()
		for j, point in enumerate(xy):
			idx.insert(j, (*point, *point)) 

		footprint_indices = []
		for j, point in enumerate(simplified_hull_points):
			matches = list(idx.intersection(point))
			if len(matches) > 0:
				vertex_index = matches[0]
				footprint_indices.append(vertex_index)
			# for k, hull_point in enumerate(simplified_hull_points):
				# if np.array_equal(point, hull_point):

		if len(footprint_indices) == 0:
			continue

		# make concave hull using alpha shape
		# polygon_xyz = []
		# hull = ConvexHull(xy) # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html, http://www.qhull.org/html/qh-optq.htm
		# for simplex in hull.simplices: 
		# 	polygon_xyz.append(xyz[simplex])

		# save polygon_xy to las file
		output_cluster_fname = output_fname.format(segment=f'{i}')
		outFile = laspy.LasData(inFile.header)
		outFile.points = points[footprint_indices] # [hull.simplices] # hull.vertices] # polygon_xyz
		outFile.write(output_cluster_fname)

		seg = {'name': name, 'output': output_cluster_fname}
		outputs.append(seg)

	return outputs

def view_tin(ground_xyz, tri):
	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_trisurf(ground_xyz[:,0], ground_xyz[:,1], ground_xyz[:,2], triangles=tri.simplices, cmap=plt.cm.Spectral)
	plt.show()

import pyvista as pv, pydeck as pdk, meshio
from shapely.geometry import Polygon, MultiPolygon, mapping

def extrude_polygon(poly, height):  
	x, y = poly.exterior.coords.xy

	coords = np.array([x, y])
	points_2d = coords.T  # shape (N, 2)
	N = len(points_2d)

	points_3d = np.pad(points_2d, [(0, 0), (0, 1)])  # shape (N, 3)
	face = [N + 1] + list(range(N)) + [0]  # cell connectivity for a single cell
	polygon = pv.PolyData(points_3d, faces=face)

	obj = polygon.extrude((0, 0, height), capping=True)   # extrude along z and plot
	
	return obj

def make_lod1_geometry(dataset, config):
	if len(dataset) == 0:
		return None
	ground_fname = config['ground']

	# load ground points
	inGroundFile = laspy.read(ground_fname)
	ground_xyz = np.vstack((inGroundFile.x, inGroundFile.y, inGroundFile.z)).transpose()

	''' if view_log:
		# make mesh from ground_xyz
		tri = Delaunay(ground_xyz) # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html , http://www.qhull.org/html/qh-quick.htm#options , https://redmine.auroville.org.in/issues/8567
		import open3d as o3d
		triangles = ground_xyz[tri.simplices]
		mesh = o3d.geometry.TriangleMesh()
		mesh.vertices = o3d.utility.Vector3dVector(triangles.reshape(-1, 3))
		mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)
		mesh.compute_vertex_normals()
		o3d.visualization.draw_geometries([mesh])

		ground_mesh_fname = ground_fname.replace('.las', '.ply')
		o3d.io.write_triangle_mesh(ground_mesh_fname, mesh) '''

	# calculate height of footprint_xyz's each point 
	from scipy.interpolate import LinearNDInterpolator
	interpolator = LinearNDInterpolator(ground_xyz[:,:2], ground_xyz[:,2])

	outputs = []
	for item in tqdm(dataset, desc='make_lod1_geometry'):
		name = item['name']
		input_fname = item['input']
		output_fname = item['output']

		inFile = laspy.read(input_fname)
		points = inFile.points
		footprint_xyz = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
		middle_point = np.mean(footprint_xyz, axis=0)
		footprint_xyz_height = interpolator(middle_point[0], middle_point[1])

		# get height of footprint
		footprint_min_z = float(footprint_xyz_height)  # bottom height of footprint_xyz
		footprint_max_z = np.max(footprint_xyz[:,2])	# top height of footprint_xyz

		# project footprint_xyz to xy plane and make LoD1 objects using footprint_xy, footprint_min_z, footprint_max_z
		footprint_xy = footprint_xyz[:,:2]
		footprint_polygon = Polygon(footprint_xy)
		area = footprint_polygon.area

		extruded_mesh = extrude_polygon(footprint_polygon, footprint_max_z - footprint_min_z)
		extruded_mesh.translate([0, 0, footprint_min_z])

		# save LoD1 objects to fbx file
		output_fname = output_fname.replace('.las', '.ply')
		extruded_mesh.save(output_fname)

		footprint_xy_tuples = [tuple(point) for point in footprint_xy]
		seg = {'name': name, 
				'top_height': footprint_max_z, 
				'bottom_height': footprint_min_z, 
				'area': area,
				'output': output_fname, 
				'footprint': footprint_xy_tuples}
		outputs.append(seg)

	return outputs

def make_spreadsheet(dataset, config):
	if len(dataset) == 0:
		return None

	merge = config['merge']
	
	import pandas as pd

	outputs = []
	rows = []
	for item_index, item in tqdm(enumerate(dataset), desc='make_spreadsheet'):
		name = item['name']
		input_fname = item['input']
		output_fname = item['output']
		top_height = item['top_height']
		bottom_height = item['bottom_height']
		area = item['area']
		footprint = item['footprint']

		if merge:
			output_fname = dataset[0]['output']

		# save to csv file of name, input, output, top_height, bottom_height, area
		base_fname = os.path.basename(output_fname)
		fname, extension = os.path.splitext(base_fname)
		path = os.path.dirname(output_fname)
		output_excel_fname = path + '/' + fname + '.xlsx'

		row = [name, input_fname, output_fname, top_height, bottom_height, area, footprint]
		rows.append(row)

		if merge == False:
			with pd.ExcelWriter(output_excel_fname, engine='openpyxl') as writer:
				df = pd.DataFrame(rows, columns=['Name', 'Input Filename', 'Output Filename', 'Top Height', 'Bottom Height', 'Area', 'Footprint'])
				df.to_excel(writer, index=False, header=True)
				rows = []

		seg = {'name': name, 'output': output_excel_fname}
		outputs.append(seg)

	if merge:
		with pd.ExcelWriter(output_excel_fname, engine='openpyxl') as writer:
			df = pd.DataFrame(rows, columns=['Name', 'Input Filename', 'Output Filename', 'Top Height', 'Bottom Height', 'Area', 'Footprint'])
			df.to_excel(writer, index=False, header=True)
	return outputs

def filtering_section(dataset, config):
	if len(dataset) == 0:
		return None

	height_range = config['height_range']		
	outputs = []

	for item_index, item in tqdm(enumerate(dataset), desc='filtering_section'):
		name = item['name']
		input_fname = item['input']
		output_fname = item['output']

		inFile = laspy.read(input_fname)
		points = inFile.points
		xyz = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()

		# filter points based on height range
		mask = (xyz[:,2] >= height_range[0]) & (xyz[:,2] <= height_range[1])

		# save filtered points to las file
		output_seg_fname = output_fname.format(segment=f'{item_index}')
		outFile = laspy.LasData(inFile.header)
		outFile.points = points[mask]
		outFile.write(output_seg_fname)

		seg = {'name': name, 'output': output_seg_fname}
		outputs.append(seg)

	return outputs

def filtering_tree(dataset, config):
	if len(dataset) == 0:
		return None
	outputs = dataset	
	return outputs

def add_fname_module(module_name, config_name, input_fname):
	output_path, output_ext = os.path.splitext(input_fname)
	if config_name == '':
		output_fname = output_path + f'_{module_name}' + output_ext
	else:
		output_fname = output_path + f'_{module_name}' + f'_{config_name}' + output_ext
	return output_fname

def update_module_output(module_name, config_name, dataset):
	for i in range(len(dataset)):
		output_fname = dataset[i]['output']
		output_fname = add_fname_module(module_name, config_name, output_fname)
		dataset[i]['output'] = output_fname
	return dataset

def update_output_to_input(module_name, config_name, dataset):
	outputs = dataset.copy()
	for i in range(len(dataset)):
		input_fname = dataset[i]['output']
		output_fname = add_fname_module(module_name, config_name, input_fname)
		outputs[i]['input'] = input_fname
		outputs[i]['output'] = output_fname
	return dataset

def update_active_inputs(dataset, key_name, value, active):
	for i in range(len(dataset)):
		if re.match(value, dataset[i][key_name]):
			dataset[i]['active'] = active
		else:
			dataset[i]['active'] = not active
	return dataset

def get_value_from_name(dataset, key_name, value, output_key):
	for i in range(len(dataset)):
		if dataset[i][key_name] == value:
			return dataset[i][output_key]
	return None

def view_pcd(file_path):
	import glob
	import numpy as np
	import laspy
	import open3d as o3d

	files = glob.glob(file_path)
	pcd = o3d.geometry.PointCloud()

	pcd_list = []
	for file in files:
		inFile = laspy.file.File(file, mode="r")
		coords = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()

		pcd_file = o3d.geometry.PointCloud()
		pcd_file.points = o3d.utility.Vector3dVector(coords)

		pcd_list.append(pcd_file)

	o3d.visualization.draw_geometries(pcd_list)

def make_folders(fname):
	folder = os.path.dirname(fname)
	if not os.path.exists(folder):
		os.makedirs(folder)	

def delete_files_in_folder(fname):
	path = os.path.dirname(fname)
	if os.path.exists(path):
		shutil.rmtree(path)

def load_pipeline(config_json):
	pipeline = None
	with open(config_json) as json_file:
		pipeline = json.load(json_file)

	'''
	for pipe in pipeline:
		module = pipe['module']
		f = find_module_function(module)
		pipe['function'] = f
	'''
	return pipeline

def run_pipeline(pipes, in_dataset):
	out_dataset = None
	for pipe in pipes:
		if 'function' not in pipe:
			continue
		module = pipe['module']
		f = pipe['function']
		if f == None:
			continue
		in_dataset = pipe['dataset']
		if out_dataset != None:
			in_dataset = out_dataset
		config = pipe['config']
		out_dataset = f(module, input_dataset, config)

def get_pipeline_stage(pipeline, name):
	for stage in pipeline:
		if stage['name'] == name:
			return stage
	return None

def scan_to_model_process(args):
	function_map = {
		'csf': filtering_csf,
		'color': filtering_color,
		'cluster': make_clusters,
		'footprint': make_footprints,
		'LoD': make_lod1_geometry,
		'sheet': make_spreadsheet
	}

	outputs_result = []

	try:	
		pipeline = load_pipeline(args.pipeline)
		make_folders(args.output)

		dataset = [{
			"input": args.input,
			"output": args.output,
			"active": True}]

		outputs_result = []
		output = dataset
		for index, stage in enumerate(pipeline):
			name = stage['name']
			output_tag = ''
			if 'output_tag' in stage:
				output_tag = stage['output_tag']
			input_filter = ''
			if 'input_filter' in stage:
				input_filter = stage['input_filter']

			if index == 0:
				dataset = update_module_output(name, output_tag, output)
			else:
				dataset = update_output_to_input(name, output_tag, output)
			if len(input_filter):
				dataset = update_active_inputs(dataset, 'name', input_filter, True)

			config = stage['config']
			if 'csf.ground' in config:
				ground_fname = get_value_from_name(outputs_result[0]['dataset'], 'name', 'ground', 'input') # TBD. should be generized.
				config['ground'] = ground_fname

			output = function_map[name](dataset, config)
			result = {
				'name': name,
				'dataset': output.copy()
			}
			outputs_result.append(result)

	except Exception as e:
		print(traceback.format_exc())

	return outputs_result

def main():
	argparser = argparse.ArgumentParser(description="CSF Filtering")
	# argparser.add_argument("--input", default="./input/belleview_group.las", required=False, help="Input file name")
	# argparser.add_argument("--output", default="./output/belleview/belleview.las", required=False, help="Output file name")
	# argparser.add_argument("--input", default="./input/downsampledlesscloudEURO3.las", required=False, help="Input file name")
	# argparser.add_argument("--output", default="./output/euro3/EURO3.las", required=False, help="Output file name")
	argparser.add_argument("--input", default="./input/OTP_EPSG26910_5703_38_-122_ca_sunrise_memorial.las", required=False, help="Input file name")
	argparser.add_argument("--output", default="./output/opt/sunrise.las", required=False, help="Output file name")
	argparser.add_argument("--pipeline", default="pipeline.json", required=False, help="pipeline file name")
	args = argparser.parse_args()

	scan_to_model_process(args)
	# scan_to_model_pipeline(args)

if __name__ == "__main__":
	main()