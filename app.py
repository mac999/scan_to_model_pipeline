# author: taewook kang
# date: 2025-02-28
# description: scan to model pipeline app
# email: laputa99999@gmail.com
import os, json, sys, time, shutil, subprocess, zipfile, requests, gradio as gr
from scan_to_model_pipeline import scan_to_model_process

module_path = os.path.dirname(os.path.realpath(__file__))

class args_param:
	input = ''
	output = ''
	pipeline = ''

def process_point_cloud(pipeline_file, input_file, progress=gr.Progress()):
    try:
        output_dir = module_path + "/output"
        os.makedirs(output_dir, exist_ok=True)

        # Run the scan to model process
        params = args_param()
        params.input = input_file
        params.output = os.path.join(output_dir, "result.las")
        params.pipeline = pipeline_file
        scan_to_model_process(params, progress.tqdm)

        # Create a zip file of the output
        zip_filename = module_path + "/output.zip"
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for root, dirs, files in progress.tqdm(os.walk(output_dir), desc="Zipping files"):
                for file in files:
                    zipf.write(os.path.join(root, file), file)

        # Clean up the output directory
        shutil.rmtree(output_dir)
    except Exception as e:
        print(e)
        gr.Warning(f'Error: {e}')
        output_dir = ''
    return zip_filename

with gr.Blocks(title="Scan to Model Pipeline") as interface:
    gr.Markdown("# Scan to Model Pipeline (ver 0.2. prototype)")
    gr.Markdown("Upload pipeline configuration file (JSON) and point cloud data file (LAS, LAZ) to process the data and download the results as a zip file.")
    gr.Markdown("1. [Upload pipeline configuration (JSON)](https://github.com/mac999/scan_to_model_pipeline/blob/main/pipeline.json)</br>2. Upload [point cloud data (LAS, LAZ)](https://github.com/mac999/scan_to_model_pipeline/tree/main/input)</br>3. Click 'Run Pipeline' button</br>In detail, refer to the [github page](https://github.com/mac999/scan_to_model_pipeline.git)")
    with gr.Row(equal_height="height"):
        input_config = gr.File(label="Pipeline Configuration File", file_types=['json'])
        input_files = gr.File(label="Point Cloud Data File", file_types=['pcd', 'txt', 'las', 'laz'])
    output_file = gr.File(label="Download Model Output Zip File")
    run_button = gr.Button("Run Pipeline")
    run_button.click(fn=process_point_cloud, inputs=[input_config, input_files], outputs=output_file)

interface.launch(share=True)