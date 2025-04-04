import requests
import re
import os
import pandas as pd

# Define the directory to save files
save_directory = "/scratch/project/stseq/Onkar/BigData/STimage_TCGA_tsv/"  # Change this to your desired directory

# Load the file list
files = pd.read_csv("/scratch/user/s4634945/group_scratch/Projects_scripts/Self/STimage/Survival_scripts/gdc_manifest.2025-01-23.153949.txt", sep="\t")

for i in range(len(files)):
    file_id = files["id"][i]  # Assuming the file_id column exists

    # Prepare the endpoint for file download
    data_endpt = f"https://api.gdc.cancer.gov/data/{file_id}"

    try:
        # Send the request to the server
        response = requests.get(data_endpt, headers={"Content-Type": "application/json"})

        # Ensure the response is successful
        response.raise_for_status()

        # The file name can be found in the header within the Content-Disposition key.
        response_head_cd = response.headers["Content-Disposition"]
        file_name = re.findall("filename=(.+)", response_head_cd)[0]

        # Set the full file path to the specified directory
        file_path = os.path.join(save_directory, file_name)

        # Write the content to the file
        with open(file_path, "wb") as output_file:
            output_file.write(response.content)
        
        print(f"Downloaded {file_name}")

    except Exception as e:
        # Handle any error and continue to the next file
        print(f"Error downloading {file_id}: {e}")
        continue
