# New Nameonly CL

## Overall Pipeline
1. Generate prompts for image generation (see [generation.md](generation.md)).
2. Perform image generation for each model (see [generation.md](generation.md)).
3. Subsample and filter the generated images (see [ensemble.md](ensemble.md)).
4. Calculate RMD scores and perform RMD ensemble using the filtered images (see [ensemble.md](ensemble.md)).
5. Upload the final image directory to Google Drive (use [./dataset/upload.py](./dataset/upload.py) or upload manually).
6. Compute stats for the image directory and generate a JSON file to push:
    - `get_stats.py`
    - `make_collections.py`

## Baselines
- [Internet Explorer (IE)](ie.md)


## How to Renew Gdrive Token

1. Run the `gdrive files list` command in your local environment.  
    - If the `gdrive` command is not available, download the gdrive binary from [https://github.com/glotlabs/gdrive/releases](https://github.com/glotlabs/gdrive/releases).
2. When a link is displayed, follow it and log in with your Google account.
3. Execute the following command to export the account:  
    ```bash
    gdrive account export <YOUR_EMAIL>
    ```
4. Transfer the generated `.tar` file to the server using `scp`.
5. On the server, remove the existing account with the following command:  
    ```bash
    gdrive account remove <YOUR_EMAIL>
    ```
6. Import the token file on the server:  
    ```bash
    gdrive account import <TOKEN_FILE.tar>
    ```
7. If necessary, switch to the correct account using the following command:  
    ```bash
    gdrive account switch <YOUR_EMAIL>
    ```

## How to download datasets
1. Setup gdrive for linux by following the instructions
    ```bash
    cd dataset
    mv ./gdrive /usr/local/bin
    gdrive account import gdrive_export-dbd05088_naver_com.tar
    ```

2. Now you can see the files in the google drive by using the following command
    ```bash
    gdrive files list
    ```

3. Download the dataset you want by using the following command. For example, if you want to download the dataset PACS, use the following command. This will automatically download the dataset and extract it to the current directory.
    ```bash
    ./PACS_gdrive.sh
    ```

## How to run the code

You can run the code by first:

### API credentials

Do not store API keys or access tokens in source files. Export the credentials required by the selected script before running it:

```bash
export OPENAI_API_KEY="..."
export GOOGLE_API_KEY="..."
export HF_TOKEN="..."
export SLACK_WEBHOOK_URL="..."
```

1. Modify the `ex.sh` to set parameters for the experiment.
2. Modify the `pyfile` path in the `ex.sh` to the path of the python file you want to run.
3. Submit the job to the slurm by using the following command
    ```bash
    sbatch ex.sh
    ```

### Detailed explanation of the parameters in `ex.sh`
- `NOTE`: Name of the experiment. You should set this to the name that you can recognize the experiment.

    (IMPORTANT: Experiments with the same name will be overwritten)

- `MODE`: Method (e.g. `xder`, `er`, `bic`, `mir`)
- `SEEDS`: Between 1 and 5.
- `DATASET`: Name of the dataset (e.g. `cifar10`, `food101`, `cct`)
- `TYPES`: Training dataset (e.g. `ma`, `generated`, `web`)



### Results
You can see the results in the `results` directory.

## Acknowledgements

This research was partially supported by AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data (IITP) funded by the Ministry of Science and ICT (2022-0-00077).
