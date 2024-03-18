# Face segmentation web app using streamlit

🤖 This is a web application for a face and hair segmentation using Streamlit.

## Highlights

1. No of face detection 
2. Face and hair segmentation


## Prerequisites

1. Python 3.10
2. Streamlit
3. opencv
4. openvino

## Setup
1. Clone the repository:

    ```bash
    git clone https://github.com/Nakul74/face_segmentation.git
    ```

2. Create a Conda environment with the specified version of Python from the `runtime.txt` file:

    ```bash
    conda create -p ./envs $(cat runtime.txt) -y
    ```

3. Activate the environment:

    ```bash
    conda activate envs/
    ```

4. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

Use the following command to run the application:

```bash
streamlit run app.py
```
