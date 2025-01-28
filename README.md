# sports-simulation 

### Dependencies
Ensure dependencies have been met:
- If running on a Virginia Tech VM, chances are it has all the dependencies met.
- Otherwise, ensure that the nvidia-container-toolkit is installed properly (see below)

Note: You need a GPU to run any of this.

#### nvidia-container-toolkit installation
[Copied from NVIDIA's webpage:](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
```sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Running this environment


1. Clone this repository and enter it
    ```
    cd sports-simulation
    ```
2. Build the Docker container
    ```sh
    docker build -t sports-simulation ./sports-simulation
    ```
    Note: If you need to make any changes to the Dockerfile and they are not being reflected when you run the container, then try pruning the container image: `docker system prune sports-simulation` and rebuilding.
3. Run the Docker container
    ```sh
    docker run --gpus all --shm-size=8g -it -v ~/sports-simulation:/sports-simulation sports-simulation
    ```
4. When in the container, make sure to update the `config.json` file with the appropriate file paths.
5. Within the container, run the program
    ```sh
    cd /sports-simulation/sports-simulation && python3 ./main.py --config_json_path config.json
    ``` 
