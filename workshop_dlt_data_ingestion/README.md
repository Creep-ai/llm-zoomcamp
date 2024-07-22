1. Install dependencies
```shell
pipenv install dlt[lancedb]==0.5.1a0 ollama
```
2. Init `dlt` project
```shell
yes | dlt init rest_api lancedb
```
3. Add all nessecary environment variable (alternatively you can fill `.dlt/secrets.toml`)
4. Download data
5. Launch ollama
```shell
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve > nohup.out 2>&1 &
ollama pull llama2-uncensored
```