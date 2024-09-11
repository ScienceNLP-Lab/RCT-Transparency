- For your own use, we'd recommend to specify your own arguments based on your environment. For setup, please follow the instructions below:

## Install dependencies
Please install all the dependency packages using the following command lines:
```bash
conda create -n CONSORT-TERM python=3.8
conda activate CONSORT-TERM
conda install --file requirements.txt
```
or
```bash
conda create -n CONSORT-TERM python=3.8
conda activate CONSORT-TERM
conda install pip
pip install -r requirements.txt
```

*Note*: We employed and modified the existing codes from [PURE](https://github.com/princeton-nlp/PURE) as a baseline for this task.

## Training and Evaluation
After setting up your environment, then all you need to do is to run the bash command below. In the shell script, please check all the arguments used for training process.

```bash
bash run_models.sh
```

You can also do evaluation for the generated outputs by using the shell script below.

```bash
bash run_evals.sh
```
