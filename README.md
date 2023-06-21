# ML Kaggle

## Running instructions

## Data installation
From your directory where you cloned this, you'll need to go up one directory, create a directory called input, then download the kaggle data files and drop them there.
## Model training

To run this you need to be able to run jupyter notebooks. You can run these in vscode by installing the extension.

I also install conda to set up the environment and install necessary packages.

1. Create a conda environment
```
$ conda env create -n kaggle --file env.yml
```

2. Export your existing environment
```
$ conda env export --from-history>ENV.yml
```

