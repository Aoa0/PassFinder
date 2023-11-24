## PassFinder
PassFinder is an automated approach to detecting passwords in code files written in different programming languages. 

## Environment
We trained the models on Ubuntu 20.04 with cudatoolkit 11.0.221. The Python dependencies are listed in `requirements.txt`.

## Dataset
`context.csv` contains the raw dataset for the Context Model. For more details you can read our [paper](https://aoa0.github.io/pubs/icse22.pdf). The datasets used for experiments (10-fold and cross language) can be generated by applying methods in `make_data.py` on the raw dataset. For desensitization, we have removed the passwords in the code snippets. If you find any unexpected sensitive information in the dataset, please let us know. The dataset for training the Password Model is relatively large, and we have uploaded it to [Google Drive](https://drive.google.com/file/d/1eleQeIQccCztKOWFG9LeZ4SbseCKITp7/view?usp=sharing).

## Train & Test
+ To train and test the two models, you can enter the `passfinder` directory and run the `train.py` script, for example: 
    + `python3 train.py --task password --lr 0.0001 --batch_size 512 --cuda --num_workers 6 --epochs 12 --save_folder password --log_interval 100 --train_path ../data/train.csv --val_path ../data/validate.csv`
    + `python3 train.py --task context --lr 0.0001 --batch_size 32 --cuda --num_workers 4 --max_length 1024 --epochs 32 --save_folder context --log_interval 100 --train_path path_to_train.csv --val_path path_to_test.csv `

