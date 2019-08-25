TRAIN_PATH=''
VALIDATION_PATH=''
EXP_DIR='' # folder to save models and write summary
ACCELERATION=''
python models/gan/train.py --train-path ${TRAIN_PATH} --val-path ${VALIDATION_PATH} --exp-dir ${EXP_DIR} --acceleration ${ACCELERATION}
