VAL_PATH='' #validation path 
ACCELERATION='' #acceleration factor
CHECKPOINT='' #best_model.
OUT_DIR='' # Path to save reconstruction files 
python models/recon_glgan/run.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR}
