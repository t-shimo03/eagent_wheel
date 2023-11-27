# ICM
python ./train.py -c ewalker_lock_icm.json -o log/thesis_lock_icm_3
# (ICM+)MDM
python ./train.py -c ewalker_lock_icm+mdm.json -o log/thesis_lock_icm+mdm_3
# MDM
python ./train.py -c ewalker_lock_mdm.json -o log/thesis_lock_mdm_3
# (MDM+)ICM
python ./train.py -c ewalker_lock_mdm+icm.json -o log/thesis_lock_mdm+icm_3
