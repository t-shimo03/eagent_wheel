# ICM
python ./train.py -c ewalker_nofail_icm.json -o log/thesis_nofail_icm_3
# (ICM+)MDM
python ./train.py -c ewalker_nofail_icm+mdm.json -o log/thesis_nofail_icm+mdm_3
# MDM
python ./train.py -c ewalker_nofail_mdm.json -o log/thesis_nofail_mdm_3
# (MDM+)ICM
python ./train.py -c ewalker_nofail_mdm+icm.json -o log/thesis_nofail_mdm+icm_3
