# 前半、故障なしMDM
python ./train.py -c ewalker_curriculum_first.json -o log/curriculum_first_half_nofail_1

# フリー故障１回目
# 後半、故障ありICM
python ./train.py -c ewalker_curriculum_second_free.json -o log/curriculum_second_half_free_1

# ロック故障１回目
# 前半はフリー故障と共有
# 後半、故障ありICM
python ./train.py -c ewalker_curriculum_second_lock.json -o log/curriculum_second_half_lock_1
