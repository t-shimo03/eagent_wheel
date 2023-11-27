import json
import os


def link(first_half, switch_generation, second_half):
    with open(os.path.join(first_half, "history.json"), "r") as f:
        history_first = json.load(f)
    with open(os.path.join(second_half, "history.json"), "r") as f:
        history_second = json.load(f)

    curriculum_history = []

    for i in history_first:
        if i["generation"] > switch_generation:
            break
        info = {}
        info["generation"] = i["generation"]
        info["best_reward"] = i["best_reward"]
        info["best_generation"] = i["best_generation"]
        info["elapsed"] = i["elapsed"]
        info["current_best_eval_reward"] = i["current_best_eval_reward"]
        info["current_best_reward"] = i["current_best_reward"]
        info["current_mean_reward"] = i["current_mean_reward"]
        info["current_min_reward"] = i["current_min_reward"]
        info["current_best_species_id"] = i["current_best_species_id"]
        info["current_best_individual_id"] = i["current_best_individual_id"]
        info["num_limbs"] = i["num_limbs"]
        info["structure_codes"] = i["structure_codes"]
        info["current_eval_rewards"] = i["current_eval_rewards"]
        info["current_mean_rewards"] = i["current_mean_rewards"]
        info["current_rewards"] = i["current_rewards"]
        info["success_rate"] = i["success_rate"]
        curriculum_history.append(info)

    generation_in_first_half = curriculum_history[-1]["generation"]
    elapsed_in_first_half = curriculum_history[-1]["elapsed"]

    for i in history_second:
        info = {}
        info["generation"] = i["generation"] + generation_in_first_half
        info["best_reward"] = i["best_reward"]
        info["best_generation"] = i["best_generation"]
        info["elapsed"] = i["elapsed"] + elapsed_in_first_half
        info["current_best_eval_reward"] = i["current_best_eval_reward"]
        info["current_best_reward"] = i["current_best_reward"]
        info["current_mean_reward"] = i["current_mean_reward"]
        info["current_min_reward"] = i["current_min_reward"]
        info["current_best_species_id"] = i["current_best_species_id"]
        info["current_best_individual_id"] = i["current_best_individual_id"]
        info["num_limbs"] = i["num_limbs"]
        info["structure_codes"] = i["structure_codes"]
        info["current_eval_rewards"] = i["current_eval_rewards"]
        info["current_mean_rewards"] = i["current_mean_rewards"]
        info["current_rewards"] = i["current_rewards"]
        info["success_rate"] = i["success_rate"]
        curriculum_history.append(info)

    with open(os.path.join(second_half, "linked_history.json"), 'w') as f:
        json.dump(curriculum_history, f, indent=4)

    origin = {"first" : first_half, "second" : second_half}
    with open(os.path.join(second_half, "origin.json"), 'w') as f:
        json.dump(origin, f, indent=4)


if __name__ == '__main__':
    first_half = "log/curriculum_free_1_first_half"
    switch_generation = 3
    second_half = "log/curriculum_free_1_second_half"
    link(first_half, switch_generation, second_half)
