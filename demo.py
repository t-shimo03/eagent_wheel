import argparse
import json
import os
import cv2
from mujoco_py import GlfwContext

from eagent.model import Model
from eagent.config import cfg_dict

from stable_baselines3.common.callbacks import EvalCallback  # noqa

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg_filename", type=str, default=None)
    parser.add_argument("-i", "--initial_params_filename", type=str, default=None)
    parser.add_argument("-s", "--search_cfg", type=bool, default=False)
    parser.add_argument("-t", "--type", type=str, default="visualize")
    args = parser.parse_args()

    if args.cfg_filename is None:
        cfg = {}
    else:
        cfg = cfg_dict[args.cfg_filename]
    cfg_fullname = None
    if args.search_cfg:
        assert args.initial_params_filename is not None

        # Go up two directories and look for a file named cfg.json
        cfg_fullname = os.path.join(
            os.path.dirname(args.initial_params_filename), "cfg.json"
        )
        if not os.path.exists(cfg_fullname):
            cfg_fullname = os.path.join(
                os.path.dirname(os.path.dirname(args.initial_params_filename)), "cfg.json"
            )
        if not os.path.exists(cfg_fullname):
            print("cfg.json does not exist")
        else:
            with open(cfg_fullname, "r") as f:
                cfg.update(json.load(f))

    assert len(cfg.keys()) > 0

    if args.initial_params_filename is not None:
        cfg["initial_params_filename"] = args.initial_params_filename
    with open(cfg["initial_params_filename"], "r") as f:
        initial_params = json.load(f)
    model = Model(cfg, initial_params)

    load_zip = False
    if load_zip:
        cfg["output_dirname"] = "log/0.8.4_20211228_134442_failure"
        zipname = os.path.join("zips", "follower13_policy_model.zip")
        path_dict = {
            "model_zipname": zipname
        }
        model = Model(cfg, initial_params, path_dict)

    if args.type == "visualize":
        while True:
            model.simulate_once(
                render_mode=True,
                num_steps=cfg["num_steps_in_eval"]
            )
    elif args.type == "eval":
        while True:
            r, _, s = model.evaluate(20, cfg['num_steps_in_eval'], False)
            print(f"eval_reward: {r}, eval_success_rate: {s}")

    # make 4 graphs
    elif args.type == "sim":

        num_episodes_in_eval = 5000  # Select a number
        print(f"to simulate {num_episodes_in_eval} times")
        r, _, _ = model.evaluate(num_episodes_in_eval, cfg['num_steps_in_eval'], False, make_graphs=True)

        result_dirname = os.path.join(os.path.dirname(cfg["initial_params_filename"]), "result")
        os.makedirs(result_dirname, exist_ok=True)
        filename = os.path.join(result_dirname, f"rewards_and_joints_{num_episodes_in_eval}_episodes.json")
        with open(filename, "w") as f:
            json.dump(r, f, indent=2)
        print("finish")

        # graph_dirname = os.path.join(os.path.dirname(cfg["initial_params_filename"]), "graph")
        # os.makedirs(graph_dirname, exist_ok=True)
        # params_filename = ""
        # for i in str(cfg["initial_params_filename"]):
        #     if i == "/":
        #         i = "_"
        #     params_filename += i

        # # variables for plot
        # episodes = []
        # rewards = []
        # num_failures = []
        # for i in range(num_episodes_in_eval):
        #     episodes.append(i + 1)
        # for i in range(num_episodes_in_eval):
        #     rewards.append(r[i][0])
        #     num_failures.append(len(r[i][1])) if len(r[i][1]) <= 4 else num_failures.append(4)  # regard more than 4 as 4
        # mean = sum(rewards) / len(rewards)

        # # scatter diagram with colorbar
        # plt.scatter(episodes, rewards, c=num_failures, cmap="binary", lw=0.5, edgecolors="k", vmin=0, vmax=4)
        # plt.hlines(mean, 0, num_episodes_in_eval, colors="r", label=f"mean = {mean}")
        # plt.legend()
        # plt.colorbar(label="num_failures")
        # plt.xlim(0, num_episodes_in_eval)
        # plt.ylim(-100, 1300)
        # plt.xlabel("episode")
        # plt.ylabel("reward")
        # filename = os.path.join(graph_dirname, f"scatter_diagram_with_colorbar_{num_episodes_in_eval}_episodes_#{params_filename}#.png")
        # plt.savefig(filename)
        # plt.close()

        # # scatter diagram
        # plt.scatter(episodes, rewards)
        # plt.hlines(mean, 0, num_episodes_in_eval, colors="r", label=f"mean = {mean}")
        # plt.legend()
        # plt.xlim(0, num_episodes_in_eval)
        # plt.ylim(-100, 1300)
        # plt.xlabel("episode")
        # plt.ylabel("reward")
        # filename = os.path.join(graph_dirname, f"scatter_diagram_{num_episodes_in_eval}_episodes_#{params_filename}#.png")
        # plt.savefig(filename)
        # plt.close()

        # # histogram
        # plt.hist(rewards, range=(-100, 1300), rwidth=0.9, bins=28, orientation="horizontal")
        # plt.hlines(mean, 0, num_episodes_in_eval * 0.3, colors="r", label=f"mean = {mean}")
        # plt.legend()
        # plt.xlim(0, num_episodes_in_eval * 0.3)
        # plt.ylim(-100, 1300)
        # plt.ylabel("reward")
        # filename = os.path.join(graph_dirname, f"histogram_{num_episodes_in_eval}_episodes_#{params_filename}#.png")
        # plt.savefig(filename)
        # plt.close()

        # # violinplot
        # plt.violinplot(rewards, showmeans=True)
        # plt.text(1, 1300, f"mean = {mean:.2f}", ha="left")
        # plt.ylim(-100, 1300)
        # plt.ylabel("reward")
        # filename = os.path.join(graph_dirname, f" violinplot_{num_episodes_in_eval}_episodes_#{params_filename}#.png")
        # plt.savefig(filename)
        # plt.close()

    # make a scatter diagram where rewards with failures within max_num_failures are shown
    # calculate a weighted average
    # ! Attention: Comment out "self.failed_joint_ids = self.failed_joints_selector()" in gym_evolving_locomotion_envs.__reset_env()
    elif args.type == "graph_fail":

        max_num_failures = 2  # Select a number

        r, _, s, wa = model.evaluate_failure(10, cfg['num_steps_in_eval'], max_num_failures)

        # weighted average
        print(f"eval_weighted_reward: {wa}, eval_success_rate: {s}, ")

        graph_dirname = os.path.join(os.path.dirname(cfg["initial_params_filename"]), "graph")
        os.makedirs(graph_dirname, exist_ok=True)
        params_filename = ""
        for i in str(cfg["initial_params_filename"]):
            if i == "/":
                i = "_"
            params_filename += i

        # variables for plot
        episodes = []
        failure_modes = []
        rewards = []
        num_failures = []
        num_failure_modes = len(r)
        for i in range(num_failure_modes):
            episodes.append(i + 1)
            rewards.append(r[i][0])
            failure_modes.append(f"{r[i][1]}")
            num_failures.append(len(r[i][1]))

        # scatter diagram with color bar
        # plt.scatter(episodes, rewards, s=20, c=num_failures, cmap='binary', edgecolors="k", vmin=0, vmax=max_num_failures)
        # plt.colorbar(label="num_failures")
        plt.hlines(wa, 0, num_failure_modes, colors="r", label=f"the weighted average = {wa}")
        for i in range(num_failure_modes):
            plt.text(episodes[i], rewards[i], failure_modes[i], size=5, ha="center", va="center")  # to plot joint ids
        plt.text(max(episodes), 1300, f"the weighted average = {wa:.2f}", ha="right")
        plt.xlim(0, num_failure_modes)
        plt.ylim(-100, 1300)
        plt.xlabel("failure mode")
        plt.ylabel("reward")
        filename = os.path.join(graph_dirname, f"scatter_graph_within_{max_num_failures}_failures_#{params_filename}#.png")
        plt.savefig(filename)
        plt.close()

    # show a change of a mean reward to select "num_episodes_in_eval"
    elif args.type == "graph_ave":
        episodes = []
        rewards = []
        reward_means = []
        max_episode = 3000
        print("start plot")
        for n in range(max_episode):
            n += 1

            r, _, s = model.evaluate(1, cfg['num_steps_in_eval'], False, make_graphs=True)

            rewards.append(r[0][0])
            episodes.append(n)
            reward_means_now = sum(rewards) / n
            reward_means.append(reward_means_now)

            graph_dirname = os.path.join(os.path.dirname(cfg["initial_params_filename"]), "graph")
            os.makedirs(graph_dirname, exist_ok=True)

            # line chart
            plt.scatter(episodes, rewards, s=5)
            plt.plot(episodes, reward_means)
            plt.hlines(reward_means_now * 1.01, 0, n, color="r", linestyles="dashed")
            plt.text(n, reward_means_now * 1.01, " + 1 %", color="red", ha="left", va="bottom")
            plt.hlines(reward_means_now * 0.99, 0, n, color="r", linestyles="dashed", label="-1%")
            plt.text(n, reward_means_now * 0.99, " - 1 %", color="red", ha="left", va="top")
            plt.xlim(0, n)
            plt.ylim(reward_means_now - 50, reward_means_now + 50)
            plt.xlabel("episode")
            plt.ylabel("reward")

            filename = os.path.join(graph_dirname, f"scatter_graph_ave_in_{max_episode}.png")
            plt.savefig(filename)
            plt.clf()

            if n % 200 == 0:
                print(f"{n} / {max_episode}")

    elif args.type == "record":
        # Create a window to init GLFW.
        GlfwContext(offscreen=True)
        # print(f"camera: {model.env.model._camera_name2id}")

        num_steps = cfg["num_steps_in_eval"]
        num_episodes = 1
        size = (640, 480)
        # size = (1280, 960)
        frame_rate = 30
        if cfg_fullname is not None:
            videoname = os.path.basename(os.path.dirname(cfg_fullname))
        else:
            videoname = os.path.basename(cfg['initial_params_filename'])
        video_path = f"./log/video/{videoname}.mp4"
        print(f"path: {video_path}")

        scale = size[0] / 640
        fmt = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        writer = cv2.VideoWriter(video_path, fmt, frame_rate, size)
        for i in range(num_episodes):
            obs = model.env.reset()
            for t in range(num_steps + 1):
                rgb_array = model.env.render(mode="rgb_array", width=size[0], height=size[1])
                bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                cv2.putText(bgr_array, f"step: {t}", (int(5 * scale), int(25 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8 * scale, (255, 255, 255), thickness=2)
                # cv2.putText(bgr_array, f"episode: {i+1}, step: {t}", (int(5 * scale), int(25 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.8 * scale, (255, 255, 255), thickness=2)
                writer.write(bgr_array)

                act = model.get_action(obs)
                obs, reward, done, info = model.env.step(act)
        print("record done")
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
