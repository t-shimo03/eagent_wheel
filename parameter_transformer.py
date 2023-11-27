import json
import os
import numpy as np
from eagent.config import cfg_dict


class Transform():
    def __init__(self, log_old, parameter_filename, cfg_name, output_dir):
        self.log_old = log_old
        self.parameter_filename = parameter_filename
        self.cfg_name = cfg_name
        self.output_dir = output_dir
        with open(os.path.join(log_old, "cfg.json"), "r") as f:
            self.cfg_old = json.load(f)
        with open(os.path.join(log_old, f"parameter_{parameter_filename}.json"), "r") as f:
            self.parameter_old = json.load(f)
        self.cfg_new = cfg_dict[cfg_name]
        with open(self.cfg_new["initial_params_filename"], "r") as f:
            self.parameter_new_format = json.load(f)
        self.max_num_limbs_old = self.cfg_old["max_num_limbs"]
        self.max_num_limbs_new = self.cfg_new["max_num_limbs"]

    def transform_structure_para(self, id, current, parent):
        for child in self.structure_tree[current]:
            dof = int(self.dofs[child])
            kind_joint = int(self.kind_joints[child])
            self.structure_edges_new.append([parent, id, dof, kind_joint])
            parent_id = id
            self.mu_format_list[id] = self.mu_old_list[child]
            self.sigma_format_list[id] = self.sigma_old_list[child]
            id += 1
            # assert id < self.max_num_limbs_new
            if id >= self.max_num_limbs_new:
                raise Exception("########num of limbs is out of code graph#######")
            id = self.transform_structure_para(id, child, parent_id)
        return id

    def make_policy_para(self):
        cfg = self.cfg_new
        max_num_limbs = cfg["max_num_limbs"]
        pi_net = []
        vf_net = []
        num_obs = max_num_limbs * 4 + 11
        pi_net.append(num_obs)
        vf_net.append(num_obs)
        net_arch_pi = cfg["policy_kwargs"]["net_arch"][0]["pi"]
        net_arch_vf = cfg["policy_kwargs"]["net_arch"][0]["vf"]
        pi_net.extend(net_arch_pi)
        vf_net.extend(net_arch_vf)
        num_act = max_num_limbs * 2
        pi_net.append(num_act)
        vf_net.append(1)

        num_params_pi = 0
        for layer in range(len(pi_net) - 1):
            num_params_pi += pi_net[layer] * pi_net[layer + 1]
            num_params_pi += pi_net[layer + 1]

        num_params_vf = 0
        for layer in range(len(vf_net) - 1):
            num_params_vf += vf_net[layer] * vf_net[layer + 1]
            num_params_vf += vf_net[layer + 1]

        num_params = num_params_pi + num_params_vf
        policy_weights_new = (np.random.randn(num_params) * 0.1).tolist()
        action_net_bias = [0 for i in range(num_act)]
        policy_weights_new[- ((vf_net[-2] * vf_net[-1] + vf_net[-1]) + pi_net[-1]) : - (vf_net[-2] * vf_net[-1] + vf_net[-1])] = action_net_bias
        policy_weights_new[-1] = 0

        return policy_weights_new

    def transform_parameter_file(self):
        parameter_old = self.parameter_old
        structure_edges_old = parameter_old["structure_edges"]
        mu_old = parameter_old["structure_weights"]["mu"]
        sigma_old = parameter_old["structure_weights"]["sigma"]

        parameter_new_format = self.parameter_new_format
        mu_format = parameter_new_format["structure_weights"]["mu"]
        sigma_format = parameter_new_format["structure_weights"]["sigma"]

        max_num_limbs_old = self.max_num_limbs_old
        self.structure_tree = [[] for i in range(max_num_limbs_old + 1)]
        self.dofs = np.zeros(max_num_limbs_old)
        self.kind_joints = np.zeros(max_num_limbs_old)
        for parent, child, dof, kind_joint in structure_edges_old:
            self.structure_tree[parent].append(child)
            self.dofs[child] = dof
            self.kind_joints[child] = kind_joint


        self.mu_old_list = np.array(mu_old).reshape([-1, 9]).tolist()
        self.sigma_old_list = np.array(sigma_old).reshape([-1, 9]).tolist()
        self.mu_format_list = np.array(mu_format).reshape([-1, 9]).tolist()
        self.sigma_format_list = np.array(sigma_format).reshape([-1, 9]).tolist()

        self.structure_edges_new = []
        self.transform_structure_para(0, -1, -1)
        structure_edges_new = self.structure_edges_new

        mu_new = sum(self.mu_format_list, [])
        sigma_new = sum(self.sigma_format_list, [])
        policy_weights_new = self.make_policy_para()

        parameter_new = {"structure_edges" : structure_edges_new, "structure_weights" : {"mu" : mu_new, "sigma" : sigma_new}, "policy_weights" : policy_weights_new}

        # with open(os.path.join("zoo", "walker", f"{os.path.basename(self.log_old)}_parameter{self.parameter_filename}_max{self.max_num_limbs_new}.json"), 'w') as f:
        #     json.dump(parameter_new, f)

        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "parameter_transformed.json"), 'w') as f:
            json.dump(parameter_new, f)


if __name__ == '__main__':
    # log_old =
    # cfg_name =
    log_old = "log/curriculum_free_1_first_half"
    parameter_filename = "best"  # have to change
    cfg_name = "ewalker_iso6.json"
    output_dir = "log/curriculum_free_1_second_half"
    Transform(log_old, parameter_filename, cfg_name, output_dir).transform_parameter_file()
