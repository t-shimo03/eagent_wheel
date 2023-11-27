import numpy as np
import random
import itertools
import Levenshtein
import copy
import pickle
import os

#デバッグログ用
import time

class stack:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
       
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)


class TreeSelector():
    def __init__(self, max_num_limbs, use_2dof=False,use_other_parts = True, require_code_graph=False):

        self.max_num_limbs = max_num_limbs
        self.use_2dof = use_2dof
        self.use_other_parts = use_other_parts
        self.codes_graph = {}
        self.codes_by_num_limbs = [[] for i in range(max_num_limbs + 1)]

        if self.use_2dof:
            self.codes_by_num_limbs[0:2] = [["10"], ["1100", "1200"]]
        else:
            self.codes_by_num_limbs[0:2] = [["10"], ["1100"]]
        if require_code_graph:
            # If pickle already exists, use it; if not, create one
            if use_2dof:
                dof = 2
            else:
                dof = 1
            
            if self.use_other_parts:
                code_list_name = os.path.join( "eagent", "data", f"code_data_{max_num_limbs}_wheel.pkl")
                code_graph_name = os.path.join( "eagent", "data", f"code_graph_{max_num_limbs}_wheel.pkl")
            else:
                code_list_name = os.path.join("eagent", "data", f"code_data_{max_num_limbs}_{dof}.pkl")
                code_graph_name = os.path.join("eagent", "data", f"code_graph_{max_num_limbs}_{dof}.pkl")
            
            if os.path.exists(code_list_name) and os.path.exists(code_graph_name):
                print("load codes")
                with open(code_list_name, "rb") as f:
                    self.codes_by_num_limbs = pickle.load(f)
                with open(code_graph_name, "rb") as f:
                    self.codes_graph = pickle.load(f)
            else:
                print("prepare codes")
                # This method creates self.codes_by_num_limbs and self.codes_graph
                self.__prepare_codes(max_num_limbs)
                with open(code_list_name, "wb") as f:
                    pickle.dump(self.codes_by_num_limbs, f)
                with open(code_graph_name, "wb") as f:
                    pickle.dump(self.codes_graph, f)

        # for BFS
        self.searched_set = set([x for x1 in self.codes_by_num_limbs[0:2] for x in x1])
        self.current_search_set = set()
        self.next_search_set = set()
        self.pivot_edges = []
        self.pivot_code = ""

    def update_pivot(self, edges):
        code = self.edges2code(edges)
        if code != self.pivot_code:
            self.searched_set.add(code)
            self.current_search_set = set(self.codes_graph[code])
            self.next_search_set = set()
            self.pivot_edges = copy.deepcopy(edges)
            self.pivot_code = code

    # update_pivot must be called before this method
    def select_next_edges(self):
        
        while True:
            if len(self.current_search_set) == 0:
                self.current_search_set = self.next_search_set
                self.next_search_set = set()
            code = random.choice(list(self.current_search_set))
            self.current_search_set.remove(code)
            for ns in self.codes_graph[code]:
                self.next_search_set.add(ns)
            if code not in self.searched_set:
                break
        self.searched_set.add(code)
        edges = self.__code2edges(code)
        #txt = open("TreeSelector_debug.txt", 'a')
        #txt.write(str(edges) + str(code) + "\n")
        #txt.close()
        # At present, edges are assigned random ids
        # Change it to match pivot as much as possible
        # Create a map for the changes
        tree = [[] for i in range(self.max_num_limbs + 1)]
        pivot_tree = [[] for i in range(self.max_num_limbs + 1)]
        for parent, child, dof, kind_joint in edges:
            tree[parent].append([child, dof, kind_joint])
        for parent, child, dof, kind_joint in self.pivot_edges:
            pivot_tree[parent].append([child, dof, kind_joint])
        node_map = self.__compare_trees(pivot_tree, -1, tree, -1, {})

        # Apply the map
        temp_edges = copy.deepcopy(edges)
        num_pending = len([x for x in node_map.values() if x <= -2])
        vacant_ids = random.sample(
            np.setdiff1d(range(self.max_num_limbs), list(node_map.values())).tolist(),
            num_pending
        )
        
        #時間があったら__compare_treesを直す
        """
        edges = []
        txt = open("TreeSelector_debug.txt", 'a')
        txt.write(str(node_map) + "\n")
        txt.write(str(vacant_ids) + "\n")
        txt.close()
        for temp_parent, temp_child, dof, kind_joint in temp_edges:
            parent = node_map[temp_parent]
            child = node_map[temp_child]
            if parent <= -2:
                parent = vacant_ids[-parent - 2]
            if child <= -2:
                child = vacant_ids[-child - 2]
            txt = open("TreeSelector_debug.txt", 'a')
            txt.write(str(parent) + str(child) + "\n")
            txt.close()
            edges.append([parent, child, dof, kind_joint])
            txt = open("TreeSelector_debug.txt", 'a')
            txt.write(str(edges) + "\n")
            txt.close()
        """

        return edges

    def select_random_changed_edges(self, old_edges):
        parent_nodes = old_edges[:, 0]
        child_nodes = old_edges[:, 1]
        kind = old_edges[:, 3]
        leaf_nodes = np.setdiff1d(child_nodes, parent_nodes)
        cmd_queue = []
        for node in child_nodes:
            if kind[node] == 1:
                cmd_queue.append([node, "increase"])
            else:
                cmd_queue.append([node, "decrease"])
        for node in leaf_nodes:
            cmd_queue.append([node, "decrease"])
        while True:
            cmd = random.choice(cmd_queue)
            if len(old_edges) >= self.max_num_limbs and cmd[1] == "increase":
                continue
            else:
                break

        new_edges = []
        list_childe = child_nodes.tolist()
        if cmd[1] == "increase":
            vacant_nodes = np.setdiff1d(range(self.max_num_limbs), child_nodes)
            new_node_id = random.choice(vacant_nodes)
            for i, j, dof, kind_joint in old_edges:
                new_edges.append([i, j, dof, kind_joint])
            if cmd[0] == -1:
                new_edges.append([cmd[0], new_node_id, 1, 1])
            else:                
                if kind[list_childe.index(cmd[0])] == 1:                    
                    new_edges.append([cmd[0], new_node_id, 1, random.choice([1, 2])])
        elif cmd[1] == "decrease":
            for i, j, dof, kind_joint in old_edges:
                if j == cmd[0]:
                    continue
                new_edges.append([i, j, dof, kind_joint])

        return new_edges

    #  As a magic number, return a number less than or equal to -2 where we want some id to be assigned later
    def __compare_trees(self, base_tree, base_current, new_tree, new_current, node_map):
        node_map.update({new_current: base_current})
        # [id of node, code of the part below it, boolean of whether it has seen all child nodes yet]
        new_children = []
        for c, dof, kind_joint in new_tree[new_current]:
            new_children.append([c, 1, self.__tree2code(new_tree, c, dof, kind_joint), True])

        if base_current < -1:
            # Since the new_tree is calculating a part of the tree that is no longer in the base_tree,
            # we look at the tree while assigning a new negative id
            for new_child,_, _, _ in new_children:
                node_map.update(self.__compare_trees(base_tree, base_current - 1, new_tree, new_child, node_map))
                print(node_map)
        else:
            base_children = []
            for c, dof, kind_joint in base_tree[base_current]:
                base_children.append([c, 1, self.__tree2code(base_tree, c, dof, kind_joint), True])

            for i in range(len(base_children)):
                base_child, _, base_child_code, _ = base_children[i]
                new_child = None
                for j in range(len(new_children)):
                    temp_new_child, _, temp_new_child_code, b = new_children[j]
                    if not b:
                        continue
                    if temp_new_child_code == base_child_code:
                        new_child = temp_new_child
                        break
                if new_child is None:
                    # There is some kind of change in this child node
                    # Go through it once and look at it again when finished looking at all the other nodes
                    pass
                else:
                    base_children[i][3] = False
                    new_children[j][3] = False
                    # node_map[new_child] = base_child
                    node_map.update(self.__compare_trees(base_tree, base_child, new_tree, new_child, node_map))

            # Finding nodes left over
            base_child_rem = [[child, code] for child, _, code, b in base_children if b]
            new_child_rem = [[child, code] for child, _, code, b in new_children if b]
            if len(new_child_rem) == 0:
                # Even if there is a rigid body in base, it will disappear and can be ignored
                pass
            else:
                new_child_patterns = itertools.permutations(new_child_rem)
                min_distance = 99999
                best_new_child_pattern = []
                for new_child_p in new_child_patterns:
                    dist = 0
                    for i in range(len(new_child_p)):
                        if i >= len(base_child_rem):
                            dist += Levenshtein.distance(new_child_p[i][1], "")
                        else:
                            dist += Levenshtein.distance(new_child_p[i][1], base_child_rem[i][1])
                    for i in range(len(new_child_p), len(base_child_rem)):
                        dist += Levenshtein.distance("", base_child_rem[i][1])
                    if dist < min_distance:
                        min_distance = dist
                        best_new_child_pattern = new_child_p
                for i in range(len(best_new_child_pattern)):
                    new_child = best_new_child_pattern[i][0]
                    if i >= len(base_child_rem):
                        # Insert a value less than any element of node_map (-2 is inserted at the beginning)
                        node_map.update(self.__compare_trees(
                            base_tree,
                            min(min(node_map.values()), -1) - 1,
                            new_tree,
                            new_child,
                            node_map
                        ))
                    else:
                        node_map.update(self.__compare_trees(
                            base_tree, base_child_rem[i][0], new_tree, new_child, node_map))
        return node_map

    def __prepare_codes(self, n=None):
        if n is None:
            n = self.max_num_limbs
        for i in range(n + 1):
            if i <= 1:
                continue
            parent_codes = self.codes_by_num_limbs[i - 1]
            child_codes = []
            for j in range(len(parent_codes)):
                if j % 100000 == 0:
                    print(f"i: {i}, j: {j} / {len(parent_codes)}")
                parent_code = parent_codes[j]
                neightbor_codes = self.__generate_neighbor_codes(parent_code)

                for s in neightbor_codes:
                    if len(s) == (i + 1) * 2:
                        child_codes.extend([s])

                # make a graph
                for neightbor_code in neightbor_codes:
                    if parent_code in self.codes_graph:
                        if neightbor_code not in self.codes_graph[parent_code]:
                            self.codes_graph[parent_code].append(neightbor_code)
                    else:
                        self.codes_graph[parent_code] = [neightbor_code]
                    if neightbor_code in self.codes_graph:
                        if parent_code not in self.codes_graph[neightbor_code]:
                            self.codes_graph[neightbor_code].append(parent_code)
                    else:
                        self.codes_graph[neightbor_code] = [parent_code]
            # list(set()) to remove duplicates
            child_codes = list(set(child_codes))
            self.codes_by_num_limbs[i] = child_codes

    def __code2edges(self, code):
        node_cnt = -2
        edges = []
        # [node id, DoF, kindjoint]
        s = stack()
        for i in range(len(code)):
            if code[i] == "1":
                node_cnt += 1
                s.push([node_cnt, 1, 1])
            elif code[i] == "2":
                node_cnt += 1
                s.push([node_cnt, 1, 2])
            elif code[i] == "0":
                child, dof, kind_joint= s.pop()
                if not s.is_empty():
                    parent, _, _ = s.peek()
                    edges.append([parent, child, dof, kind_joint])
        
        return np.array(edges)

    def __tree2code(self, tree, current, dof, kind_joint):
        child_codes = []
        for child, d, kind  in tree[current]:
            child_codes.append(self.__tree2code(tree, child, d, kind))
            
        code = str(kind_joint)
        for child_code in sorted(child_codes):
            code += child_code
        code += "0"
        return code

    def edges2code(self, edges):
        tree = [[] for i in range(self.max_num_limbs + 1)]
        for parent, child, dof, kind_joint in edges:
            tree[parent].append([child, dof, kind_joint])
        return self.__tree2code(tree, -1, 1, 1)

    def __generate_neighbor_codes(self, code):
        edges = self.__code2edges(code)
        
        parent_nodes = edges[:, 0]
        child_nodes = edges[:, 1]
        kind = edges[:, 3]
        leaf_nodes = np.setdiff1d(child_nodes, parent_nodes)
        vacant_nodes = np.setdiff1d(range(self.max_num_limbs), child_nodes)
  

        neighbor_codes = []
        if self.use_2dof:
            dofs = [1, 2]
        else:
            dofs = [1]
        if self.use_other_parts:
            kind_joints = [1, 2]
        else:
            kind_joints = [1]
        list_child = child_nodes.tolist()
        list_parent = parent_nodes.tolist()

        for c in child_nodes:
            # If there is at least one vacant, generate a graph with one more rigid body and store the code
            if len(vacant_nodes) > 0:
                if kind[list_child.index(c)] == 1:
                    if c in leaf_nodes:
                        for kind_joint in kind_joints:
                            e = [c, vacant_nodes[0], dofs[0], kind_joint]
                            added_code = self.edges2code(np.concatenate([edges, [e]]))
                            neighbor_codes.append(added_code)
                    else:
                        if kind[list_parent.index(c)] == 1:
                            
                            e = [c, vacant_nodes[0], dofs[0], kind_joints[0]]
                            added_code = self.edges2code(np.concatenate([edges, [e]]))
                            neighbor_codes.append(added_code)
                            
            # Generate a graph in which one rigid body vanishes and store the code
            if c in leaf_nodes:
                removed_code = self.edges2code(edges[edges[:, 1] != c])
                neighbor_codes.append(removed_code)
        # If there is at least one vacant, generate a graph with one new rigid body added to the torso
        if len(vacant_nodes) > 0:
            e = [-1, vacant_nodes[0], dofs[0], kind_joints[0]]
            added_code = self.edges2code(np.concatenate([edges, [e]]))
            neighbor_codes.append(added_code)
        
        return list(set(neighbor_codes))


"""
if __name__ == "__main__":
    ts = TreeSelector(6, use_2dof = False, require_code_graph = True)
    txt = open("TreeSelector_debug.txt", 'a')
    for key in ts.codes_graph:
        txt.write("-----"+ "\n")
        txt.write(str(key)+str(ts.codes_graph[key])+"\n")
    s = [len(x) for x in ts.codes_by_num_limbs]
    txt.write(str(s) + "\n")
    txt.close()
    ts.update_pivot([
        [
            2,
            3,
            1,
            1
        ],
        [
            -1,
            2,
            1,
            1
        ],
        [
            4,
            1,
            1,
            1
        ],
        [
            4,
            0,
            1,
            1
        ],
        [
            5,
            4,
            1,
            1
        ],
        [
            -1,
            5,
            1,
            1
        ]
     ])
    for _ in range(6):
        
        for i in range(8):
            txt = open("TreeSelector_debug.txt", 'a')
            edges = ts.select_next_edges()
            code = ts.edges2code(edges)
            txt.write(str(edges) + str(code) + "\n")
            txt.close()
            if i == 2:
                new_piv = edges
        ts.update_pivot(new_piv)
    """