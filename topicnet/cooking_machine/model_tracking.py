import numpy as np

from copy import deepcopy
from .models.base_model import MODEL_NAME_LENGTH


def padd_model_name(model_id):
    padding = MODEL_NAME_LENGTH - len(model_id)
    if padding > 0:
        add = padding // 2
        odd = padding % 2
        return '<' * add + model_id + '>' * (add + odd)
    else:
        return model_id[:MODEL_NAME_LENGTH]


START = padd_model_name('root')

UP_END = "┌"
DOWN_END = "└"
MIDDLE = "├"
LAST = "┤"
EMPTY = "│"
START_END = "┐"
SPACE = " "


class Tree(object):
    """
    Contains tree of an experiment and methods to work with it.

    """

    def __init__(self, tree: dict = None):
        """
        Initial stage.

        Parameters
        ----------
        tree : dict
            tree of an experiment (Default value = None)

        """
        if tree is None:
            self.tree = {'model_id': START, 'models': []}
        else:
            self.tree = tree

    def _append_description(self,
                            tree: dict, current_part: list, leaf,
                            up_sub_glue: str, down_sub_glue: str,
                            branching_marker: str):
        """

        Parameters
        ----------
        tree : dict
            tree of an experiment
        current_part : list
        leaf : dict
        up_sub_glue : str
        down_sub_glue : str
        branching_marker : str

        """
        cur_string = SPACE * len(tree["model_id"])
        up_sub_part, middle_sub_part, down_sub_part = self._get_description(leaf)
        for string in up_sub_part:
            current_part.append(SPACE * len(cur_string) + up_sub_glue + string)
        current_part.append(cur_string + branching_marker + middle_sub_part[0])
        for string in down_sub_part:
            current_part.append(SPACE * len(cur_string) + down_sub_glue + string)

    def _get_description(self, tree: dict):
        """
        Internal method to create description of the tree.

        Parameters
        ----------
        tree : dict
            tree of an experiment

        Returns
        -------
        3-list
            strings of description for up, middle and down tree parts

        """
        up_part, middle_part, down_part = [], [], []
        num_leaves = len(tree["models"])
        if num_leaves > 0:
            for id_leaf, leaf in zip(range(num_leaves)[:num_leaves // 2],
                                     tree["models"][:num_leaves // 2]):
                if id_leaf == 0:
                    self._append_description(
                        tree, up_part, leaf,
                        up_sub_glue=SPACE, down_sub_glue=EMPTY, branching_marker=UP_END
                    )
                else:
                    self._append_description(
                        tree, up_part, leaf,
                        up_sub_glue=EMPTY, down_sub_glue=EMPTY, branching_marker=MIDDLE
                    )
            if num_leaves == 1:
                middle_part.append(tree["model_id"] + START_END)
            else:
                middle_part.append(tree["model_id"] + LAST)
            for id_leaf, leaf in zip(range(num_leaves)[num_leaves // 2:],
                                     tree["models"][num_leaves // 2:]):
                if id_leaf == num_leaves - 1:
                    self._append_description(
                        tree, down_part, leaf,
                        up_sub_glue=EMPTY, down_sub_glue=SPACE, branching_marker=DOWN_END
                    )
                else:
                    self._append_description(
                        tree, down_part, leaf,
                        up_sub_glue=EMPTY, down_sub_glue=EMPTY, branching_marker=MIDDLE
                    )
        else:
            middle_part.append(tree["model_id"])

        return up_part, middle_part, down_part

    def _get_depth(self, tree):
        """
        Gets depth of the tree.

        Parameters
        ----------
        tree : dict
            tree of an experiment

        Returns
        -------
        int
            tree depth

        """
        depths = [1]
        for leaf in tree["models"]:
            depths += [self._get_depth(leaf)]

        return np.array(depths).max() + 1 * (len(tree["models"]) > 0)

    def _add_model_in_tree(self, tree, topic_model):
        """
        Adds model in the tree of an experiment.

        Parameters
        ----------
        tree : dict
            tree of an experiment
        topic_model : TopicModel
            topic model

        """
        if tree["model_id"] == topic_model.parent_model_id:
            tree["models"].append(self.transform_to_leaf(topic_model))
        else:
            for leaf in tree["models"]:
                self._add_model_in_tree(leaf, topic_model)

    def _prune(self, tree, depth, level: int = 1):
        """
        Prunes tree to get particular depth.

        Parameters
        ----------
        tree : dict
            tree of an experiment.
        depth : int
            desired tree depth
        level : int
            internal variable (current depth) (Default value = 0)

        Returns
        -------
        tree : dict
            pruned tree with desired depth

        """
        models = []
        if level <= depth:
            for model in tree["models"]:
                pruned_model = self._prune(model, depth, level + 1)
                if pruned_model is None:
                    break
                else:
                    models.append(pruned_model)
            tree["models"] = models
        else:
            return None

        return tree

    def _get_model_ids(self, tree):
        """
        Gets all model_ids of models in the tree.

        Parameters
        ----------
        tree : dict
            tree of an experiment

        Returns
        -------
        list
            model_ids of all models in the tree

        """
        model_ids = [tree["model_id"]]
        for model in tree["models"]:
            model_ids += self._get_model_ids(model)

        return model_ids

    def get_depth(self):
        """
        Gets current depth of the tree.

        Returns
        -------
        int
            depth of the tree

        """
        return self._get_depth(self.tree)

    def get_model_ids(self):
        """
        Gets models_ids of all models in the tree.

        Returns
        -------
        list
            model_ids of all models in the tree

        """
        return self._get_model_ids(self.tree)

    @staticmethod
    def transform_to_leaf(topic_model):
        """
        Transforms TopicModel to a leaf for the tree for further integration in the tree.

        Parameters
        ----------
        topic_model : TopicModel
            topic model

        Returns
        -------
        dict
            leaf of the tree

        """
        leaf = {"model_id": topic_model.model_id,
                "models": []}

        return leaf

    def show(self):
        """
        Shows the tree of an experiment in text format.
        Shows description ot the tree.

        Returns
        -------
        str
            description in txt format

        """
        up, middle, down = self._get_description(self.tree)
        print("\n".join(up + middle + down))

    def get_description(self):
        """
        Creates description of the tree.

        Returns
        -------
        list
            strings of description

        """
        up, middle, down = self._get_description(self.tree)

        return up + middle + down

    def add_model(self, topic_model):
        """
        Adds model in the tree of an experiment.

        Parameters
        ----------
        topic_model : TopicModel
            topic model

        """
        self._add_model_in_tree(self.tree, topic_model)

    def prune(self, depth):
        """
        Prunes tree to get particular depth and updates it.

        Parameters
        ----------
        depth : int
            desired tree depth

        """
        self.tree = self._prune(self.tree, depth)

    def clone(self):
        """
        Clones Tree class object.

        Returns
        -------
        tree : Tree
            copy of Tree object
        """
        tree = Tree(deepcopy(self.tree))

        return tree
