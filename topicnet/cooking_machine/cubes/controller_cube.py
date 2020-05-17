"""
Allows to add `ControllerAgent` (with unknown parameters) to the model, which enables user to
change `tau` during the `_fit` method.


`parameters` is a dict with four fields:

Fields
------
reg_name: str
    The name of regularizer. We want to change the tau coefficient of it during training
    Note that only one of ("reg_name", "regularizer") should be provided
regularizer: artm.regularizer.Regularizer
    Regularizer object (if we want to add non-existing regularizer to the model)
    Note that only one of ("reg_name", "regularizer") should be provided
score_to_track: str
    The name of metric which we will track.
    We assume that if that metric is 'sort of decreasing', then everything is OK
    and we are allowed to change tau coefficient further; otherwise we revert back
    to the last "safe" value and stop
    
    'sort of decreasing' perform best with PerplexityScore.

    More formal definition of "sort of decreasing": if we divide a curve into two parts like so:


        ##################################### 
        #. . . .. . . . ..  . .. . .  ... . # 
        #%. . .  . . . .  .. . . . . .  . ..# 
        #:t . . . . . . . . . . . . . . .  .# 
        # t: . . . . . . . . . . . . . . ...# 
        #. %. . . . . . . . . . . . . . .  .# 
        #. :t. . . . . . . . .  .  . . . . .# 
        #.. ;; . .  . . . .  . . . .  . . ..# 
        #  ..t..  . .  . . . . . . . . . . .# 
        #. . :t .. . . .  . . . . . . . . ..# 
        #. .. t: . . . . . . . . . . . . . .# 
        #.   ..S: . . . . . . . . . . . . ..# 
        #. . . .:;: . . . . .  . . . . . . .# 
        #. . .  . :;;  . . . . . . . . . . .# 
        #. . . . .. :%.      nmmMMmmn   .  .# 
        # .   . .  . .tt%.ztttt"' '""ttttttt# 
        #. . .    . . . '"' . . . . . . . . # 
        ##################################### 
        |                |                  | 
        |   left part    |                  | 
                   global minimum           | 
                         |     right part   | 

    then the right part is no higher than 5% of global minimum
    (you can change 5% if you like by adjusting `fraction_threshold` parameter)

    If score_to_track is None and score_controller is None, then `ControllerAgent` will never stop
    (useful for e.g. decaying coefficients)
fraction_threshold: float
    Threshold to control a score by 'sort of decreasing' metric
score_controller: ScoreControllerBase
    Custom score controller
    In case of 'sort of decreasing' is not proper to control score, you are able to create custom Score Controller 
    inherited from ScoreControllerBase. New controller must contain function `is_score_out_of_control`
     which takes TopicModel as input and return answer of type OutOfControlAnswer

tau_converter: str or callable
    Notably, def-style functions and lambda functions are allowed
    If it is function, then it should accept four arguments:
        `(initial_tau, prev_tau, cur_iter, user_value)`
    For example:

        >> lambda initial_tau, prev_tau, cur_iter, user_value:
        >>     initial_tau if cur_iter % 2 == 0 else 0

    (Note that experiment description might display lambda functions incorrectly;
     Try to keep them to a single line or use def-style functions instead)

        >> def func(initial_tau, prev_tau, cur_iter, user_value):
        >>     relu_grower = user_value * (cur_iter - 8) if cur_iter > 8 else 0
        >>     return 0 if cur_iter % 2 else relu_grower

    If it is a string, then it should be an expression consisting of numbers, operations
        and variables (four are allowed: `initial_tau, prev_tau, cur_iter, user_value`)
    For example:

    `>> "initial_tau * ((cur_iter + 1) % 2)"`

    or

    `>> "prev_tau * user_value"`

user_value_grid: list of numeric
    Values for user_value variable
    When writing `tau_converter`, you can use user_value variable.

    For example:

        >> tau_converter: "prev_tau * user_value"
        >> user_value_grid: [1, 0.99, 0.95, 0.90, 0.80, 0.5]

    (I know that tau should decay exponentially, but I'm unsure of exact half-life)

        >> tau_converter: "prev_tau + user_value"
        >> user_value_grid: [50, 100, 150, 200, 250]

    (I know that tau should increase linearly, but I'm unsure of exact speed)

        >> def func(initial_tau, prev_tau, cur_iter, user_value):
        >>     new_tau = 50 * (cur_iter - user_value) if cur_iter > user_value else 0
        >>     return new_tau
        >> tau_converter: func
        >> user_value_grid: [10, 15, 20, 25, 30]

    (Tau should start with zero, then increase linearly. I don't know when to start this process)

max_iter: numeric
    Optional (default value is `num_iter` specified for cube)
    Agent will stop changing tau after `max_iters` iterations
    `max_iters` could be `float("NaN")` and `float("inf")` values:
    that way agent will continue operating even outside this `RegularizationControllerCube`
"""  # noqa: W291

import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional

import numexpr as ne
import numpy as np
from dill.source import getsource

from .base_cube import BaseCube
from ..rel_toolbox_lite import count_vocab_size, handle_regularizer

W_HALT_CONTROL = "Process of dynamically changing tau was stopped at {} iteration"
W_MAX_ITERS = "Maximum number of iterations is exceeded; turning off"


@dataclass
class OutOfControlAnswer:
    answer: Optional[bool]
    error_message: Optional[str] = None


class ScoreControllerBase:
    def __init__(self, score_name):
        self.score_name = score_name

    def get_vals(self, model):
        if self.score_name not in model.scores:  # case of None is handled here as well
            return None

        vals = model.scores[self.score_name]
        if len(vals) == 0:
            return None

        return vals

    def __call__(self, model):
        values = self.get_vals(model)

        if values is None:
            return False

        try:
            out_of_control_result = self.is_out_of_control(values)
        except Exception as ex:
            message = (f"An error occured while controlling {self.score_name}. Message: {ex}. Score values: {values}")
            warnings.warn(message)
            return True

        if out_of_control_result.error_message is not None:
            warnings.warn(out_of_control_result.error_message)

        return out_of_control_result.answer

    def is_out_of_control(self, values: List[float]) -> OutOfControlAnswer:
        raise NotImplementedError


class ScoreControllerPerplexity(ScoreControllerBase):
    """
        Controller is properto control the Perplexity score. For others, please ensure for yourself.
    """
    DEFAULT_FRACTION_THRESHOLD = 0.05

    def __init__(self, score_name, fraction_threshold=DEFAULT_FRACTION_THRESHOLD):
        super().__init__(score_name)
        self.fraction_threshold = fraction_threshold

    def is_out_of_control(self, values: List[float]):
        idxmin = np.argmin(values)

        if idxmin == len(values):  # score is monotonically decreasing
            return False

        right_maxval = max(values[idxmin:])
        minval = values[idxmin]

        if minval <= 0:
            message = f"""Score {self.score_name} has min_value = {minval} which is <=0. 
                        This control scheme is using to control scores acting like Perplexity.
                        Ensure you control the Perplexity score or write your own controller"""
            return OutOfControlAnswer(answer=True, error_message=message)

        answer = (right_maxval - minval) / minval > self.fraction_threshold

        if answer:
            message = (f"Score {self.score_name} is too high! Right max value: {right_maxval}, min value: {minval}")
            return OutOfControlAnswer(answer=answer, error_message=message)

        return OutOfControlAnswer(answer=answer)


class ControllerAgentException(Exception): pass


class ControllerAgent:
    """
    Allows to change `tau` during the `_fit` method.

    Each `TopicModel` has a `.callbacks` attribute.
    This is a list consisting of various `ControllerAgent`s.
    Each agent is described by:

    * reg_name: the name of regularizer having `tau` which needs to be changed
    * tau_converter: function or string describing how to get new `tau` from old `tau`
    * score_to_track: score name providing control of the callback execution
    * fraction_threshold: threshold to control score_to_track
    * score_controller: custom score controller providing control of the callback execution
    * local_dict: dictionary containing values of several variables,
            most notably, `user_value`
    * is_working:
            if True, agent will attempt to change tau until something breaks.
            if False, agent will assume that something had been broken and will
            revert to the last known safe value (without trying to change anything further)

    See top-level docstring for details.
    """

    def __init__(self, reg_name, tau_converter, max_iters, score_to_track=None, fraction_threshold=None,
                 score_controller=None, local_dict=None):
        """

        Parameters
        ----------
        reg_name : str
        tau_converter : callable or str
        max_iters : int or float
            Agent will stop changing tau after `max_iters` iterations
            `max_iters` could be `float("NaN")` and `float("inf")` values:
            that way agent will continue operating even outside this `RegularizationControllerCube`
        score_to_track : str, list of str or None
            Name of score to track
            Please, use this definition to track only scores of type PerplexityScore.
            In other cases we recommend you to write you own ScoreController
        fraction_threshold : float, list of float of the same length as score_to_track  or None
            Uses to define threshold to control PerplexityScore
            Default value is 0.05
        score_controller : ScoreControllerBase, list of ScoreControllerBase or None
        local_dict : dict
        """
        if local_dict is None:
            local_dict = dict()

        self.reg_name = reg_name
        self.tau_converter = tau_converter

        self.score_controllers = []
        if isinstance(score_to_track, list):
            if fraction_threshold is None:
                scores = [(ScoreControllerPerplexity.DEFAULT_FRACTION_THRESHOLD, name) for name in score_to_track]
            elif isinstance(fraction_threshold, list) and len(score_to_track) == len(fraction_threshold):
                scores = list(zip(score_to_track, fraction_threshold))
            else:
                err_message = """Length of score_to_track and fraction_threshold must be same.
                Otherwise fraction_threshold must be None"""
                raise ControllerAgentException(err_message)

            self.score_controllers.append([ScoreControllerPerplexity(name, threshold) for (name, threshold) in scores])

        elif isinstance(score_to_track, str):
            self.score_controllers.append([ScoreControllerPerplexity(
                score_to_track,
                fraction_threshold or ScoreControllerPerplexity.DEFAULT_FRACTION_THRESHOLD
            )])

        if isinstance(score_controller, ScoreControllerBase):
            self.score_controllers.append(score_controller)
        elif isinstance(score_controller, list):
            if not all(isinstance(score, ScoreControllerBase) for score in score_controller):
                err_message = """score_controller must be of type ScoreControllerBase od list of ScoreControllerBase"""
                raise ControllerAgentException(err_message)

            self.score_controllers.extend(score_controller)

        self.is_working = True
        self.local_dict = local_dict
        self.tau_history = []
        self.max_iters = max_iters

    def _convert_tau(self):
        """ """
        if isinstance(self.tau_converter, str):
            new_tau = ne.evaluate(self.tau_converter, local_dict=self.local_dict)
            # numexpr returns np.ndarray (which is a scalar in our case)
            new_tau = float(new_tau)
        else:
            new_tau = self.tau_converter(**self.local_dict)
        return new_tau

    def _find_safe_tau(self):
        """ """
        if len(self.tau_history) < 2:
            warnings.warn("Reverting tau to 0")
            safe_tau = 0
        else:
            safe_tau = self.tau_history[-2]
        return safe_tau

    def invoke(self, model, cur_iter):
        """
        Attempts to change tau if `is_working == True`. Otherwise, keeps to the last safe value.

        Parameters
        ----------
        model : TopicModel
        cur_iter : int
            Note that zero means "cube just started", not "the model is brand new"

        """
        current_tau = model.regularizers[self.reg_name].tau
        self.tau_history.append(current_tau)
        self.local_dict["prev_tau"] = current_tau
        self.local_dict["cur_iter"] = cur_iter

        if "initial_tau" not in self.local_dict:
            self.local_dict["initial_tau"] = current_tau

        if self.is_working and len(self.tau_history) > self.max_iters:
            warnings.warn(W_MAX_ITERS)
            self.is_working = False

        if self.is_working:
            should_stop = any(
                score_controller(model) for score_controller in self.score_controllers
            )
            if should_stop:
                warnings.warn(W_HALT_CONTROL.format(len(self.tau_history)))
                self.is_working = False
                model.regularizers[self.reg_name].tau = self._find_safe_tau()
            else:
                model.regularizers[self.reg_name].tau = self._convert_tau()


class RegularizationControllerCube(BaseCube):
    def __init__(self, num_iter: int, parameters,
                 reg_search='grid', use_relative_coefficients: bool = True, strategy=None,
                 tracked_score_function=None, verbose: bool = False, separate_thread: bool = True):
        """
        Initialize stage. Checks params and update internal attributes.

        Parameters
        ----------
        num_iter : int
            number of iterations or method
        parameters : list[dict] or dict
            regularizers params
            each dict should contain the following fields: 
                ("reg_name" or "regularizer"),
                "tau_converter",
                "score_to_track" (optional),
                "fraction_threshold" (optional),
                "score_controller" (optional),
                "user_value_grid"
                See top-level docstring for details.
            Examples:

                    >>  {"regularizer": artm.regularizers.<...>,
                    >>   "tau_converter": "prev_tau * user_value",
                    >>   "score_to_track": "PerplexityScore@all",
                    >>   "fraction_threshold": 0.1,
                    >>   "user_value_grid": [0.5, 1, 2]}


            -----------

                    >>  {"reg_name": "decorrelator_for_ngramms",
                    >>   "tau_converter": (
                    >>       lambda initial_tau, prev_tau, cur_iter, user_value:
                    >>       initial_tau * (cur_iter % 2) + user_value
                    >>   )
                    >>   "score_to_track": None,
                    >>   "fraction_threshold": None,
                    >>   "score_controller": [ScoreControllerPerplexity("PerplexityScore@all", 0.1)],
                    >>   "user_value_grid": [0, 1]}

        reg_search : str
            "grid", "pair", "add" or "mul". 
            "pair" for elementwise grid search in the case of several regularizers 
            "grid" for the fullgrid search in the case of several regularizers 
            "add" and "mul" for the ariphmetic and geometric progression
            respectively for PerplexityStrategy 
            (Default value = "grid")
        use_relative_coefficients : bool
            forces the regularizer coefficient to be in relative form
            i.e. normalized over collection properties
        strategy : BaseStrategy
            optimization approach (Default value = None)
        tracked_score_function : str ot callable
            optimizable function for strategy (Default value = None)
        verbose : bool
            visualization flag (Default value = False)

        """  # noqa: W291
        super().__init__(num_iter=num_iter, action='reg_controller',
                         reg_search=reg_search, strategy=strategy, verbose=verbose,
                         tracked_score_function=tracked_score_function,
                         separate_thread=separate_thread)
        self._relative = use_relative_coefficients
        self.data_stats = None
        self.raw_parameters = parameters
        if isinstance(parameters, dict):
            parameters = [parameters]
        self._convert_parameters(parameters)

    def _convert_parameters(self, all_parameters):
        """

        Parameters
        ----------
        all_parameters : list of dict

        """
        for params_dict in all_parameters:
            assert ("reg_name" in params_dict) != ("regularizer" in params_dict)
            if "regularizer" in params_dict:
                assert params_dict["regularizer"].tau is not None

        self.parameters = [
            {
                "object": {
                    "reg_name": params_dict.get("reg_name", None),
                    "regularizer": params_dict.get("regularizer", None),
                    "score_to_track": params_dict.get("score_to_track", None),
                    "tau_converter": params_dict["tau_converter"],
                    "local_dict": {"user_value": None},
                    "max_iters": params_dict.get("max_iters", self.num_iter)
                },
                "field": "callback",
                "values": params_dict.get('user_value_grid', [0])
            }
            for params_dict in all_parameters
        ]

    def apply(self, topic_model, one_model_parameter, dictionary=None, model_id=None):
        """
        Applies regularizers and controller agents to model

        Parameters
        ----------
        topic_model : TopicModel
        one_model_parameter : list or tuple
        dictionary : Dictionary
            (Default value = None)
        model_id : str
            (Default value = None)

        Returns
        -------
        TopicModel

        """
        new_model = topic_model.clone(model_id)
        new_model.parent_model_id = topic_model.model_id

        modalities = dict()
        if self._relative:
            modalities = new_model.class_ids
            if self.data_stats is None:
                self.data_stats = count_vocab_size(dictionary, modalities)

        for (agent_blueprint_template, field_name, current_user_value) in one_model_parameter:
            agent_blueprint = dict(agent_blueprint_template)
            if agent_blueprint["reg_name"] is None:
                regularizer = agent_blueprint["regularizer"]
                new_regularizer = deepcopy(regularizer)
                handle_regularizer(
                    self._relative,
                    new_model,
                    new_regularizer,
                    self.data_stats,
                )
                agent_blueprint["reg_name"] = new_regularizer.name
            else:
                if agent_blueprint['reg_name'] not in new_model.regularizers.data:
                    error_msg = (f"Regularizer {agent_blueprint['reg_name']} does not exist. "
                                 f"Cannot be modified.")
                    raise ValueError(error_msg)

            agent_blueprint['local_dict']['user_value'] = current_user_value
            # ControllerAgent needs only reg_name in constructor
            agent_blueprint.pop("regularizer")
            agent = ControllerAgent(**agent_blueprint)
            new_model.callbacks.append(agent)
        return new_model

    def get_jsonable_from_parameters(self):
        """ """
        jsonable_parameters = []

        for one_model_parameters in self.raw_parameters:
            one_jsonable = dict(one_model_parameters)
            converter = one_model_parameters['tau_converter']

            if not isinstance(converter, str):
                try:
                    # not always works, but this is not important
                    one_jsonable["tau_converter"] = str(getsource(converter))
                except (TypeError, OSError):
                    # OSError: may arise if working in Jupyter Notebook
                    one_jsonable["tau_converter"] = "<NOT AVAILABLE>"

            jsonable_parameters.append(one_jsonable)

        return jsonable_parameters
