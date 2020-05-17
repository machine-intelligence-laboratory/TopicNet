from topicnet.cooking_machine.cubes.controller_cube import ScoreControllerPerplexity

DATA_REG_CONTROLLER_SORT_OF_DECREASING = [
    ([246.77072143554688,
      124.72193908691406,
      107.95775604248047,
      105.27597045898438,
      112.46900939941406,
      132.88259887695312], 0.1, True),
    ([246.77072143554688,
      124.72193908691406,
      107.95775604248047,
      105.27597045898438,
      112.46900939941406], 0.1, False),
    ([246.77072143554688,
      124.72193908691406,
      107.95775604248047,
      105.27597045898438,
      112.46900939941406], 0.05, True),

]
DATA_AGENT_CONTROLLER_LEN_CHECK = [
    ({
         "reg_name": "decorrelation",
         "score_to_track": "PerplexityScore@all",
         "tau_converter": "prev_tau * user_value",
         "max_iters": float("inf")
     }, 1),
    ({
         "reg_name": "decorrelation",
         "score_to_track": ["PerplexityScore@all"],
         "tau_converter": "prev_tau + user_value",
         "max_iters": float("inf")
     }, 1),
    ({
         "reg_name": "decorrelation",
         "score_to_track": None,  # never stop working
         "tau_converter": "prev_tau * user_value",
         "max_iters": float("inf")
     }, 0),
    ({
         "reg_name": "decorrelation",
         "score_to_track": None,  # never stop working
         "score_controller": ScoreControllerPerplexity("PerplexityScore@all", 0.1),
         "tau_converter": "prev_tau * user_value",
         "max_iters": float("inf")
     }, 1),
    ({
         "reg_name": "decorrelation",
         "score_to_track": "PerplexityScore@all",  # never stop working
         "score_controller": ScoreControllerPerplexity("PerplexityScore@all", 0.1),
         "tau_converter": "prev_tau * user_value",
         "max_iters": float("inf")
     }, 2)
]
