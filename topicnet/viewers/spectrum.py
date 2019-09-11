"""
A few ways to obtain "decent" solution to TSP problem
which returns a spectre of topics in our case.  
If speed is the essence I recommend to use functions providing
good initial solution. Which are, get_nearest_neighbour_init.  
If that solution is not good enough use annealing heuristic (get_annealed_spectrum).  
Another good but time-heavy option is full check with get_three_opt_path.  
Performs well on < 50 topics.  
Within a few runs with right temperature selected it can provide a
solution better than the initial.
"""  # noqa: W291
import numpy as np
import warnings
from scipy.spatial import distance
from tqdm import tqdm
from .base_viewer import BaseViewer


def get_nearest_neighbour_init(phi_matrix, metric='jensenshannon', start_topic=0):
    """
    Given the matrix calculates the initial path by nearest neighbour heuristic.

    Parameters
    ----------
    phi_matrix : np.array of float
        a matrix of N topics x M tokens from the model
    metric : str
        name of a metric to compute distances (Default value = 'jensenshannon')
    start_topic : int
        an index of a topic to start and end the path with (Default value = 0)

    Returns
    -------
    init_path : list of int
        order of initial topic distribution

    """
    init_path = [start_topic, ]
    connection_candidates = [int(topic) for topic in np.arange(phi_matrix.shape[0])
                             if topic not in init_path]
    neighbour_vectors = phi_matrix[connection_candidates, :]

    while len(connection_candidates) > 0:
        last_connection = phi_matrix[[init_path[-1]]]
        nearest_index = distance.cdist(last_connection, neighbour_vectors, metric=metric).argmin()
        init_path.append(connection_candidates[nearest_index])
        connection_candidates = [int(topic) for topic in np.arange(phi_matrix.shape[0])
                                 if topic not in init_path]
        neighbour_vectors = np.delete(neighbour_vectors, nearest_index, axis=0)

    init_path.append(start_topic)
    init_path = [int(topic) for topic in init_path]
    return init_path


def generate_all_segments(n):
    """
    Generates all segments combinations for 3-opt swap operation.

    Parameters
    ----------
    n : int > 5
        length of path for fixed endpoint

    Yields
    -------
    list of int

    """
    for i in range(n-1):
        for j in range(i + 2, n - 1):
            for k in range(j + 2, n - 1):  # + (i > 0)
                yield [i, j, k]


def generate_three_opt_candidates(path, sequence):
    """
    Generates all possible tour connections and filters out a trivial one.

    Parameters
    ----------
    path : np.array of float
        square matrix of distances between all topics
    sequence : list of int
        list of indices to perform swap on

    Yields
    ------
    list of int
        possible tour

    """
    chunk_start = path[:sequence[0] + 1]
    chunk_one = path[sequence[0] + 1:sequence[1] + 1]
    chunk_two = path[sequence[1] + 1:sequence[2] + 1]
    chunk_end = path[sequence[2] + 1:]

    for change_chunks in [True, False]:
        middle_chunks = [chunk_two, chunk_one] if change_chunks else [chunk_one, chunk_two]

        for reverse_first_chunk in [True, False]:
            if reverse_first_chunk:
                first_chunk = middle_chunks[0][::-1]
            else:
                first_chunk = middle_chunks[0]

            for reverse_second_chunk in [True, False]:

                if reverse_second_chunk:
                    second_chunk = middle_chunks[1][::-1]
                else:
                    second_chunk = middle_chunks[1]

                if change_chunks or reverse_first_chunk or reverse_second_chunk:
                    tour = chunk_start + first_chunk + second_chunk + chunk_end
                    yield tour


def make_three_opt_swap(path, distance_m, sequence, temperature=None):
    """
    Performs swap based on the selection candidates,
    allows for non-optimal solution to be accepted
    based on Boltzman distribution.

    Parameters
    ----------
    path : list of int
        current path
    distance_m : np.array of float
        square matrix of distances between all topics
    sequence : list of int
        list of indices to perform swap on
    temperature : float
        "temperature" parameter regulates strictness of
        the new candidate choice (Default value = None)
        if None - works in a regime when only better solutions are chosen  
        This regime is used for 3-opt heuristic

    Returns
    -------
    path : list of int
        best path after the permutation
    val : float
        a value gained after the path permutation

    """  # noqa: W291

    cut_connections = sum([[path[ind], path[ind + 1]] for ind in sequence], [])
    baseline = np.sum(distance_m[cut_connections[:-1], cut_connections[1:]])

    # 6 == len(cut_connections) always
    new_connections = list(generate_three_opt_candidates(cut_connections,
                                                         generate_index_candidates(6)))

    candidates = list(generate_three_opt_candidates(path, sequence))
    scores = [np.sum(distance_m[new[:-1], new[1:]]) - baseline for new in new_connections]
    best_score = np.min(scores)

    if best_score < 0.0:
        path = candidates[np.argmin(scores)]
        val = best_score
    else:
        if temperature is None:
            val = 0.0
        else:
            # 1e-8 saves from division by 0
            boltzman = np.exp(- best_score / temperature)
            val = 0.0
            if np.random.rand() > boltzman:
                path = candidates[np.argmin(scores)]
                val = best_score

    return path, val


def get_three_opt_path(path, distance_m, max_iter=20):
    """
    Iterative improvement based on 3 opt exchange.

    Parameters
    ----------
    path : list of int
        path to optimize
    distance_m : np.array of float
        square matrix of distances between all topics, 
        attempt at optimizing path from the other end
    max_iter : int
        maximum iteration number (Default value = 20)

    Returns
    -------
    path : list of int
        end optimization of the route

    """  # noqa: W291
    count_iter = 0
    while True and count_iter <= max_iter:
        delta = 0

        for segment in generate_all_segments(len(path)):
            path, d = make_three_opt_swap(path, distance_m, segment)
            delta += d
        count_iter += 1
        if count_iter >= max_iter:
            warnings.warn('Reached maximum iterations', UserWarning)
        if delta >= 0:
            break

    return path


def generate_index_candidates(n):
    """
    Randomly chooses 3 indexes from the path.  
    Does not swap the first or the last point because they fixed.

    Parameters
    ----------
    n : int > 5
        length of the path

    Returns
    -------
    segment: list of int
        sorted list of candidates for 3 opt swap optimization

    """  # noqa: W291
    segment = np.zeros(3, dtype='int')

    first_interval = np.arange(n - 5)
    segment[0] = np.random.choice(first_interval)

    second_interval = np.arange(segment[0] + 2, n - 3)
    segment[1] = np.random.choice(second_interval)

    third_interval = np.arange(segment[1] + 2, n - 1)
    segment[2] = np.random.choice(third_interval, 1)

    return segment


def get_annealed_spectrum(phi_matrix,
                          t_coeff,
                          start_topic=0,
                          metric='jensenshannon',
                          init_path=None,
                          max_iter=1000000,
                          early_stopping=100000,):
    """
    Returns annealed spectrum for the topics in the Phi matrix
    with default metrics being Jensen-Shannon.

    Parameters
    ----------
    phi_matrix : np.array of float
        Phi matrix of N topics x M tokens from the model
    t_coeff : float
        coefficient that brings ambiguity to the process,
        bigger coefficient allows to jump from local minima.
    start_topic : int
        index of a topic to start and end the path with (Default value = 0)
    metric : str
        name of a metric to compute distances (Default value = 'jensenshannon')
    init_path : list of int
        initial route, contains all numbers from 0 to N-1,
        starts and ends with the same number from the given range (Default value = None)
    max_iter : int
        number of iterations for annealing (Default value = 1000000)
    early_stopping : int
        number of iterations without improvement before stop (Default value = 100000)

    Returns
    -------
    best_path : list of int
        best path obtained during the run
    best_score : float
        length of the best path during the run

    """  # noqa: W291
    distance_m = distance.squareform(distance.pdist(phi_matrix, metric=metric))
    np.fill_diagonal(distance_m, 10 * np.max(distance_m))
    if init_path is None:
        current_path = get_nearest_neighbour_init(phi_matrix,
                                                  metric=metric,
                                                  start_topic=start_topic)
    else:
        current_path = init_path

    if len(current_path) < 6:
        warnings.warn('The path is too short, returning nearest neighbour solution.',
                      UserWarning)
        return current_path, np.sum(distance_m[current_path[:-1], current_path[1:]])

    best_score = np.sum(distance_m[current_path[:-1], current_path[1:]])
    best_path = current_path
    running_score = best_score

    no_progress_steps = 0
    for i in tqdm(range(max_iter), total=max_iter, leave=False):
        temperature_iter = t_coeff * (max_iter / (i + 1))
        sequence = generate_index_candidates(len(current_path))
        current_path, score = make_three_opt_swap(current_path,
                                                  distance_m,
                                                  sequence,
                                                  temperature=temperature_iter)
        running_score += score

        if running_score < best_score:
            best_path = current_path
            best_score = running_score
            no_progress_steps = 0
        else:
            no_progress_steps += 1
        if no_progress_steps >= early_stopping:
            break
    return best_path, best_score


class TopicSpectrumViewer(BaseViewer):
    def __init__(
        self,
        model,
        t_coeff=1e5,
        start_topic=0,
        metric='jensenshannon',
        init_path=None,
        max_iter=1000000,
        early_stopping=100000,
        verbose=False,
        class_ids=None
    ):
        """
        Class providing wrap around for functions
        that allow to view a collection of topics
        in order of their similarity to each other.

        Parameters
        ----------
        model : TopicModel
            topic model from TopicNet library
        t_coeff : float
            coefficient for annealing, value should be chosen
        start_topic : int
            number of model topic to start from
        metric : string or function
            name of the default metric implemented in scipy or function 
            that calculates metric based on the input matrix
        init_path : list of int
            initial tour that could be provided by the user
        max_iter : int
            number of iterations for annealing
        early_stopping : int
            number of iterations without improvement before stop
        verbose : boolean
            if print the resulting length of the tour
        class_ids : list of str
            parameter for model.get_phi method
            contains list of modalities to obtain from the model
            (Default value = None)
        """  # noqa: W291
        super().__init__(model=model)
        self.metric = metric
        self.start_topic = start_topic
        self.t_coeff = t_coeff
        self.init_path = init_path
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.max_iter = max_iter
        self.class_ids = class_ids

    def view(self, class_ids=None):
        """
        The class method returning ordered spectrum of
        the topics.

        Parameters
        ----------
        class_ids : list of str
            parameter for model.get_phi method
            contains list of modalities to obtain from the model (Default value = None)
        ordered_topics : list of str
            topic names from the model ordered as spectrum

        """  # noqa: W291
        # default get_phi returns N x T matrix while we implemented T x N
        if class_ids is None:
            class_ids = self.class_ids
        model_phi = self.model.get_phi(class_ids=class_ids).T
        spectrum, distance = get_annealed_spectrum(model_phi.values,
                                                   self.t_coeff,
                                                   metric=self.metric,
                                                   start_topic=self.start_topic,
                                                   init_path=self.init_path,
                                                   max_iter=self.max_iter,
                                                   early_stopping=self.early_stopping,)
        if self.verbose:
            print('the resulting path length: ', distance)
        ordered_topics = (
            model_phi
            .iloc[spectrum]
            .index.values
        )
        return ordered_topics
