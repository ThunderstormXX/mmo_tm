from typing import Optional

import numpy as np
import time
from tqdm import tqdm
from scipy.optimize import minimize_scalar
from src.models import TrafficModel, BeckmannModel, TwostageModel, Model

from src.utils import *


np.random.seed(228)


def frank_wolfe(
    model: BeckmannModel,
    eps_abs: float,
    max_iter: int = 10000,  # 0 for no limit (some big number)
    max_time: int = 60,
    times_start: Optional[np.ndarray] = None,
    stop_by_crit: bool = True,
    use_tqdm: bool = True,
    linesearch: bool = False,
) -> tuple:
    """One iteration == 1 shortest paths call"""

    optimal = False

    # init flows, not used in averaging
    if times_start is None:
        times_start = model.graph.ep.free_flow_times.a.copy()

    flows_averaged = model.flows_on_shortest(times_start)

    max_dual_func_val = -np.inf
    dgap_log = []
    time_log = []
    relative_gap_log = []
    primal_log = []

    rng = range(1_000_000) if max_iter == 0 else tqdm(range(max_iter), disable=not use_tqdm)
    # steps = []
    for k in rng:
        beforetime = time.time()

        times = model.tau(flows_averaged)
        flows = model.flows_on_shortest(times)

        if linesearch:
            res = minimize_scalar(
                lambda y: model.primal(flows_averaged * (1 - y) + y * flows), bounds=(0.0, 1.0), tol=1e-12
            )
            stepsize = res.x
            # print(gamma)
        else:
            stepsize = 2.0 / (k + 2)
        # dgap_log.append(times @ (flows_averaged - flows))  # FW gap

        flows_averaged = stepsize * flows + (1 - stepsize) * flows_averaged
        # flows_averaged = (
        #     flows if k == 0 else stepsize * flows + (1 - stepsize) * flows_averaged
        # )

        # ## TEST FLOW CORRECT
        # is_correct , corrects = check_correct_flow(flows_averaged, model )
        # print(is_correct,  corrects)
        # raise Exception('TEST')

        aftertime = time.time()
        time_log.append((time_log[-1] if len(time_log) > 0 else 0) + aftertime - beforetime)

        dual_val = model.dual(times, flows)
        max_dual_func_val = max(max_dual_func_val, dual_val)
        # print(dual_val ,max_dual_func_val)
        # if max_dual_func_val == dual_val  :
        #     steps.append(k)

        # equal to FW gap if dual_val == max_dual_func_val
        primal = model.primal(flows_averaged)
        primal_log.append(primal)
        dgap_log.append(primal - max_dual_func_val)
        # print((primal - max_dual_func_val)/max_dual_func_val)
        relative_gap_log.append((primal - max_dual_func_val) / max_dual_func_val)

        if time_log[-1] > max_time:
            break

        if stop_by_crit and dgap_log[-1] <= eps_abs:
            optimal = True
            break

    return (
        times,
        flows_averaged,
        (dgap_log, np.array(time_log) - time_log[0], {"primal": primal_log, "relative_gap": relative_gap_log}),
        optimal,
    )


## Моя реализция NFW
def N_conjugate_frank_wolfe(
    model: BeckmannModel,
    eps_abs: float,
    max_iter: int = 100,  # 0 for no limit (some big number)
    max_time: int = 60,
    times_start: Optional[np.ndarray] = None,
    stop_by_crit: bool = True,
    use_tqdm: bool = True,
    linesearch: bool = False,
    cnt_conjugates: int = 3,
) -> tuple:
    """One iteration == 1 shortest paths call"""

    optimal = False

    # init flows, not used in averaging
    if times_start is None:
        times_start = model.graph.ep.free_flow_times.a.copy()
    flows = model.flows_on_shortest(times_start)

    max_dual_func_val = -np.inf
    dgap_log = []
    time_log = []
    primal_log = []
    relative_gap_log = []

    times = model.tau(flows)

    beforetime = time.time()
    flows = model.flows_on_shortest(times)

    dual_val = model.dual(times, flows)
    max_dual_func_val = max(max_dual_func_val, dual_val)
    primal = model.primal(flows)
    primal_log.append(primal)
    dgap_log.append(primal - max_dual_func_val)
    relative_gap_log.append((primal - max_dual_func_val) / max_dual_func_val)

    aftertime = time.time()
    time_log.append((time_log[-1] if len(time_log) > 0 else 0) + aftertime - beforetime)

    rng = range(1, 1_000_000) if max_iter == 0 else tqdm(range(1, max_iter), disable=not use_tqdm)

    gamma = 1.0
    d_list = []
    S_list = []
    gamma_list = []
    gamma = 1
    epoch = 0
    for k in rng:
        beforetime = time.time()
        if gamma > 0.99999:
            # if gamma < 0.0001:
            epoch = 0
            S_list = []
            d_list = []
        if k == 1 or epoch == 0:
            epoch = epoch + 1
            t = model.tau(flows)
            sk_FW = model.flows_on_shortest(t)
            dk = sk_FW - flows
            S_list.append(sk_FW)
            d_list.append(dk)
        else:
            t = model.tau(flows)
            sk_FW = model.flows_on_shortest(t)
            dk_FW = sk_FW - flows
            hessian = model.diff_tau(flows)

            B = np.sum(d_list * hessian * d_list, axis=1)
            A = np.sum(d_list * hessian * dk_FW, axis=1)
            N = len(B)
            betta = [-1] * (N + 1)
            betta_sum = 0
            delta = 0.0001
            for m in range(N, 0, -1):
                betta[m] = -A[-m] / (B[-m] * (1 - gamma_list[-m])) + betta_sum * gamma_list[-m] / (1 - gamma_list[-m])
                if betta[m] < 0:
                    betta[m] = 0
                # elif betta[m] > 1- delta :
                #     betta[m] = 1 - delta
                #     betta_sum = betta_sum + 1 - delta
                else:
                    betta_sum = betta_sum + betta[m]
            alpha_0 = 1 / (1 + betta_sum)
            alpha = np.array(betta)[1:] * alpha_0

            # alpha_default = 0.01
            # if alpha_0 < alpha_default :
            #     alpha_0 = alpha_default
            #     alpha = alpha  / np.sum(alpha)
            #     alpha = alpha * (1 - alpha_default)

            # if max(np.max(alpha) , alpha_0) > 0.99 :
            #     alpha_0 = 0.2
            #     alpha = 0.8 * np.ones(len(alpha)) / len(alpha)

            alpha = alpha[::-1]

            # print('-----------------ITER:' , k , '--- directions =' , [np.sum(sk_FW)]+list(np.sum(S_list,axis=1)) )
            # print('-----------------ITER:',k,'--- alpha =' , [alpha_0 ]+ list(alpha[::-1]) )

            sk = alpha_0 * sk_FW + np.sum(alpha * np.array(S_list).T, axis=1)
            dk = sk - flows

            # print('CHECK CONJUGATE :' , len(d_list)  , 'alpha:' , alpha  , 'alpha_0: ' , alpha_0 , 'list_conjugates: ' , np.sum(dk*hessian*d_list , axis=1))
            d_list.append(dk)
            S_list.append(sk)

            epoch = epoch + 1

            if epoch > cnt_conjugates:
                d_list.pop(0)
                S_list.pop(0)
                gamma_list.pop(0)

        if linesearch:
            res = minimize_scalar(lambda y: model.primal(flows + y * dk), bounds=(0.0, 1.0), tol=1e-12)
            gamma = res.x
        else:
            gamma = 2.0 / (k + 2)

        gamma_list.append(gamma)

        dual_val = model.dual(t, sk_FW)
        max_dual_func_val = max(max_dual_func_val, dual_val)

        flows = flows + gamma * dk

        # equal to FW gap if dual_val == max_dual_func_val
        primal = model.primal(flows)
        primal_log.append(primal)
        dgap_log.append(primal - max_dual_func_val)
        relative_gap_log.append((primal - max_dual_func_val) / max_dual_func_val)

        aftertime = time.time()
        time_log.append((time_log[-1] if len(time_log) > 0 else 0) + aftertime - beforetime)
        if stop_by_crit and dgap_log[-1] <= eps_abs:
            optimal = True
            break

    return (
        t,
        flows,
        (dgap_log, np.array(time_log) - time_log[0], {"primal": primal_log, "relative_gap": relative_gap_log}),
        optimal,
    )


def ustm(
    model: Model,
    eps_abs: float,
    eps_cons_abs: float = np.inf,
    max_iter: int = 10000,  # 0 for no limit (some big number)
    max_sp_calls: int = 10000,  # max shortest paths calls, dont count the first (preprocessing) call
    stop_by_crit: bool = True,
    use_tqdm: bool = True,
) -> tuple:
    """for primal-dual minimization of composite minus dual function -D(t) =  Ф(t) + h(t).
    subgrad Ф(t) = -flows_on_shortest(t) = -flows_subgd(t)"""

    dgap_log = []
    cons_log = []
    time_log = []
    primal_log = []
    relative_gap_log = []

    A_prev = 0.0
    # fft = model.graph.ep.free_flow_times.a
    # t_start = fft.copy()  # times
    t_start = model.init_dual_point()
    y_start = u_prev = t_prev = np.copy(t_start)
    assert y_start is u_prev  # acceptable at first initialization
    grad_sum_prev = np.zeros(len(t_start))

    # grad_y = -model.flows_on_shortest(y_start)  # Ф'(y)
    _, grad_y, _ = model.func_psigrad_primal(y_start)

    L_value = np.linalg.norm(grad_y) / 10

    A = u = t = y = None
    optimal = False

    max_dual_func_val = -np.inf

    rng = range(1_000_000) if max_iter == 0 else tqdm(range(max_iter), disable=not use_tqdm)
    for k in rng:
        inner_iters_num = 0
        while True:
            inner_iters_num += 1

            alpha = 0.5 / L_value + (0.25 / L_value**2 + A_prev / L_value) ** 0.5
            A = A_prev + alpha

            y = (alpha * u_prev + A_prev * t_prev) / A
            func_y, grad_y, primal_var_y = model.func_psigrad_primal(y)  # -model.dual(y, flows_y), -flows_y

            grad_sum = grad_sum_prev + alpha * grad_y

            u = model.dual_composite_prox(y_start - grad_sum, A)

            t = (alpha * u + A_prev * t_prev) / A
            func_t, _, _ = model.func_psigrad_primal(t)

            max_dual_func_val = max(max_dual_func_val, -func_t)

            lvalue = func_t
            rvalue = (
                func_y
                + np.dot(grad_y, t - y)
                + 0.5 * L_value * np.sum((t - y) ** 2)
                +
                # 0.5 * alpha / A * eps_abs )  # because, in theory, noise accumulates
                0.5 * eps_abs
                # 0.1 * eps_abs)
            )

            if lvalue <= rvalue:
                break
            else:
                L_value *= 2

            assert L_value < np.inf

        A_prev = A
        L_value /= 2

        t_prev = t
        u_prev = u
        grad_sum_prev = grad_sum

        gamma = alpha / A

        # primal variable can be tuple
        if k == 0:
            primal_var_averaged = primal_var_y
        elif type(primal_var_averaged) == tuple:
            primal_var_averaged = tuple(
                primal_var_averaged[i] * (1 - gamma) + primal_var_y[i] * gamma for i in range(len(primal_var_averaged))
            )
        else:
            primal_var_averaged = primal_var_averaged * (1 - gamma) + primal_var_y * gamma

        # consider every model.flows_on_shortest() call for fair algs comparison
        if type(primal_var_averaged) == tuple:
            primal, cons = model.primal(*primal_var_averaged), model.capacity_violation(*primal_var_averaged)
        else:
            primal, cons = model.primal(primal_var_averaged), model.capacity_violation(primal_var_averaged)
        dgap_log += [primal - max_dual_func_val] * (inner_iters_num * 2)
        cons_log += [cons] * (inner_iters_num * 2)
        time_log += [time.time()] * (inner_iters_num * 2)

        primal_log.append(primal)
        # if k % int( count_sources /count_random_correspondences) == 0:
        relative_gap_log.append((primal - max_dual_func_val) / max_dual_func_val)

        if stop_by_crit and dgap_log[-1] <= eps_abs and cons_log[-1] <= eps_cons_abs:
            optimal = True
            break

        if len(dgap_log) > max_sp_calls:
            break

    return (
        t,
        primal_var_averaged,
        (dgap_log, np.array(time_log) - time_log[0], {"primal": primal_log, "relative_gap": relative_gap_log}),
        optimal,
    )


def subgd(
    model: TrafficModel,
    R: float,
    eps_abs: float,
    eps_cons_abs: float,
    max_iter: int = 1000000,  # 0 for no limit (some big number)
    stop_by_crit: bool = True,
    use_tqdm: bool = True,
) -> tuple:
    num_nodes, num_edges = model.graph.num_vertices(), model.graph.num_edges()
    flows_averaged = np.zeros(num_edges)

    fft = model.graph.ep.free_flow_times.a
    times = fft.copy()

    dgap_log = []
    cons_log = []

    S = 0  # sum of stepsizes

    optimal = False

    max_dual_func_val = -np.inf

    rng = range(1_000_000) if max_iter == 0 else tqdm(range(max_iter), disable=not use_tqdm)
    for k in rng:
        # inlined subgradient calculation with paths set saving
        flows_subgd = model.flows_on_shortest(times)

        h = R / (k + 1) ** 0.5 / np.linalg.norm(flows_subgd)

        dual_val = model.dual(times, flows_subgd)
        max_dual_func_val = max(max_dual_func_val, dual_val)

        flows_averaged = (S * flows_averaged + h * flows_subgd) / (S + h)
        S += h

        dgap_log.append(model.primal(flows_averaged) - max_dual_func_val)
        cons_log.append(model.capacity_violation(flows_averaged))

        if stop_by_crit and dgap_log[-1] <= eps_abs and cons_log[-1] <= eps_cons_abs:
            optimal = True
            break

        times = model.dual_composite_prox(times + h * flows_subgd, h)

    return times, flows_averaged, (dgap_log, cons_log), optimal


def cyclic(
    model: TwostageModel,
    eps_abs: float,  # dgap tolerance
    traffic_assigment_eps_abs: float,
    traffic_assigment_max_iter: int = 100,
    # entropy_eps: Union[float, None] = None,
    max_iter: int = 20,
    stop_by_crit: bool = True,
) -> tuple:
    """For twostage model"""

    dgap_log = []
    cons_log = []
    time_log = []

    rng = range(1_000_000) if max_iter == 0 else tqdm(range(max_iter))

    traffic_model = model.traffic_model
    distance_mat_averaged = model.distance_mat(traffic_model.graph.ep.free_flow_times.a.copy())

    optimal = False
    times = None
    for k in rng:
        traffic_mat, lambda_l, lambda_w = model.solve_entropy_model(distance_mat_averaged)

        traffic_model.set_traffic_mat(traffic_mat)
        # isinstance fails after autoreload
        if traffic_model.__class__.__name__ == "BeckmannModel":
            times, flows, inner_dgap_log, *_ = frank_wolfe(
                traffic_model,
                eps_abs=traffic_assigment_eps_abs,
                max_iter=traffic_assigment_max_iter,
                times_start=times,
                use_tqdm=False,
            )
        elif traffic_model.__class__.__name__ == "SDModel":
            times, flows, inner_dgap_log, *_ = ustm(
                traffic_model,
                eps_abs=traffic_assigment_eps_abs,
                max_iter=traffic_assigment_max_iter,
                use_tqdm=False,
            )
        else:
            assert False, f"traffic_model has wrong class name : {type(traffic_model.__class__.__name__)}"

        # print(f"inner iters={len(inner_dgap_log)}")
        distance_mat = model.distance_mat(times)

        dgap_log.append(model.primal(flows, traffic_mat) - model.dual(times, lambda_l, lambda_w, distance_mat))
        cons_log.append(traffic_model.capacity_violation(flows))
        time_log.append(time.time())

        if stop_by_crit and dgap_log[-1] <= eps_abs:
            optimal = True
            break

        distance_mat_averaged = (distance_mat_averaged + distance_mat) / 2  # average to fix oscillations

    return (
        times,
        flows,
        traffic_mat,
        (dgap_log, cons_log, np.array(time_log) - time_log[0]),
        optimal,
    )


def stochastic_correspondences_frank_wolfe(
    model: BeckmannModel,
    eps_abs: float,
    max_iter: int = 10000,  # 0 for no limit (some big number)
    max_time: int = 60,
    times_start: Optional[np.ndarray] = None,
    stop_by_crit: bool = True,
    use_tqdm: bool = True,
    linesearch: bool = False,
    weighted: bool = False,
    count_random_correspondences: int = 5,
) -> tuple:
    """One iteration == 1 shortest paths call"""

    optimal = False

    # init flows, not used in averaging
    if times_start is None:
        times_start = model.graph.ep.free_flow_times.a.copy()

    max_dual_func_val = -np.inf
    dgap_log = []
    time_log = []

    relative_gap_log = []
    primal_log = []

    rng = range(1_000_000) if max_iter == 0 else tqdm(range(max_iter), disable=not use_tqdm)

    flows_averaged, storage = model.flows_on_shortest(times_start, return_flows_by_sources=True)
    sources = model.correspondences.sources
    count_sources = len(sources)

    node_traffic = model.correspondences.node_traffic_mat
    weights = np.sum(node_traffic, axis=-1)[: len(sources)]
    weights /= np.sum(weights)

    for k in rng:
        beforetime = time.time()

        times = model.tau(flows_averaged)

        if weighted:
            source = np.random.choice(sources, size=(count_random_correspondences,), replace=False, p=weights)
        else:
            source = np.random.choice(sources, size=(count_random_correspondences,), replace=False)

        # try:
        flows, flows_by_sources = model.flows_on_shortest(times, sources_indexes=source, return_flows_by_sources=True)
        # except Exception as e:
        #     print(f"Error in flows_on_shortest: {e}")
        #     print(times, source)
        #     print(flows_by_sources)

        if weighted:
            weighted_storage_flows = np.zeros_like(flows)
            weighted_flows = np.zeros_like(flows)
            for key in source:
                weighted_storage_flows += storage[key] / weights[key]
                weighted_flows += flows_by_sources[key] / weights[key]
        else:
            storage_flows = np.zeros_like(flows)
            for key in source:
                storage_flows += storage[key]

        if weighted:
            direction = weighted_flows - weighted_storage_flows
        else:
            direction = flows - storage_flows

        if weighted:
            max_stepsize = min(weights[key] for key in source)
        else:
            max_stepsize = 1.0

        if linesearch:
            res = minimize_scalar(
                lambda y: model.primal(flows_averaged + y * direction),
                bounds=(0.0, max_stepsize),
                method="bounded",
                tol=1e-12,
            )
            stepsize = res.x
        else:
            stepsize = 2.0 / (k + 2)

        flows_averaged = flows_averaged + stepsize * direction

        if weighted:
            for key in source:
                storage[key] = (1 - stepsize / weights[key]) * storage[key] + stepsize * flows_by_sources[
                    key
                ] / weights[key]
        else:
            for key in source:
                storage[key] = (1 - stepsize) * storage[key] + stepsize * flows_by_sources[key]

        aftertime = time.time()
        time_log.append((time_log[-1] if len(time_log) > 0 else 0) + aftertime - beforetime)

        primal = model.primal(flows_averaged)
        primal_log.append(primal)

        if k % int(count_sources / count_random_correspondences) == 0:
            dual_val = model.dual(times, model.flows_on_shortest(times))
            max_dual_func_val = max(max_dual_func_val, dual_val)
            dgap_log.append(primal - max_dual_func_val)
            relative_gap_log.append((primal - max_dual_func_val) / max_dual_func_val)

        if time_log[-1] > max_time:
            break

        if stop_by_crit and dgap_log[-1] <= eps_abs:
            optimal = True
            break

    return (
        times,
        flows_averaged,
        (dgap_log, np.array(time_log) - time_log[0], {"primal": primal_log, "relative_gap": relative_gap_log}),
        optimal,
    )


def stochastic_correspondences_averaging_frank_wolfe(
    model: BeckmannModel,
    eps_abs: float,
    max_iter: int = 10000,  # 0 for no limit (some big number)
    max_time: int = 60,
    times_start: Optional[np.ndarray] = None,
    stop_by_crit: bool = True,
    use_tqdm: bool = True,
    linesearch: bool = False,
    weighted: bool = False,
    count_random_correspondences: int = 5,
) -> tuple:
    """One iteration == 1 shortest paths call"""

    optimal = False

    if times_start is None:
        times_start = model.graph.ep.free_flow_times.a.copy()

    max_dual_func_val = -np.inf
    dgap_log = []
    time_log = []

    relative_gap_log = []
    primal_log = []

    rng = range(1_000_000) if max_iter == 0 else tqdm(range(max_iter), disable=not use_tqdm)

    flows_averaged, storage = model.flows_on_shortest(times_start, return_flows_by_sources=True)
    full_storage_flow = flows_averaged
    sources = model.correspondences.sources
    count_sources = len(sources)

    node_traffic = model.correspondences.node_traffic_mat
    weights = np.sum(node_traffic, axis=-1)
    weights /= np.sum(weights)

    for k in rng:
        beforetime = time.time()

        times = model.tau(flows_averaged)
        if weighted:
            source = np.random.choice(sources, size=(count_random_correspondences,), replace=False, p=weights)
        else:
            source = np.random.choice(sources, size=(count_random_correspondences,), replace=False)

        flows, flows_by_sources = model.flows_on_shortest(times, sources_indexes=source, return_flows_by_sources=True)
        sub_storage_flows = np.zeros_like(flows)
        for key in source:
            sub_storage_flows += storage[key]
            storage[key] = flows_by_sources[key]  # update storage shortest paths

        full_storage_flow = full_storage_flow + (flows - sub_storage_flows)  # update full storage flows

        if linesearch:  # step to storage direction
            res = minimize_scalar(
                lambda y: model.primal(flows_averaged + y * (full_storage_flow - flows_averaged)),
                bounds=(0.0, 1.0),
                tol=1e-12,
            )
            stepsize = res.x
        else:
            stepsize = 2.0 / (k + 2)

        flows_averaged = flows_averaged + stepsize * (full_storage_flow - flows_averaged)

        aftertime = time.time()
        time_log.append((time_log[-1] if len(time_log) > 0 else 0) + aftertime - beforetime)

        primal = model.primal(flows_averaged)
        primal_log.append(primal)
        if k % int(count_sources / count_random_correspondences) == 0:
            dual_val = model.dual(times, model.flows_on_shortest(times))
            max_dual_func_val = max(max_dual_func_val, dual_val)
            dgap_log.append(primal - max_dual_func_val)
            relative_gap_log.append((primal - max_dual_func_val) / max_dual_func_val)

        if time_log[-1] > max_time:
            break
        if stop_by_crit and dgap_log[-1] <= eps_abs:
            optimal = True
            break

    return (
        times,
        flows_averaged,
        (dgap_log, np.array(time_log) - time_log[0], {"primal": primal_log, "relative_gap": relative_gap_log}),
        optimal,
    )


def stochastic_correspondences_n_conjugate_frank_wolfe(
    model: BeckmannModel,
    eps_abs: float,
    max_iter: int = 100,  # 0 for no limit (some big number)
    max_time: int = 60,
    times_start: Optional[np.ndarray] = None,
    stop_by_crit: bool = False,
    use_tqdm: bool = True,
    linesearch: bool = False,
    cnt_conjugates: int = 3,
    count_random_correspondences: int = 1,
) -> tuple:
    """One iteration == 1 shortest paths call"""

    optimal = False

    # init flows, not used in averaging
    if times_start is None:
        times_start = model.graph.ep.free_flow_times.a.copy()
    flows = model.flows_on_shortest(times_start)

    max_dual_func_val = -np.inf
    dgap_log = []
    time_log = []
    primal_log = []
    relative_gap_log = []

    rng = range(1, 1_000_000) if max_iter == 0 else tqdm(range(1, max_iter), disable=not use_tqdm)

    flows, storage = model.flows_on_shortest(times_start, return_flows_by_sources=True)
    sources = model.correspondences.sources
    count_sources = len(sources)
    node_traffic = model.correspondences.node_traffic_mat

    # raise Exception('AASDd')

    gamma = 1.0
    d_list = []
    S_list = []
    flows_by_sources_list = []
    gamma_list = []
    gamma = 1
    epoch = 0
    for k in rng:
        # print('Iteration : ',k, 'epoch:', epoch)

        # print('TEST STORAGE')
        # test_flow = np.zeros_like(flows)
        # for k, v in storage.items():
        #     test_flow += v
        # print(test_flow - flows )

        # print(len(S_list) , len(flows_by_sources_list))
        beforetime = time.time()
        if gamma > 0.99999:
            epoch = 0
            S_list = []
            d_list = []
            flows_by_sources_list = []

        # print('epoch after filter: ', epoch)
        if k == 1 or epoch == 0:
            # print('FW step')
            epoch = epoch + 1
            t = model.tau(flows)

            ### Original NFW:
            # sk_FW = model.flows_on_shortest(t) --> s_sk_FW = .. (stochastic sk FW)
            # dk = sk_FW - flows --> s_sk_FW - storage_flows
            ###

            ### SNFW
            source = np.random.choice(sources, size=(count_random_correspondences,), replace=False)
            sk_FW, flows_by_sources_FW = model.flows_on_shortest(
                t, sources_indexes=source, return_flows_by_sources=True
            )
            storage_flows = np.zeros_like(sk_FW)
            for key in source:
                storage_flows += storage[key]
            # по сути это вне ограничений, но можно вернуть
            # этот вектор в целевое множество прибавив к нему вектор
            # потоков по оставшимся коррепонденциям в предыдущей точке
            sk = sk_FW
            # Однако dk -- корректно (умный ноль в виде +- потоков по остальным корреспонденциям)
            # dk = sk - storage_flows
            flows_by_sources = flows_by_sources_FW

            for key in storage.keys():
                if key not in flows_by_sources_FW.keys():
                    sk += storage[key]
                    flows_by_sources[key] = storage[key]

            dk = sk - flows

            ###

            ### TEST PART
            # sk_flow = np.zeros_like(flows)
            # for k in storage.keys():
            #     if k in flows_by_sources.keys():
            #         sk_flow += flows_by_sources[k]
            #     else:
            #         sk_flow += storage[k]
            # dk = sk_flow - flows

            d_list.append(dk)
            S_list.append(sk)

            ### SNFW
            flows_by_sources_list.append(flows_by_sources)
            ###
        else:
            # print('NFW step')
            t = model.tau(flows)

            ### ORIGINAL NFW
            # sk_FW = model.flows_on_shortest(t)
            # dk_FW = sk_FW - flows
            ###

            ### SNFW
            source = np.random.choice(sources, size=(count_random_correspondences,), replace=False)
            sk_FW, flows_by_sources_FW = model.flows_on_shortest(
                t, sources_indexes=source, return_flows_by_sources=True
            )

            # storage_flows = np.zeros_like(sk_FW)
            # for key in source:
            #     storage_flows += storage[key]

            for key in storage.keys():
                if key not in flows_by_sources_FW.keys():
                    sk_FW += storage[key]
                    flows_by_sources_FW[key] = storage[key]
            # dk_FW = sk_FW - storage_flows

            dk_FW = sk_FW - flows

            ###

            hessian = model.diff_tau(flows)

            B = np.sum(d_list * hessian * d_list, axis=1)
            A = np.sum(d_list * hessian * dk_FW, axis=1)
            N = len(B)
            betta = [-1] * (N + 1)
            betta_sum = 0
            # delta = 0.0001
            for m in range(N, 0, -1):
                betta[m] = -A[-m] / (B[-m] * (1 - gamma_list[-m])) + betta_sum * gamma_list[-m] / (1 - gamma_list[-m])
                if betta[m] < 0:
                    betta[m] = 0
                else:
                    betta_sum = betta_sum + betta[m]
            alpha_0 = 1 / (1 + betta_sum)
            alpha = np.array(betta)[1:] * alpha_0
            alpha = alpha[::-1]

            sk = alpha_0 * sk_FW + np.sum(alpha * np.array(S_list).T, axis=1)
            flows_by_sources = sum_flow_dicts_with_intersection(
                [flows_by_sources_FW] + list(flows_by_sources_list), weights=[alpha_0] + list(alpha)
            )
            # sk = full_flows_from_dict(flows_by_sources)

            # print('KEYS DIRECTIONS')
            # print(flows_by_sources_FW.keys())
            # for test_flow_dict in flows_by_sources_list :
            #     print(test_flow_dict.keys())
            # print('ALPHAS')
            # print(alpha_0 , alpha)
            # print('RESULT KEYS')
            # print(flows_by_sources.keys())

            # print('TEST SK + STORAGE FEASIBLE')
            # test_flows = full_flows_from_dict(sum_flow_dicts_without_intersection( [flows_by_sources , storage] ))
            # a,b = check_correct_flow(test_flows, model)
            # print(a ,b )
            # raise Exception('TEST')

            # if k == 9:
            #     for i in range(len(flows)):
            #         print(i , ':    '  , sk_FW[i] *alpha_0 + sum([full_flows_from_dict(f)[i] * a for f, a in zip(flows_by_sources_list, alpha)])  , sk[i])

            # if k == 4:
            #     print('TEST LIST OF SK and SK_FW DICTS')

            #     is_correct , corrects = check_correct_flow( sk_FW , model)
            # sk_A_storage_B = full_flows_from_dict(sum_flow_dicts_without_intersection( [flows_by_sources_FW, storage] ))
            # print(sum_flow_dicts_without_intersection( [flow_dict , storage] ))
            # a,b = check_correct_flow(sk_A_storage_B, model)
            # print(a ,b )
            # for flow_dict in flows_by_sources_list:
            # sk_A_storage_B = full_flows_from_dict(sum_flow_dicts_without_intersection( [flow_dict , storage] ))
            # print(sum_flow_dicts_without_intersection( [flow_dict , storage] ))
            # a,b = check_correct_flow(sk_A_storage_B, model)
            # print(a ,b )

            # print('TEST FLOW BY SOURCES')
            # test_flows = full_flows_from_dict(flows_by_sources)
            # print(sk == test_flows)

            ## TEST ALPHAS

            # ### TEST
            # print('TEST S_LIST:')
            # test_flow = np.zeros_like(flows)
            # for k, v in flows_by_sources.items():
            #     test_flow += v
            # print(test_flow - sk )
            # print(len(flows_by_sources.keys()))

            # ## TEST sk FEASIBLE
            # is_correct, corrects - check_correct_flow(full_flows_from_dict())

            # storage_flows = np.zeros_like(flows)
            # for key in flows_by_sources.keys():
            #     storage_flows += storage[key]
            # dk = sk - storage_flows
            # dk = np.zeros_like(flows)
            # for key in flows_by_sources.keys():
            #     dk += flows_by_sources[key] - storage[key]

            dk = sk - flows

            # print('TEST FLOWS + DK FEASIBLE')
            # test_flows = flows + 0.5 * dk
            # a,b = check_correct_flow(test_flows, model)
            # print(a,b)

            # print('COUNT USED CORRS BY SK TEST')
            # print(len(flows_by_sources.keys()))

            # sk_flow = np.zeros_like(flows)
            # for k in storage.keys():
            #     if k in flows_by_sources.keys():
            #         sk_flow += flows_by_sources[k]
            #     else:
            #         sk_flow += storage[k]
            # dk = sk_flow - flows
            # print( sk_flow - flows - dk)
            # raise Exception('TEST')

            d_list.append(dk)
            S_list.append(sk)

            ### SNFW
            flows_by_sources_list.append(flows_by_sources)
            ###

            epoch = epoch + 1
            if epoch > cnt_conjugates:
                d_list.pop(0)
                S_list.pop(0)
                ### SNFW
                flows_by_sources_list.pop(0)
                ###
                gamma_list.pop(0)

        # new_flows = flows_A + flows_B + gamma * (sk_flows_A + sk_flows_B - flows_A - flows_B)
        # sk_flows_B = flows_B
        # new_flows = flows_A + flows_B + gamma * ( sk_flows_A - flows_A )
        # new_flows = old_flows + gamma * dk

        if linesearch:
            res = minimize_scalar(lambda y: model.primal(flows + y * dk), bounds=(0.0, 1.0), tol=1e-12)
            gamma = res.x
        else:
            gamma = 2.0 / (k + 2)

        gamma_list.append(gamma)

        # flows = flows + gamma*(sk - storage_flows)
        flows = flows + gamma * dk

        # print('TEST FEASIBLE SK')
        # is_correct , corrects = check_correct_flow(sk, model )
        # print(is_correct)
        # # ### TEST FEASIBLE flows
        # print('TEST FEASIBLE flows')
        # is_correct , corrects = check_correct_flow(flows, model )
        # print(is_correct)
        # # print(corrects)
        # if not is_correct:
        #     raise Exception('TEST')

        assert check_correct_flow(flows, model)[0]

        ### SNFW update storage
        for key in storage.keys():
            storage[key] = (1 - gamma) * storage[key] + gamma * flows_by_sources[key]

        # print('AFTER: ' , flows - np.array([value for value in storage.values()]).sum(axis = 0) )

        aftertime = time.time()

        time_log.append((time_log[-1] if len(time_log) > 0 else 0) + aftertime - beforetime)

        primal = model.primal(flows)
        # print('Primal: ',primal)
        primal_log.append(primal)
        if k % int(count_sources / count_random_correspondences) == 0:
            dual_val = model.dual(t, model.flows_on_shortest(t))
            max_dual_func_val = max(max_dual_func_val, dual_val)
            dgap_log.append(primal - max_dual_func_val)
            relative_gap_log.append((primal - max_dual_func_val) / max_dual_func_val)
        # print(max_time , time_log[-1])
        if time_log[-1] > max_time:
            break
        if stop_by_crit and dgap_log[-1] <= eps_abs:
            optimal = True
            break

    return (
        t,
        flows,
        (dgap_log, np.array(time_log) - time_log[0], {"primal": primal_log, "relative_gap": relative_gap_log}),
        optimal,
    )
