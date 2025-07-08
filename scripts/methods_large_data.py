import sys, os
import numpy as np
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import src.test as test
from src.algs import subgd, ustm, frank_wolfe, cyclic

# from src.my_algs import conjugate_frank_wolfe , Bi_conjugate_frank_wolfe , N_conjugate_frank_wolfe ,fukushima_frank_wolfe
from src.algs import (
    N_conjugate_frank_wolfe,
    stochastic_correspondences_frank_wolfe,
    stochastic_correspondences_averaging_frank_wolfe,
    stochastic_correspondences_n_conjugate_frank_wolfe,
)


networks_path = Path("./TransportationNetworks")


# folder = "SiouxFalls"
# net_name = "SiouxFalls_net"
# traffic_mat_name = "SiouxFalls_trips"

# folder = "Anaheim"
# net_name = "Anaheim_net"
# traffic_mat_name = "Anaheim_trips"

# Не работает (mu != inf , но rho = 0) (sigma * = ... / rho ...)
# folders.append("Philadelphia")
# net_names.append("Philadelphia_net")
# traffic_mat_names.append("Philadelphia_trips")

# rho = 0
# folder = "Berlin-Tiergarten"
# net_name = 'berlin-tiergarten_net'
# traffic_mat_name = 'berlin-tiergarten_trips'

# folder = "Terrassa-Asymmetric"
# net_name = 'Terrassa-Asym_net'
# traffic_mat_name = 'Terrassa-Asym_trips'

# folder = "Eastern-Massachusetts"
# net_name = 'EMA_net'
# traffic_mat_name = 'EMA_trips'

# rho = 0 and fft = 0
# folder = "Chicago-Sketch"
# net_name = 'ChicagoSketch_net'
# traffic_mat_name = 'ChicagoSketch_trips'


# rho = 0
# folder = "Berlin-Mitte-Center"
# net_name = 'berlin-mitte-center_net'
# traffic_mat_name = 'berlin-mitte-center_trips'

# Не работает (ибо пока архитектура не учитывает дуги )
# folder = "Berlin-Center"
# net_name = 'berlin-center_net'
# traffic_mat_name = "berlin-center_trips"

# folder = "Berlin-Friedrichshain"
# net_name = "friedrichshain-center_net"
# traffic_mat_name = "friedrichshain-center_trips"

# key error  в sum_flows_from_tree
# folder = "Winnipeg-Asymmetric"
# net_name = 'Winnipeg-Asym_net'
# traffic_mat_name = "Winnipeg-Asym_trips"

# folder = "Winnipeg"
# net_name = 'Winnipeg_net'
# traffic_mat_name = "Winnipeg_trips"

# folder = "Austin"
# net_name = 'Austin_net'
# traffic_mat_name = "Austin_trips"

# rho = 0
# folder = "Barcelona"
# net_name = 'Barcelona_net'
# traffic_mat_name = "Barcelona_trips"

# folder = "Berlin-Mitte-Prenzlauerberg-Friedrichshain-Center"
# net_name = 'berlin-mitte-prenzlauerberg-friedrichshain-center_net'
# traffic_mat_name = "berlin-mitte-prenzlauerberg-friedrichshain-center_trips"

# folder = "Hessen-Asymmetric"
# net_name = 'Hessen-Asym_net'
# traffic_mat_name = "Hessen-Asym_trips"

# rho = 0
# folder = "GoldCoast"
# net_name = 'Goldcoast_network_2016_01'
# traffic_mat_name = "Goldcoast_trips_2016_01"

# Key error
# folder = "Sydney"
# net_name = 'Sydney_net'
# traffic_mat_name = "Sydney_trips"

# Key Error
# folder = "Austin"
# net_name = 'Austin_net'
# traffic_mat_name = "Austin_trips_am"


# folders.append("Birmingham-England")
# net_names.append("Birmingham_Net")
# traffic_mat_names.append("Birmingham_Trips")

# folders.append("chicago-regional")
# net_names.append("ChicagoRegional_net")
# traffic_mat_names.append("ChicagoRegional_trips")


# print(type(print(beckmann_model.correspondences)))
# print(beckmann_model.correspondences.traffic_mat.shape)
# print(beckmann_model.correspondences.node_traffic_mat.shape)
# print(beckmann_model.correspondences.sources.shape)
# print(beckmann_model.correspondences.targets.shape)

# from collections import Counter

# for k ,v in Counter(beckmann_model.graph.ep.mu.a).items() :
#     print(k , v)

# print('asdad')

# for k ,v in Counter(beckmann_model.graph.ep.rho).items() :
#     print(k , v)


# ##EXPERIMENTS RUN


# print(beckmann_model.correspondences.sources


# print(node_traffic)

# print(np.sum(node_traffic, axis= -1))

# print()


# raise Exception('STOP')

## Stochastic correspondences FW (SCFW)


# list_methods.append((stochastic_correspondences_averaging_frank_wolfe ,f'stochastic correspondences averaging FW  corrs = {cnt/ num_of_sources}' ,
#     {'eps_abs' : eps_abs , 'max_iter': max_iter , 'max_time': max_time , 'stop_by_crit': False ,'linesearch':False, f'count_random_correspondences': cnt }  ))

# list_methods.append((stochastic_correspondences_averaging_frank_wolfe ,f'stochastic correspondences averaging FW linesearch corrs = {cnt/ num_of_sources}' ,
#     {'eps_abs' : eps_abs , 'max_iter': max_iter , 'max_time': max_time , 'stop_by_crit': False ,'linesearch':True, 'count_random_correspondences': cnt }  ))


# list_methods.append((stochastic_correspondences_frank_wolfe ,f'stochastic correspondences FW corrs = {cnt/ num_of_sources}' ,
#     {'eps_abs' : eps_abs , 'max_iter': max_iter , 'max_time': max_time , 'stop_by_crit': False ,'linesearch':False, 'count_random_correspondences': cnt }  ))

# list_methods.append((frank_wolfe ,'frank_wolfe' ,
#     {'eps_abs' : eps_abs , 'max_iter': max_iter , 'max_time': max_time , 'stop_by_crit': False} ))


## Stochastic correspondences FW (SCFW) linesearch
# for cnt in [100]:
#     list_methods.append((stochastic_correspondences_frank_wolfe ,'stochastic correspondences FW linesearch' ,
#         {'eps_abs' : eps_abs , 'max_iter':max_iter , 'stop_by_crit': False ,'linesearch':True, 'count_random_correspondences': cnt }  ))

## FUKUSHIMA FW

# weight_param = [0.1]
# for weight in weight_param :
#     list_methods.append((fukushima_frank_wolfe ,'fukushima_frank_wolfe linesearch weighted =' + str(weight) ,
#         {'eps_abs' : eps_abs , 'max_iter':max_iter , 'stop_by_crit': False , 'linesearch': True  , 'weight_parameter' : weight  } ))
# cnts = [4,5,6]
# for cnt in cnts :
#     list_methods.append((fukushima_frank_wolfe ,'fukushima_frank_wolfe linesearch N =' + str(cnt) ,
#         {'eps_abs' : eps_abs , 'max_iter':max_iter , 'stop_by_crit': False , 'linesearch': True  , 'cnt_directional' : cnt  } ))

##NFW
# cnts = [3]
# for cnt in cnts :
#     list_methods.append((N_conjugate_frank_wolfe ,'N_conjugate_frank_wolfe linesearch N =' + str(cnt) ,
#         {'eps_abs' : eps_abs , 'max_iter':max_iter , 'stop_by_crit': False , 'linesearch': True  , 'cnt_conjugates' : cnt  } ))

# ##BCFW linesearch
# list_methods.append((Bi_conjugate_frank_wolfe ,'Bi_conjugate_frank_wolfe linesearch' ,
#     {'eps_abs' : eps_abs , 'max_iter':max_iter , 'stop_by_crit': False , 'linesearch': True } ))
# # ##CFWM linesearch
# list_methods.append((conjugate_frank_wolfe ,'conjugate_frank_wolfe linesearch' ,
#     {'eps_abs' : eps_abs , 'max_iter':max_iter , 'stop_by_crit': False , 'linesearch': True  ,'alpha_default' : 0.6} ))
# # ##CFWM
# list_methods.append((conjugate_frank_wolfe ,'conjugate_frank_wolfe' ,
#     {'eps_abs' : eps_abs , 'max_iter':max_iter , 'stop_by_crit': False} ))
# # ## FWM
# list_methods.append((frank_wolfe ,'frank_wolfe' ,
#     {'eps_abs' : eps_abs , 'max_iter':max_iter , 'stop_by_crit': False} ))
# # # # ## FWM linesearch
# list_methods.append((frank_wolfe ,'frank_wolfe linesearch' ,
#     {'eps_abs' : eps_abs , 'max_iter':max_iter , 'stop_by_crit': False ,'linesearch':True}  ))
# # method , name , solver_kwargs = list_methods[0]
# # result = test.run_method(method , name , solver_kwargs , beckmann_model ,city_name = folder , max_iter = max_iter)

# # dgaps =result[0][0]['duality_gap']
# # steps = result[1]


# # test.plt.figure(figsize = (20, 20))
# # test.plt.plot(dgaps)
# # test.plt.scatter(steps , dgaps[steps] , color = 'red')
# # test.plt.yscale('log')
# # test.plt.show()
# # plt.plot(result[''])


folders = []
net_names = []
traffic_mat_names = []

# folders.append("SiouxFalls")
# net_names.append("SiouxFalls_net")
# traffic_mat_names.append("SiouxFalls_trips")

# folders.append("")
# net_names.append("")
# traffic_mat_names.append("")

folders.append("Birmingham-England")
net_names.append("Birmingham_Net")
traffic_mat_names.append("Birmingham_Trips")

folders.append("chicago-regional")
net_names.append("ChicagoRegional_net")
traffic_mat_names.append("ChicagoRegional_trips")

folders.append("GoldCoast")
net_names.append("Goldcoast_network_2016_01")
traffic_mat_names.append("Goldcoast_trips_2016_01")

folders.append("Philadelphia")
net_names.append("Philadelphia_net")
traffic_mat_names.append("Philadelphia_trips")

folders.append("Chicago-Sketch")
net_names.append("ChicagoSketch_net")
traffic_mat_names.append("ChicagoSketch_trips")


for folder, net_name, traffic_mat_name in zip(folders, net_names, traffic_mat_names):
    ## LOAD CITY
    beckmann_model, city_info = test.init_city(
        networks_path=networks_path, folder=folder, net_name=net_name, traffic_mat_name=traffic_mat_name
    )
    eps_abs = city_info["eps_abs"]

    print("Number of sources", len(beckmann_model.correspondences.sources))
    num_of_sources = len(beckmann_model.correspondences.sources)
    node_traffic = beckmann_model.correspondences.node_traffic_mat
    # print(node_traffic)
    # print(node_traffic.shape)
    # raise Exception('TESt')
    max_iter = 11112

    max_time = 120

    N = 3

    list_methods = []

    # list_methods.append((N_conjugate_frank_wolfe ,f'NFW N = {N} ' ,
    #     {'eps_abs' : eps_abs , 'max_iter': max_iter , 'max_time': max_time , 'stop_by_crit': False ,'linesearch':True}  ))

    list_methods.append(
        (
            frank_wolfe,
            "frank_wolfe linesearch",
            {
                "eps_abs": eps_abs,
                "max_iter": max_iter,
                "max_time": max_time,
                "linesearch": True,
                "stop_by_crit": False,
            },
        )
    )
    for cnt in [ int(num_of_sources/k ) for k in range(8,13)  ]:
        list_methods.append(
            (
                stochastic_correspondences_frank_wolfe,
                f"stochastic correspondences FW linesearch corrs = {cnt / num_of_sources:.3f}",
                {
                    "eps_abs": eps_abs,
                    "max_iter": max_iter,
                    "max_time": max_time,
                    "stop_by_crit": False,
                    "linesearch": True,
                    "count_random_correspondences": cnt,
                },
            )
        )

        list_methods.append(
            (
                stochastic_correspondences_frank_wolfe,
                f"stochastic correspondences FW linesearch weighted corrs = {cnt / num_of_sources:.3f}",
                {
                    "eps_abs": eps_abs,
                    "max_iter": max_iter,
                    "max_time": max_time,
                    "stop_by_crit": False,
                    "linesearch": True,
                    "weighted": True,
                    "count_random_correspondences": cnt,
                },
            )
        )


        # list_methods.append((stochastic_correspondences_n_conjugate_frank_wolfe ,f'stochastic correspondences NFW N = {N} linesearch corrs = {cnt/ num_of_sources}' ,
        #     {'eps_abs' : eps_abs , 'max_iter': max_iter , 'max_time': max_time , 'stop_by_crit': False ,'linesearch':True, 'count_random_correspondences': cnt }  ))

    experiments = test.run_experiment(list_methods, model=beckmann_model, city_name=folder, max_iter=max_iter)

    # print(experiments)

    # raise Exception('ad')
    # #DISPLAY RESULTS
    # test.plot( experiments , name_output_values=['primal'] , save=True  ,time_iters=False)
    test.plot(experiments, name_output_values=["relative_gap"], save=True, time_iters=True, loglog=False)
# test.plot( experiments , name_output_values=['primal', 'relative_gap'] , save=False  ,time_iters=False)
# test.plot( experiments , name_output_values=['primal'] , save=False  ,time_iters=True,loglog=False,last_quantile=1)
