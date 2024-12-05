
import numba
from numba.core import types
import numpy as np


def check_correct_flow(flow, model ):    
    num_nodes, num_edges = model.graph.num_vertices(), model.graph.num_edges()
    traffic_mat, sources, targets = model.correspondences.traffic_mat, model.correspondences.sources, model.correspondences.targets
    edges_arr = model.graph.get_edges()
    edge_to_ind = numba.typed.Dict.empty(
        key_type=types.UniTuple(types.int64, 2), value_type=numba.core.types.int64
    )
    for i, edge in enumerate(edges_arr):
        edge_to_ind[tuple(edge)] = i

    node_traffic = model.correspondences.node_traffic_mat
    edges = [ e for e in model.graph.edges()]    
    vertices = [ e for e in model.graph.vertices()]
    
    corrects = [] 
    for j in range(len(vertices)):
        value = 0
        traffic_from_j = sum(traffic_mat[j]) - sum(traffic_mat[:,j])
        v = vertices[j]
        
        for e_out in v.out_edges():
            e_out_ind = edge_to_ind[(int(e_out.source()) ,int(e_out.target()))]
            value += flow[e_out_ind]
        for e_in in v.in_edges():
            e_in_ind = edge_to_ind[(int(e_in.source()) ,int(e_in.target()))]
            value -= flow[e_in_ind]
        corrects.append((j , traffic_from_j - value))
    corrects = np.array(corrects)
    return np.all(np.abs(corrects[:,1]) < 1e-8 ) , corrects


def full_flows_from_dict( flows_dict ):
    assert len(flows_dict.keys()) != 0
    flows = None
    for key in flows_dict.keys():
        if flows is None:
            flows = flows_dict[key]
        else:
            flows += flows_dict[key]
    return flows

def sum_flow_dicts( flow_dicts , weights = None):
    new_flow_dict = dict()
    for i, flow_dict in enumerate(flow_dicts):
        if weights is not None:
            weight = weights[i]
        else:
            weight = 1

        for key in flow_dict.keys():
            
            if key not in new_flow_dict.keys():
                new_flow_dict[key] = weight * flow_dict[key]
            else:
                new_flow_dict[key] += weight * flow_dict[key]
    return new_flow_dict

def sum_flow_dicts_without_intersection( flow_dicts , weights = None ):
    new_flow_dict = dict()
    current_keys = []
    # print('test full sum', weights)
    for i, flow_dict in enumerate(flow_dicts):
        if weights is not None:
            weight = weights[i]
        else:
            weight = 1
        # print('Add')
        for key in flow_dict.keys():
            if key not in current_keys:
                
                # print(key , weight , flow_dict[key])
                new_flow_dict[key] = weight * flow_dict[key]
                current_keys.append(key)
        # print('keys curr:' , current_keys)
    return new_flow_dict

def sum_flow_dicts_with_intersection( flow_dicts , weights = None):
    new_flow_dict = dict()
    key_sum_weights = dict()
    # print('TEST SUMM ', weights)
    for i, flow_dict in enumerate(flow_dicts):
        if weights is not None:
            weight = weights[i]
        else:
            weight = 1

        # print('Add', weight)
        # print('dict for add:', flow_dict)
        if weight != 0:
            for key in flow_dict.keys():
                if key not in new_flow_dict.keys():
                    new_flow_dict[key] = weight * flow_dict[key]
                else:
                    new_flow_dict[key] += weight * flow_dict[key]
            
                if key not in key_sum_weights.keys():
                    key_sum_weights[key] = weight 
                else:
                    key_sum_weights[key] += weight

    # print('keySUMweight:' , key_sum_weights)
    for key in new_flow_dict.keys():
        if key_sum_weights[key] != 0 :
            new_flow_dict[key] /= key_sum_weights[key]
    
    # print('result dict',new_flow_dict)

    return new_flow_dict


## TEST
# flow_dicts = [{'1': 20 ,'2': 40}, {'2': 40 , '3': 60}, { '3': 60 , '4': 20 }]

# print(sum_flow_dicts(flow_dicts))

# print(sum_flow_dicts_with_intersection( flow_dicts , weights = [0.5 , 0.5 , 0.0]))

# print(sum_flow_dicts_without_intersection( flow_dicts ))
