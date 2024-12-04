
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