from pandas import read_csv, to_datetime
import networkx as nx
from ast import literal_eval
from os.path import join
from os import makedirs
import json
from torch_geometric.data import HeteroData
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import gc
from config import MAX_WORKERS, INPUT_PATH
from collections import Counter, defaultdict
from functools import partial
_shared_data = None


def _encode(case_ig):
    global _shared
    name, igs, ohe_activity, ohe_resource, path, max_active_cases, max_running_events = (
        _shared['name'],
        _shared['igs'],
        _shared['ohe_activity'],
        _shared['ohe_resource'],
        _shared['path'],
        _shared['max_active_cases'],
        _shared['max_running_events']
    )

    _, ig = case_ig

    prefix = HeteroData()
    i, all_activities = 0, [d for n, d in ig.nodes(data=True) if d.get('ntype') == 'activity']
    while i < len(all_activities) - 2:
        item = {
            'case_id': ig.case_id,
            'set': ig.set,
            'y': torch.tensor([ohe_activity[all_activities[i + 1].get('activity')]], dtype=torch.float),
            'next_activity': all_activities[i + 1].get('activity'),
            'prefix_size': i + 1,
            'concurrent_events': all_activities[i].get('n_concurrent_events')
        }

        act_ids, activities = map(list, zip(*((node, data) for node, data in ig.nodes(data=True)
                                             if data.get('ntype') == 'activity' and int(float(node.split('_')[1])) <= i)))

        res_ids, resources = map(list, zip(*((node, data.get('resource')) for node, data in ig.nodes(data=True)
                                             if data.get('ntype') == 'resource' and int(float(node.split('_')[1])) <= i)))

        edges = list(ig.subgraph(act_ids + res_ids).edges(data=True))

        resource_mapping = defaultdict(list)
        for res_id, res_label in zip(res_ids, resources):
            resource_mapping[res_label].append(res_id)
        resources = list(resource_mapping.keys())

        ata, rta, rtr = [], [], []
        ata_f, rta_f, rtr_f = [], [], []
        for src, dst, data in edges:
            if data.get('etype') == 'act_to_act':
                src, dst = act_ids.index(src), act_ids.index(dst)
                ata.append([src, dst])
                ata_f.append(literal_eval(f"{data.get('features')}"))

            elif data.get('etype') == 'res_to_res':
                src_label = next((k for k, v in resource_mapping.items() if src in v), None)
                dst_label = next((k for k, v in resource_mapping.items() if dst in v), None)
                src, dst = resources.index(src_label), resources.index(dst_label)
                rtr.append([src, dst])

                rtr_f.append(literal_eval(f"{data.get('features')}"))

            elif data.get('etype') == 'res_to_act':
                src_label = next((k for k, v in resource_mapping.items() if src in v), None)
                src = resources.index(src_label)
                dst = act_ids.index(dst)
                rta.append([src, dst])

                f = literal_eval(f"{data.get('features')}")
                f = [f[0] / max_active_cases, f[1] / max_running_events]
                rta_f.append(f)

        activities_x = list(map(lambda x: ohe_activity[x.get('activity')], activities))
        prefix['activity'].x = torch.tensor(activities_x, dtype=torch.float)

        resources_x = list(map(lambda x: ohe_resource[x], resources))
        prefix['resource'].x = torch.tensor(resources_x, dtype=torch.float)

        prefix['activity', 'act_to_act', 'activity'].edge_index = torch.tensor(ata, dtype=torch.int).T
        prefix['activity', 'act_to_act', 'activity'].edge_attr = torch.tensor(ata_f, dtype=torch.float)

        prefix['resource', 'res_to_res', 'resource'].edge_index = torch.tensor(rtr, dtype=torch.int).T
        prefix['resource', 'res_to_res', 'resource'].edge_attr = torch.tensor(rtr_f, dtype=torch.float)

        prefix['resource', 'res_to_act', 'activity'].edge_index = torch.tensor(rta, dtype=torch.int).T
        prefix['resource', 'res_to_act', 'activity'].edge_attr = torch.tensor(rta_f, dtype=torch.float)

        context = HeteroData()
        inter_cases = literal_eval(all_activities[i].get('concurrent_events'))

        if inter_cases:
            context_act_ids, context_activities = [act_ids[0]], [all_activities[0]]
            context_res_ids, context_resources = [res_ids[0]], [resources[0]]
            context_edges = []

            for conc_case, conc_act_id in inter_cases:
                context_ig = next((g for case, g in igs if case == conc_case), None)

                context_activity = context_ig.nodes.get(f'act_{conc_act_id}')
                context_act_id = f'act_{conc_case}_{conc_act_id}'

                context_resource = context_ig.nodes.get(f'res_{conc_act_id}')
                context_res_id = f'res_{conc_case}_{conc_act_id}'

                context_activities.append(context_activity)
                context_act_ids.append(context_act_id)

                context_resources.append(context_resource.get('resource'))
                context_res_ids.append(context_res_id)

                # start_act to context_act
                data = {'etype': 'act_to_act', 'features': f"{context_activities[0].get('temporal_features')}"}
                context_edges.append((context_act_ids[0], context_act_id, data))

                # start_res to context_res
                data = {'etype': 'res_to_res', 'features': f"{context_activities[0].get('resource_features')}"}
                context_edges.append((context_res_ids[0], context_res_id, data))

                # conc_res to context_act
                data = {'etype': 'res_to_act', 'features': f"{context_activity.get('case_features')}"}
                context_edges.append((context_res_id, context_act_id, data))

            resource_mapping = defaultdict(list)
            for res_id, res_label in zip(context_res_ids, context_resources):
                resource_mapping[res_label].append(res_id)
            resources = list(resource_mapping.keys())

            ata, rta, rtr = [], [], []
            ata_f, rta_f, rtr_f = [], [], []
            for src, dst, data in context_edges:
                if data.get('etype') == 'act_to_act':
                    src, dst = context_act_ids.index(src), context_act_ids.index(dst)
                    ata.append([src, dst])
                    ata_f.append(literal_eval(f"{data.get('features')}"))

                elif data.get('etype') == 'res_to_res':
                    src_label = next((k for k, v in resource_mapping.items() if src in v), None)
                    dst_label = next((k for k, v in resource_mapping.items() if dst in v), None)
                    src, dst = resources.index(src_label), resources.index(dst_label)
                    rtr.append([src, dst])

                    rtr_f.append(literal_eval(f"{data.get('features')}"))

                elif data.get('etype') == 'res_to_act':
                    src_label = next((k for k, v in resource_mapping.items() if src in v), None)
                    src = resources.index(src_label)
                    dst = context_act_ids.index(dst)
                    rta.append([src, dst])

                    f = literal_eval(f"{data.get('features')}")
                    f = [f[0] / max_active_cases, f[1] / max_running_events]
                    rta_f.append(f)

        else:
            context_activities = ['dummy']
            context_resources = ['dummy']

            ata_f = [[0]*len(literal_eval(activities[0].get('temporal_features')))]
            ata = [[0, 0]]

            rta_f = [[0]*len(literal_eval(activities[0].get('case_features')))]
            rta = [[0, 0]]

            rtr_f = [[0]*len(literal_eval(activities[0].get('resource_features')))]
            rtr = [[0, 0]]


        context_activities_x = list(map(lambda x: [0] * (len(ohe_activity)) if x == 'dummy' else
        ohe_activity[x.get('activity')], context_activities))
        context['activity'].x = torch.tensor(context_activities_x, dtype=torch.float)

        context_resources_x = list(map(lambda x: [0] * (len(ohe_resource)) if x == 'dummy' else
        ohe_resource[x], context_resources))
        context['resource'].x = torch.tensor(context_resources_x, dtype=torch.float)

        context['activity', 'act_to_act', 'activity'].edge_index = torch.tensor(ata, dtype=torch.int).T
        context['activity', 'act_to_act', 'activity'].edge_attr = torch.tensor(ata_f, dtype=torch.float)

        context['resource', 'res_to_res', 'resource'].edge_index = torch.tensor(rtr, dtype=torch.int).T
        context['resource', 'res_to_res', 'resource'].edge_attr = torch.tensor(rtr_f, dtype=torch.float)

        context['resource', 'res_to_act', 'activity'].edge_index = torch.tensor(rta, dtype=torch.int).T
        context['resource', 'res_to_act', 'activity'].edge_attr = torch.tensor(rta_f, dtype=torch.float)

        item.update({
            'concurrent_events': len(inter_cases),
            'prefix': prefix,
            'context': context
        })

        if i > 0:
            torch.save(item, join(path, f"{name}_{item.get('case_id')}_{item.get('prefix_size')}.pt"))

        i += 1


def encode_prefix_igs(igs, shared_items):
    f = partial(_encode)

    with ProcessPoolExecutor(max_workers=int(MAX_WORKERS/2), initializer=init_worker, initargs=(shared_items,)) as executor:
        for _ in tqdm(executor.map(f, igs), total=len(igs), desc='Encoding prefix-IGs'):
            pass

    with open(join(shared_items.get('path'), 'done'), 'w') as _:
        pass


def _discovery(case_id):
    global _shared
    resources_counter, cases_time, nodes, igs, activities_to_filter = (
        _shared['resources_counter'],
        _shared['cases_time'],
        _shared['nodes'],
        _shared['igs'],
        _shared['activities_to_filter']
    )

    max_active_case, max_running_events = 0, 0

    ig = igs.loc[igs['case_id'] == case_id]
    g = nx.MultiDiGraph()
    g.case_id, g.set = case_id, ig['set'].drop_duplicates().tolist()[0]

    for event in ig.itertuples(index=False):
        if event.type == 'v':
            event_start_time = to_datetime(event.start_time, format='ISO8601', utc=True)
            # get concurrent cases and remove itself
            start_time = to_datetime(event.start_time)
            active_cases = set(cases_time.loc[
                                   (event.set == cases_time['set']) &
                                   (start_time <= cases_time['max']) &
                                   (case_id != cases_time['case_id']), 'case_id'].tolist())

            # keep concurrent events, i.e., events that run when x runs, except end activities
            concurrent_events = nodes.loc[(
                    (nodes['case_id'].isin(active_cases)) &
                    (~nodes['activity'].isin(activities_to_filter)) &
                    (nodes['start_time'] <= start_time) & (start_time <= nodes['end_time'])
            )]

            concurrent_events['duration'] = (event_start_time - concurrent_events['start_time']).dt.total_seconds()
            median = concurrent_events['duration'].median()
            concurrent_events = concurrent_events.loc[concurrent_events['duration'] <= median]
            concurrent_nodes = list(zip(concurrent_events['case_id'], concurrent_events['node1']))

            resources_workload = Counter(dict.fromkeys(resources_counter, 0))
            resources_workload.update(concurrent_events['resource'].tolist())

            resources_workload = {}
            for resource, workload in resources_workload.items():
                try:
                    resources_workload[resource] = workload / resources_counter[resource]
                except ZeroDivisionError:
                    resources_workload[resource] = workload

            n_active_cases = len(concurrent_events['case_id'].drop_duplicates())
            max_active_case = n_active_cases if n_active_cases > max_active_case else max_active_case

            n_concurrent_events = len(concurrent_events)
            max_running_events = n_concurrent_events if n_concurrent_events > max_running_events else max_running_events

            temporal_features = [event.norm_time, event.trace_time, event.prev_event_time]
            case_features = [n_active_cases, n_concurrent_events]
            resource_features = list(resources_workload.values())

            g.add_node(f'act_{event.node1}', activity=event.activity, resource=event.resource,
                       n_concurrent_events=n_concurrent_events,
                       concurrent_events=f'{concurrent_nodes}',
                       temporal_features=f'{temporal_features}',
                       case_features=f'{case_features}',
                       resource_features=f'{resource_features}',
                       ntype='activity'
                       )

            g.add_node(f'res_{event.node1}', resource=event.resource, ntype='resource')
            g.add_edge(f'res_{event.node1}', f'act_{event.node1}', etype='res_to_act',
                       features=g.nodes.get(f'act_{event.node1}').get('case_features'))

        elif event.type == 'e':
            g.add_edge(f'act_{event.node1}', f'act_{event.node2}', etype='act_to_act',
                       features=g.nodes.get(f'act_{event.node1}').get('temporal_features'))

            g.add_edge(f'res_{event.node1}', f'res_{event.node2}', etype='res_to_res',
                       features=g.nodes.get(f'act_{event.node1}').get('resource_features'))

    t = (case_id, g)
    return t, g.set, max_active_case, max_running_events


def discovery_inter_cases(cases, shared_items):
    f = partial(_discovery)
    igs = []
    max_active_cases, max_running_events = [], []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker, initargs=(shared_items,)) as executor:
        for out in tqdm(executor.map(f, cases), total=len(cases), desc='Discovering inter cases'):
            if out is not None:
                igs.append(out[0])
                if out[1] == 'train':
                    max_active_cases.append(out[2])
                    max_running_events.append(out[3])

    return igs, max(max_active_cases), max(max_running_events)


def get_ohe_encoding(name, items, attribute):
    try:
        with open(join(INPUT_PATH, f'{name}_ohe_{attribute}.json'), 'r') as f:
            ohe = json.load(f)

    except (Exception, ):
        ohe = {item:[1 if i == j else 0 for j in range(len(items))] for i, item in enumerate(items)}
        with open(join(INPUT_PATH, f'{name}_ohe_{attribute}.json'), 'w') as f:
            f.write(json.dumps(ohe, indent=1))

    return ohe


def init_worker(shared_data):
    global _shared
    _shared = shared_data


def main(name):
    print('Reading dataset')
    igs = read_csv(join(INPUT_PATH, f'{name}_processed.g'), header=0, sep=',')

    if 'Helpdesk_no_resources' in name:
        igs['resource'] = 'artificial_resource'

    igs.rename(columns={'org_resource': 'resource'}, inplace=True)
    igs.loc[(igs['type'] == 'v') & (igs['resource'].isna()), 'resource'] = 'artificial_resource'

    nodes = igs.loc[igs['type'] == 'v']
    case_ids = nodes['case_id'].drop_duplicates().tolist()

    nodes['start_time'] = to_datetime(nodes['start_time'], format='ISO8601', utc=True)
    nodes['end_time'] = to_datetime(nodes['end_time'], format='ISO8601', utc=True)
    cases_time = nodes.groupby('case_id').agg(
        min=('end_time', 'min'),
        max=('end_time', 'max'),
        set=('set', 'first')
    ).reset_index()

    ohe_activity = get_ohe_encoding(name, nodes['activity'].drop_duplicates().tolist(), 'activity')
    ohe_resource = get_ohe_encoding(name, nodes['resource'].drop_duplicates().tolist(), 'resource')

    activities_to_filter = ['artificial_start', 'artificial_end']
    nodes_for_counter = nodes.loc[(nodes['set'] == 'train') & (~nodes['activity'].isin(activities_to_filter))]
    resources_counter = Counter(nodes_for_counter['resource'].tolist())

    shared_items = {
        'name': name,
        'igs': igs,
        'activities_to_filter': activities_to_filter,
        'cases_time': cases_time,
        'nodes': nodes,
        'ohe_activity': ohe_activity,
        'ohe_resource': ohe_resource,
        'resources_counter': resources_counter,
    }
    igs, max_active_cases, max_running_events = discovery_inter_cases(case_ids, shared_items)

    del nodes, shared_items
    gc.collect()

    path = join('dataset', f'{name}_tensors')
    makedirs(path, exist_ok=True)

    shared_items = {
        'name': name,
        'igs': igs,
        'ohe_activity': ohe_activity,
        'ohe_resource': ohe_resource,
        'max_active_cases': max_active_cases,
        'max_running_events': max_running_events,
        'resources_counter': resources_counter,
        'path': path
    }

    encode_prefix_igs(igs, shared_items)

    del shared_items, igs
    gc.collect()


if __name__ == '__main__':
    main('Helpdesk_no_resources')
