"""Strict PartNeXt annotation contract: GLB mesh-local faces and hierarchy maskId.

Never infer mask identity from hierarchy traversal position, nor flatten local
face IDs before associating them with the original GLB mesh index.
"""
import io
import json
import struct
from pathlib import Path
import numpy as np
import trimesh

ANNOTATION_POLICY = 'gltf_mesh_local_maskid_v1'


def object_json(value):
    return json.loads(value) if isinstance(value, str) else value


def integer_key(value):
    if isinstance(value, bool):
        raise ValueError('Boolean annotation ID')
    number = int(value)
    if number < 0 or (isinstance(value, float) and value != number):
        raise ValueError(f'Invalid nonnegative annotation ID: {value}')
    return str(number)


def canonical_masks(value):
    raw = object_json(value)
    if not isinstance(raw, dict) or not raw:
        raise ValueError('Missing annotation masks')
    result = {}
    for key, submeshes in raw.items():
        key = integer_key(key)
        if key in result or not isinstance(submeshes, dict):
            raise ValueError('Duplicate mask ID or invalid submesh mapping')
        result[key] = {}
        for mesh_id, indices in submeshes.items():
            mesh_id = integer_key(mesh_id)
            if mesh_id in result[key]:
                raise ValueError('Duplicate mesh ID')
            if isinstance(indices, int):
                indices = [indices]
            a = np.asarray(indices)
            if a.ndim != 1 or (a.size and (not np.issubdtype(a.dtype, np.integer) or a.min() < 0)):
                raise ValueError('Face IDs must be nonnegative integers')
            result[key][mesh_id] = a.astype(np.int64)
    return result


def partition_masks(masks, hierarchy, max_parts):
    """Keep every mask if possible; otherwise merge only complete tree subtrees."""
    if max_parts < 1:
        raise ValueError('max_parts must be positive')
    keys = set(masks)
    nodes = object_json(hierarchy) if hierarchy else []
    if isinstance(nodes, dict):
        nodes = [nodes]
    candidates, occurrences, names = [], {}, {}
    def walk(node):
        descendants = set()
        if node.get('maskId') is not None:
            mid = integer_key(node['maskId'])
            if mid in keys:
                occurrences[mid] = occurrences.get(mid, 0) + 1
                names[mid] = node.get('name', f'mask {mid}')
                descendants.add(mid)
        for child in node.get('children', []):
            descendants |= walk(child)
        if descendants:
            candidates.append((frozenset(descendants), node.get('nodeId'), node.get('name', 'group')))
        return descendants
    for node in nodes:
        walk(node)
    duplicate_references = sorted([mid for mid,count in occurrences.items() if count>1],key=int)
    if duplicate_references and len(keys) > max_parts:
        raise ValueError('A raw maskId is referenced by multiple hierarchy nodes')
    groups = [frozenset([k]) for k in sorted(keys, key=int)]
    merges = []
    if len(groups) > max_parts and set(occurrences) != keys:
        raise ValueError('Over-limit masks cannot be merged: hierarchy does not cover every maskId')
    while len(groups) > max_parts:
        options = []
        for subtree, node_id, name in candidates:
            inside = [g for g in groups if g <= subtree]
            if len(inside) < 2 or frozenset().union(*inside) != subtree:
                continue
            options.append((len(inside)-1, len(subtree), tuple(sorted(map(int, subtree))), subtree, node_id, name))
        if not options:
            raise ValueError('No complete hierarchy subtree can satisfy the part limit without dropping masks')
        _, _, _, subtree, node_id, name = min(options, key=lambda x: x[:3])
        groups = [g for g in groups if not g <= subtree] + [subtree]
        merges.append({'node_id': node_id, 'name': name, 'mask_ids': sorted(subtree, key=int)})
    groups.sort(key=lambda g: min(map(int, g)))
    mapping = {mid: label for label, group in enumerate(groups, 1) for mid in group}
    if set(mapping) != keys or len(mapping) != sum(map(len, groups)):
        raise ValueError('Hierarchy partition lost or duplicated masks')
    info = [{'label': label, 'mask_ids': sorted(group, key=int),
             'names': [names.get(mid, f'mask {mid}') for mid in sorted(group, key=int)]}
            for label, group in enumerate(groups, 1)]
    report = {'raw_parts': len(keys), 'parts': len(groups), 'groups': info, 'merges': merges}
    if duplicate_references:
        report['hierarchy_duplicate_references_ignored_without_merging'] = duplicate_references
    return mapping, report


def decode_dense_masks(masks, mesh_counts):
    """Recognize full-length binary masks only when every entry has this form.

    Require length >2 to distinguish the legitimate index list [0,1].
    The two dense-vector source objects omit mesh_face_num; GLB counts provide
    the independent expected vector length. Raw records are never rewritten.
    """
    entries=[(sm,a) for group in masks.values() for sm,a in group.items()]
    if entries and all(sm in mesh_counts and mesh_counts[sm]>2 and len(a)==mesh_counts[sm]
                       and np.isin(a,[0,1]).all() for sm,a in entries):
        return {mid:{sm:np.flatnonzero(a).astype(np.int64) for sm,a in group.items()}
                for mid,group in masks.items()},True
    return masks,False


def local_labels(masks, mesh_counts, mapping):
    """Assign before concatenation; reject overlaps between distinct raw masks."""
    labels = {key: np.zeros(count, np.int32) for key, count in mesh_counts.items()}
    owner = {key: np.full(count, -1, np.int64) for key, count in mesh_counts.items()}
    for mask_id in sorted(masks, key=int):
        for mesh_id, indices in masks[mask_id].items():
            if mesh_id not in labels:
                raise ValueError(f'Annotation references unknown GLB mesh index {mesh_id}')
            if indices.size and indices.max() >= mesh_counts[mesh_id]:
                raise ValueError(f'Local face index exceeds mesh {mesh_id} face count')
            indices = np.unique(indices)
            if np.any((owner[mesh_id][indices] >= 0) & (owner[mesh_id][indices] != int(mask_id))):
                raise ValueError(f'Overlapping raw part masks on mesh {mesh_id}')
            owner[mesh_id][indices] = int(mask_id)
            labels[mesh_id][indices] = mapping[mask_id]
    for key in labels:
        if not np.array_equal(owner[key] >= 0, labels[key] > 0):
            raise ValueError('Grouping lost annotated face coverage')
    return labels


def tagged_glb(path):
    """Carry original mesh/node indices through trimesh, independent of names/order.

    Only the in-memory GLB JSON is tagged; original files stay untouched. Multiple
    primitives are merged in their original order. Scene transforms are applied
    by trimesh rather than manually assuming identity transforms.
    """
    blob = Path(path).read_bytes()
    magic, version, total = struct.unpack_from('<4sII', blob)
    length, kind = struct.unpack_from('<II', blob, 12)
    if magic != b'glTF' or version != 2 or total != len(blob) or kind != 0x4E4F534A:
        raise ValueError('Expected a complete GLB v2 with leading JSON chunk')
    header = json.loads(blob[20:20+length])
    counts = {}
    for i, mesh in enumerate(header.get('meshes', [])):
        count = 0
        for primitive in mesh.get('primitives', []):
            if primitive.get('mode', 4) != 4:
                raise ValueError('Only triangle primitives are supported for annotation indexing')
            accessor = primitive.get('indices', primitive['attributes']['POSITION'])
            n = header['accessors'][accessor]['count']
            if n % 3:
                raise ValueError('Non-triangular primitive index count')
            count += n // 3
        counts[str(i)] = count
        mesh['extras'] = dict(mesh.get('extras', {}), segpart_mesh_index=i)
    for i, node in enumerate(header.get('nodes', [])):
        node['name'] = f'segpart_node_{i:08d}'
    chunk = json.dumps(header, separators=(',', ':')).encode()
    chunk += b' ' * (-len(chunk) % 4)
    tail = blob[20+length:]
    packed = struct.pack('<4sII', b'glTF', 2, 20+len(chunk)+len(tail)) + struct.pack('<II', len(chunk), kind) + chunk + tail
    scene = trimesh.load(io.BytesIO(packed), file_type='glb', force='scene', process=False,
                         merge_primitives=True)
    return scene, counts


def _parse_raw_mesh_standard(mesh_path, record, max_parts):
    masks = canonical_masks(record['masks'])
    scene, counts = tagged_glb(mesh_path)
    expected = {integer_key(k): int(v) for k, v in object_json(record['mesh_face_num']).items()}
    if counts != expected:
        raise ValueError(f'GLB mesh face counts differ from annotation: glb={counts}, annotation={expected}')
    mapping, report = partition_masks(masks, record.get('hierarchy'), max_parts)
    per_mesh = local_labels(masks, counts, mapping)
    pieces, labels, blocks = [], [], []
    seen, offset = set(), 0
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name].copy()
        mesh_id = str(mesh.metadata['segpart_mesh_index'])
        if len(mesh.faces) != counts[mesh_id]:
            raise ValueError('Loaded primitive concatenation changed face counts')
        mesh.apply_transform(transform)
        pieces.append(mesh)
        labels.append(per_mesh[mesh_id])
        blocks.append({'mesh_id': int(mesh_id), 'node': node, 'face_offset': offset, 'face_count': len(mesh.faces)})
        offset += len(mesh.faces)
        seen.add(mesh_id)
    if not pieces or any((per_mesh[k] > 0).any() for k in counts.keys() - seen):
        raise ValueError('Active GLB scene omits annotated meshes')
    combined = trimesh.util.concatenate(pieces)
    label = np.concatenate(labels)
    if len(label) != len(combined.faces):
        raise ValueError('Global label length differs from mesh')
    report.update(policy=ANNOTATION_POLICY, mesh_blocks=blocks, faces=len(label),
                  annotated_faces=int((label > 0).sum()), unannotated_faces=int((label == 0).sum()),
                  labeled_area_fraction=float(combined.area_faces[label > 0].sum() / combined.area_faces.sum()))
    return combined, label, report


def _primitive_scene(mesh_path):
    """Expose original triangle primitives and active scene DFS order explicitly.

    No non-triangle primitive is triangulated or given synthetic faces. Metadata
    face counts must prove that omitted line/point geometry was not annotated.
    """
    import copy
    blob = Path(mesh_path).read_bytes()
    magic,version,total = struct.unpack_from('<4sII',blob)
    size,kind = struct.unpack_from('<II',blob,12)
    if magic!=b'glTF' or version!=2 or total!=len(blob) or kind!=0x4E4F534A:
        raise ValueError('Invalid GLB header')
    source = json.loads(blob[20:20+size])
    header = copy.deepcopy(source)
    header['meshes'] = []
    lookup, omitted = {}, []
    for mi,mesh in enumerate(source.get('meshes',[])):
        lookup[mi] = []
        for pi,primitive in enumerate(mesh.get('primitives',[])):
            mode = primitive.get('mode',4)
            if mode!=4:
                omitted.append({'mesh_index':mi,'primitive_index':pi,'mode':mode})
                continue
            accessor=primitive.get('indices',primitive['attributes']['POSITION'])
            count=source['accessors'][accessor]['count']
            if count%3:
                raise ValueError('Triangle primitive has a non-triangular index count')
            new=copy.deepcopy(mesh)
            new['primitives']=[copy.deepcopy(primitive)]
            new['name']=f'segpart_primitive_{mi}_{pi}'
            new['extras']=dict(new.get('extras') or {},segpart_original_mesh=mi,segpart_original_primitive=pi)
            index=len(header['meshes']);header['meshes'].append(new)
            lookup[mi].append((pi,index,count//3))
    if not header['meshes']:
        raise ValueError('No triangle surface in GLB')
    # Preserve the original transform hierarchy; each primitive gets a child node.
    for ni,node in enumerate(source.get('nodes',[])):
        header['nodes'][ni]['name']=f'segpart_node_{ni:08d}'
        if 'mesh' not in node:
            continue
        del header['nodes'][ni]['mesh']
        header['nodes'][ni].pop('skin',None)
        for pi,mi,count in lookup[node['mesh']]:
            child_index=len(header['nodes'])
            header['nodes'].append({'name':f'segpart_surface_{ni}_{pi}','mesh':mi})
            header['nodes'][ni].setdefault('children',[]).append(child_index)
    payload=json.dumps(header,separators=(',',':')).encode()
    payload+=b' '*(-len(payload)%4)
    tail=blob[20+size:]
    packed=struct.pack('<4sII',b'glTF',2,20+len(payload)+len(tail))+struct.pack('<II',len(payload),kind)+payload+tail
    scene=trimesh.load(io.BytesIO(packed),file_type='glb',force='scene',process=False)
    order=[]
    def walk(ni,ancestry):
        if ni in ancestry:
            raise ValueError('Cyclic GLB scene')
        node=source['nodes'][ni]
        if 'mesh' in node:
            for pi,_,count in lookup[node['mesh']]:
                order.append((ni,node['mesh'],pi,count))
        for child in node.get('children',[]):
            walk(child,ancestry|{ni})
    for ni in source['scenes'][source.get('scene',0)].get('nodes',[]):
        walk(ni,set())
    return scene,order,omitted


def _parse_raw_mesh_compat(mesh_path,record,max_parts):
    masks=canonical_masks(record['masks'])
    expected=object_json(record.get('mesh_face_num'))
    expected={integer_key(k):int(v) for k,v in expected.items()} if expected is not None else None
    scene,dfs,omitted=_primitive_scene(mesh_path)
    if len({(x[0],x[2]) for x in dfs})!=len(dfs):
        raise ValueError('Instanced scene traversal is ambiguous for annotation IDs')
    # Each candidate preserves either GLB file order or active-scene DFS order;
    # never arbitrary sorting by size. Count vectors must agree exactly.
    candidates=[]
    # Some annotations retain GLB mesh indices, including gaps caused by points
    # or lines that have no surface annotations. Preserve the IDs if independently
    # verified counts match; do not relabel them to a compact range.
    mesh_groups={}
    for item in sorted(dfs,key=lambda x:(x[1],x[0],x[2])):
        mesh_groups.setdefault(str(item[1]),[]).append(item)
    original_counts={k:sum(x[3] for x in group) for k,group in mesh_groups.items()}
    sparse_original=(expected is not None and original_counts==expected)
    if sparse_original:
        group_ids=list(mesh_groups)
        candidates.append(('original_mesh_ids',[tuple(mesh_groups[k]) for k in group_ids],original_counts))
    for ordering,items in [('scene_dfs',dfs),('file',sorted(dfs,key=lambda x:(x[1],x[0],x[2])))]:
        primitives=[(item,) for item in items]
        groups=[]
        for item in items:
            if groups and groups[-1][0][0]==item[0]:
                groups[-1]=groups[-1]+(item,)
            else:
                groups.append((item,))
        for unit,grouped in [('primitive',primitives),('mesh',groups)]:
            counts={str(i):sum(item[3] for item in group) for i,group in enumerate(grouped)}
            if expected is not None and counts==expected:
                candidates.append((ordering+'_'+unit,grouped,counts))
    if expected is None:
        # Without mesh_face_num, only a single referenced surface is unambiguous.
        referenced={mesh_id for m in masks.values() for mesh_id,ids in m.items() if len(ids)}
        if len(dfs)==1 and referenced=={'0'}:
            candidates=[('single_surface_missing_face_count',[(dfs[0],)],{'0':dfs[0][3]})]
    if not candidates:
        raise ValueError(f'No count-verified triangle mapping; annotated geometry may be non-triangular or missing '
                         f'(annotation_meshes={len(expected) if expected else None}, triangle_primitives={len(dfs)}, '
                         f'nontriangle_primitives={len(omitted)})')
    # Prefer scene traversal only if the complete counts agree; file/DFS ambiguity
    # with different associations is rejected rather than guessed.
    associations={tuple(tuple((x[0],x[2]) for x in group) for group in grouped) for _,grouped,_ in candidates}
    if len(associations)>1:
        raise ValueError('Multiple count-verified scene/file mappings disagree; annotation association is ambiguous')
    method,groups,counts=candidates[0]
    masks,dense=decode_dense_masks(masks,counts)
    mapping,report=partition_masks(masks,record.get('hierarchy'),max_parts)
    per_mesh=local_labels(masks,counts,mapping)
    pieces,labels,blocks=[],[],[]
    offset=0
    for annotation_id,group in enumerate(groups):
        label_key=list(counts)[annotation_id]
        local_offset=0
        for ni,mi,pi,count in group:
            name=f'segpart_surface_{ni}_{pi}'
            transform,geometry_name=scene.graph[name]
            part=scene.geometry[geometry_name].copy()
            if len(part.faces)!=count:
                raise ValueError('Primitive loader changed face order/count')
            part.apply_transform(transform)
            pieces.append(part)
            labels.append(per_mesh[label_key][local_offset:local_offset+count])
            blocks.append({'mesh_id':int(label_key),'original_mesh_index':mi,'primitive_index':pi,
                           'node':ni,'face_offset':offset,'face_count':count})
            local_offset+=count;offset+=count
    combined=trimesh.util.concatenate(pieces)
    label=np.concatenate(labels)
    report.update(policy=ANNOTATION_POLICY,mesh_blocks=blocks,faces=len(label),
                  annotated_faces=int((label>0).sum()),unannotated_faces=int((label==0).sum()),
                  labeled_area_fraction=float(combined.area_faces[label>0].sum()/combined.area_faces.sum()),
                  compatibility_mapping=method,omitted_nontriangle_primitives=omitted)
    if dense:
        report['raw_mask_encoding']='dense_binary_face_vector'
    return combined,label,report


def parse_raw_mesh(mesh_path,record,max_parts):
    try:
        return _parse_raw_mesh_standard(mesh_path,record,max_parts)
    except (ValueError,AttributeError) as original:
        try:
            return _parse_raw_mesh_compat(mesh_path,record,max_parts)
        except (ValueError,AttributeError) as fallback:
            raise ValueError(f'{fallback}; initial parser: {original}') from fallback
