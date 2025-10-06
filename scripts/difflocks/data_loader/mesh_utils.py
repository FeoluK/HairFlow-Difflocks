import trimesh
import numpy as np
import igl
from trimesh.triangles import points_to_barycentric
import torch
from typing import Union, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def closest_point_barycentrics(query_points, mesh_verts, mesh_faces):
    """Given a 3D mesh and a set of query points, return closest point barycentrics
    Args:
        query_points: np.array (float)
        [Q, 3] query points
        mesh_verts: np.array (float)
        [N, 3] mesh vertices
        mesh_faces: np.array (int)
        [N, 3] mesh triangle indices
    Returns:
        Tuple[closest_points, barys, vertex_idxs, face_idxs]
            closest_points:       [Q, 3] approximated (closest) points on the mesh
            barys:        [Q, 3] barycentric weights that produce "approx"
            vertex_idxs:  [Q, 3] vertex indices for barycentric interpolation
            face_idxs:    [Q] face indices for barycentric interpolation. vertex_idxs = mesh_faces[face_idxs]
    """
    sqr_distances, face_idxs, closest_points = igl.point_mesh_squared_distance(query_points, mesh_verts, mesh_faces)  

    # if filtering:
    #     valid_q_idx = np.where(np.sqrt(sqr_distances) < filter_dis_thres)[0]
    #     p = p[valid_q_idx]
    #     face_idxs = face_idxs[valid_q_idx]
    # else:
    #     valid_q_idx = np.arange(p.shape[0])


    vertex_idxs = mesh_faces[face_idxs]
    face_v0 = mesh_verts[vertex_idxs[:, 0]]
    face_v1 = mesh_verts[vertex_idxs[:, 1]]
    face_v2 = mesh_verts[vertex_idxs[:, 2]]

    barys = igl.barycentric_coordinates_tri(closest_points, face_v0, face_v1, face_v2)

    # #sanity check
    # b0, b1, b2 = np.split(barys, 3, axis=1)
    # approx = b0 * face_v0 + b1 * face_v1 + b2 * face_v2
    # diff = closest_points-approx
    # print("max diff",np.max(diff))

    return closest_points, barys, vertex_idxs, face_idxs


def compute_vertex_tbn(vertex_pos, vertex_uv, faces, eps=1e-5):
    """Compute tangents, bitangents, normals.
    Args:
        vertex_pos: [N,3] vertex coordinates
        vertex_uv: [N,2] texture coordinates
        faces: [M,3] texture coordinates
    Returns:
        tangents, bitangents, normals
    """


    #sample the positions and uv for every face vertex 
    #as a results the triangled vertices are (B,N,3,3) and triangled_vertices_uv is (N,3,2)
    triangled_vertices = torch.tensor(vertex_pos[faces, :])
    triangled_vertices_uv = torch.tensor(vertex_uv[faces, :])

    v01 = triangled_vertices[:, 1] - triangled_vertices[:, 0]
    v02 = triangled_vertices[:, 2] - triangled_vertices[:, 0]


    normals = torch.cross(v01, v02, dim=-1)
    normals = normals / torch.norm(normals, dim=-1, keepdim=True).clamp(min=eps)

    vt01 = triangled_vertices_uv[:, 1] - triangled_vertices_uv[:, 0]
    vt02 = triangled_vertices_uv[:, 2] - triangled_vertices_uv[:, 0]

    f = 1.0 / (vt01[..., 0] * vt02[..., 1] - vt01[..., 1] * vt02[..., 0])

    tangents = f[..., np.newaxis] * (
        v01 * vt02[..., 1][..., np.newaxis] - v02 * vt01[..., 1][..., np.newaxis])
    tangents = tangents / torch.norm(tangents, dim=-1, keepdim=True).clamp(min=eps)


    bitangents = torch.cross(normals, tangents, dim=-1)
    bitangents = bitangents / torch.norm(bitangents, dim=-1, keepdim=True).clamp(min=eps)


    #splat the tangent bitangent and normals from faces onto the vertices
    v_t = torch.zeros(vertex_pos.shape[0],3)
    v_b = torch.zeros(vertex_pos.shape[0],3)
    v_n = torch.zeros(vertex_pos.shape[0],3)
    for i in range(vertex_pos.shape[0]):
        index = np.where(faces==i)[0]
        v_t[i,:]=torch.mean(tangents[index],axis=0)
        v_b[i,:]=torch.mean(bitangents[index],axis=0)
        v_n[i,:]=torch.mean(normals[index],axis=0)

    # for f_t, f_b, f_n, f in zip(tangents,bitangents,normals, faces):
    #     for vertex_idx in f:
    #         v_t[vertex_idx,:]+=f_t
    #         v_b[vertex_idx,:]+=f_b
    #         v_n[vertex_idx,:]+=f_n
        
  

    v_t=torch.nn.functional.normalize(v_t)
    v_b=torch.nn.functional.normalize(v_b)
    v_n=torch.nn.functional.normalize(v_n)

    return v_t, v_b, v_n

def interpolate_tbn(barys, vertex_idxs, v_tangents, v_bitangents, v_normals):
    nr_positions=barys.shape[0]

    sampled_tangents = v_tangents[vertex_idxs.reshape(-1),:].reshape(nr_positions,3,3) #nr_positions x vertices_on_face(3) x 3
    weighted_tangents = sampled_tangents*barys.reshape(nr_positions,3,1)
    point_tangents = weighted_tangents.sum(axis=1)
    point_tangents= point_tangents/np.linalg.norm(point_tangents,axis=-1, keepdims=True)


    sampled_normals = v_normals[vertex_idxs.reshape(-1),:].reshape(nr_positions,3,3)
    weighted_normals = sampled_normals*barys.reshape(nr_positions,3,1)
    point_normals = weighted_normals.sum(axis=1)
    point_normals= point_normals/np.linalg.norm(point_normals,axis=-1, keepdims=True)


    point_bitangents=np.cross(point_normals,point_tangents)
    point_bitangents= point_bitangents/np.linalg.norm(point_bitangents,axis=-1, keepdims=True)

    #make sure the tangent is also orthogonal
    point_tangents=np.cross(point_bitangents,point_normals)
    point_tangents= point_tangents/np.linalg.norm(point_tangents,axis=-1, keepdims=True)

    return point_tangents, point_bitangents, point_normals


#strands_tbn is [B, Nr_strands, 3, 3] where each column of the 3x3 matrix is T,B,N
#strands_positions is [B, Nr_strands, nr_points_per_strand, 3]
#strands_normals is [B, Nr_strands, 3]
# @torch.compile
def world_to_tbn_space(strands_tbn, strands_positions, root_normals):
    # print("strands_tbn",strands_tbn.shape)
    nr_batch= strands_tbn.shape[0]
    nr_strands= strands_tbn.shape[1]
    root_pos = strands_positions[:,:,0:1,:] #[B, Nr_strands, 1, 3]

    #transform from scalp to world

    #remove the root for the positional data 
    strands_positions = strands_positions-root_pos

    #we want to map tangent to X, bitangent to Z and normal to Y, so we swap B and N
    indices_tbn=torch.tensor([0,2,1], device="cuda").long()
    strands_tbn=torch.index_select(strands_tbn, 3, indices_tbn)
    #make the Tangent to be along +x
    strands_tbn[..., 0] = -strands_tbn[..., 0]

    #rotate so that the TBN is identity
    #TBN is basically the rotation from scalp to world, we want the inverse
    strands_tbn_inv = strands_tbn.transpose(2,3)

    #rotate positional data  [B, Nr_strands, 3, 3] x [B, Nr_strands, nr_points_per_strand, 3]
    strands_tbn_inv = strands_tbn_inv.reshape(nr_batch, nr_strands, 1, 3, 3)
    # print("strands_positions",strands_positions.shape)
    strands_positions = strands_positions.reshape(nr_batch, nr_strands, -1, 3, 1)
    strands_positions= torch.matmul(strands_tbn_inv, strands_positions)
    strands_positions = strands_positions.reshape(nr_batch, nr_strands, -1, 3)

    #roundtrip
    # strands_tbn = strands_tbn.reshape(nr_batch, nr_strands, 1, 3, 3)
    # strands_positions = strands_positions.reshape(nr_batch, nr_strands, -1, 3, 1)
    # strands_positions= torch.matmul(strands_tbn, strands_positions)
    # strands_positions = strands_positions.reshape(nr_batch, nr_strands, -1, 3)
    # strands_positions = strands_positions+root_pos

    #rotate normals
    root_normals = root_normals.reshape(nr_batch, nr_strands, 3, 1)
    strands_tbn_inv = strands_tbn_inv.reshape(nr_batch, nr_strands, 3, 3)
    root_normals= torch.matmul(strands_tbn_inv, root_normals)
    root_normals = root_normals.reshape(nr_batch, nr_strands, 3)

    return strands_positions, root_normals

#incurs a copy to cpu
#generates uv space map where each pixel has the index of the triangle, the 3 vertex indices of the triangle and the barycentric weights
def compute_uv_space_data(
    vt: torch.Tensor,
    vti: torch.Tensor,
    uv_shape: Union[Tuple[int, int], int],
    flip_uv: bool = True,
):
    """Compute a UV-space barycentric map where each texel contains barycentric
    coordinates for the closest point on a UV triangle.
    Args:
        vt: torch.Tensor
        Texture coordinates. Shape = [n_texcoords, 2]
        vti: torch.Tensor
        Face texture coordinate indices. Shape = [n_faces, 3]
        uv_shape: Tuple[int, int] or int
        Shape of the texture map. (HxW)
        flip_uv: bool
        Whether or not to flip UV coordinates along the V axis (OpenGL -> numpy/pytorch convention).
    Returns:
        torch.Tensor: index_img: Face index image, shape [uv_shape[0], uv_shape[1]]
        torch.Tensor: Barycentric coordinate map, shape [uv_shape[0], uv_shape[1], 3]Â
    """

    if isinstance(uv_shape, int):
        uv_shape = (uv_shape, uv_shape)

    if flip_uv:
        # Flip here because texture coordinates in some of our topo files are
        # stored in OpenGL convention with Y=0 on the bottom of the texture
        # unlike numpy/torch arrays/tensors.
        vt = vt.clone()
        vt[:, 1] = 1 - vt[:, 1]

    # Texel to UV mapping (as per OpenGL linear filtering)
    # https://www.khronos.org/registry/OpenGL/specs/gl/glspec46.core.pdf
    # Sect. 8.14, page 261
    # uv=(0.5,0.5)/w is at the center of texel [0,0]
    # uv=(w-0.5, w-0.5)/w is the center of texel [w-1,w-1]
    # texel = floor(u*w - 0.5)
    # u = (texel+0.5)/w

    uv_grid = torch.meshgrid(
        torch.linspace(0.5, uv_shape[0] - 1 + 0.5, uv_shape[0]) / uv_shape[0],
        torch.linspace(0.5, uv_shape[1] - 1 + 0.5, uv_shape[1]) / uv_shape[1],
        indexing="ij")  # HxW, v,u
    uv_grid = torch.stack(uv_grid[::-1], dim=2)  # HxW, u, v
    uv = uv_grid.reshape(-1, 2).data.to("cpu").numpy()
    vth = np.hstack((vt, vt[:, 0:1] * 0 + 1))
    uvh = np.hstack((uv, uv[:, 0:1] * 0 + 1))

    # print("vth",vth)

    closest_points, barys, vertex_idxs, face_idxs = closest_point_barycentrics(uvh, vth, vti.numpy())

    index_img = torch.from_numpy(face_idxs.reshape(uv_shape[0], uv_shape[1])).long()
    vert_idxs_img = torch.from_numpy(vertex_idxs.reshape(uv_shape[0], uv_shape[1],3)).long()
    bary_img = torch.from_numpy(barys.reshape(uv_shape[0], uv_shape[1], 3)).float()
    return index_img, vert_idxs_img, bary_img


#strands_tbn is [Nr_strands, 3, 3] where each column of the 3x3 matrix is T,B,N
# strands_positions is [Nr_strands, nr_points_per_strand, 3]
# root_uv is [Nr_strand,2]
# face_idxs is [nr_strands]
# def tbn_space_to_local(self, strands_tbn, strands_positions, root_uv, scalp_index_map, scalp_bary_map):
#         # binary_uv_map = binary_uv_map.detach().cpu().numpy()[..., 0]
#         # # binary_uv_map[binary_uv_map < 0.01] = 0
#         # binary_uv_map[binary_uv_map < 0.5] = 0
#         # binary_uv_map[binary_uv_map >= 0.5] = 1
#         # if np.sum(binary_uv_map) < 1:
#         #     return
#         # index = np.nonzero(binary_uv_map)
#         # index = np.concatenate([index[0][..., None], index[1][..., None]], 1)

#         # face_idxs = self.scalp_index_map[index[:, 0], index[:, 1]]
#         # barys = self.scalp_bary_map[index[:, 0], index[:, 1], :]

#         # strands, tbn_strands = self.world_strands_from_scalptexture(uv_map, binary_uv_map, face_idxs, barys)
#         # strands = strands.detach().cpu().numpy()
#         # return strands,tbn_strands
    
#     assert strands_tbn.dim() == 3
#     assert strands_positions.dim() == 3
    
#     #we want to map tangent to X, bitangent to Z and normal to Y, so we swap B and N
#     indices_tbn=torch.LongTensor([0,2,1], device="cuda")
#     strands_tbn=torch.index_select(strands_tbn, 2, indices_tbn)
#     #make the Tangent to be along +x
#     strands_tbn[:,:,:,0] = -strands_tbn[:,:,:,0]


#     # basis change
#     orig_points = torch.matmul(strands_tbn, strands_positions.permute(0, 2, 1)).permute(0, 2, 1)


#     # translate to world space with brad and triangle vertices
#     triangled_vertices = torch.tensor(self.head_mesh.vertices[self.head_mesh.faces, :])
#     roots_triangles = triangled_vertices[face_idxs]


#     roots_positions = roots_triangles[:, 0] * face_barys[:, 0:1] + \
#                         roots_triangles[:, 1] * face_barys[:, 1:2] + \
#                         roots_triangles[:, 2] * face_barys[:, 2:3]
#     strds_points = orig_points + roots_positions[:, None, :].to(device)
#     # density = torch.from_numpy(density).to(device).type(torch.float32)
#     # indexs = torch.gt(torch.minimum(density,torch.ones_like(density)),torch.rand_like(density))

#     return strds_points[indexs],pred_points[indexs]


def tbn_space_to_world(root_uv, strands_positions, scalp_mesh_data):

    scalp_index_map=scalp_mesh_data["index_map"]
    scalp_vertex_idxs_map=scalp_mesh_data["vertex_idxs_map"]
    scalp_bary_map=scalp_mesh_data["bary_map"]
    mesh_v_tangents=scalp_mesh_data["v_tangents"]
    mesh_v_bitangents=scalp_mesh_data["v_bitangents"]
    mesh_v_normals=scalp_mesh_data["v_normals"]
    scalp_v=scalp_mesh_data["verts"]
    scalp_f=scalp_mesh_data["faces"]
        # binary_uv_map = binary_uv_map.detach().cpu().numpy()[..., 0]
        # # binary_uv_map[binary_uv_map < 0.01] = 0
        # binary_uv_map[binary_uv_map < 0.5] = 0
        # binary_uv_map[binary_uv_map >= 0.5] = 1
        # if np.sum(binary_uv_map) < 1:
        #     return
        # index = np.nonzero(binary_uv_map)
        # index = np.concatenate([index[0][..., None], index[1][..., None]], 1)

        # face_idxs = self.scalp_index_map[index[:, 0], index[:, 1]]
        # barys = self.scalp_bary_map[index[:, 0], index[:, 1], :]

        # strands, tbn_strands = self.world_strands_from_scalptexture(uv_map, binary_uv_map, face_idxs, barys)
        # strands = strands.detach().cpu().numpy()
        # return strands,tbn_strands
    
    # assert strands_tbn.dim() == 3
    # assert strands_positions.dim() == 3

    #given the uv, we get pixel indices
    # print("scalp_vertex_idxs_map",scalp_vertex_idxs_map.shape)
    tex_size = scalp_vertex_idxs_map.shape[0]
    # print("tex size", tex_size)
    pixel_indices = (root_uv*tex_size).floor().int()
    # print("pixel_indices",pixel_indices.shape)
    # print("pixel_indices",pixel_indices)

    #sample from the vertex idxs map and bary map
    face_idxs = scalp_index_map[pixel_indices[:, 0], pixel_indices[:, 1]]
    vertex_idxs = scalp_vertex_idxs_map[pixel_indices[:, 0], pixel_indices[:, 1], :]
    barys = scalp_bary_map[pixel_indices[:, 0], pixel_indices[:, 1], :]

    # print("vertex_idxs",vertex_idxs.shape)
    # print("barys",barys.shape)
    # print("barys",barys)
    # print("vertex_idxs",vertex_idxs.shape)
    # print("vertex_idxs",vertex_idxs)
    # vertex_idxs= scalp_f[face_idxs]
    # print("vertex_idxs",vertex_idxs)

    #interpolate
    root_tangent, root_bitangent, root_normal = interpolate_tbn(barys, vertex_idxs, mesh_v_tangents, mesh_v_bitangents, mesh_v_normals) 
    strands_tbn = np.stack((root_tangent,root_bitangent,root_normal),axis=2) 
    strands_tbn = torch.from_numpy(strands_tbn).to(device)
    # print("strands_tbn",strands_tbn.shape)
    
    # #we want to map tangent to X, bitangent to Z and normal to Y, so we swap B and N
    indices_tbn=torch.tensor([0,2,1], device=device)
    strands_tbn=torch.index_select(strands_tbn, 2, indices_tbn)
    # #make the Tangent to be along +x
    strands_tbn[..., 0] = -strands_tbn[..., 0]


    # # basis change
    orig_points = torch.matmul(strands_tbn, strands_positions.permute(0, 2, 1)).permute(0, 2, 1)
    # print("orig_points",orig_points.shape)


    # # translate to world space with brad and triangle vertices
    # triangled_vertices = torch.tensor(scalp_v[scalp_f, :])
    # roots_triangles = triangled_vertices[face_idxs]
    # roots_positions = roots_triangles[:, 0] * barys[:, 0:1] + \
    #                     roots_triangles[:, 1] * barys[:, 1:2] + \
    #                     roots_triangles[:, 2] * barys[:, 2:3]


    #attempt 2 to get root
    nr_positions=vertex_idxs.shape[0]
    sampled_v = scalp_v[vertex_idxs.reshape(-1),:].reshape(nr_positions,3,3) #nr_positions x vertices_on_face(3) x 3
    # print("sampled_v",sampled_v.shape)
    sampled_v= torch.from_numpy(sampled_v)
    weighted_v = sampled_v*barys.reshape(nr_positions,3,1)
    roots_positions = weighted_v.sum(axis=1)
    # print("roots_positions",roots_positions.shape)

    # print("root in computing tbn to root", roots_positions)


    strds_points = orig_points + roots_positions[:, None, :].to(device)
    # # density = torch.from_numpy(density).to(device).type(torch.float32)
    # # indexs = torch.gt(torch.minimum(density,torch.ones_like(density)),torch.rand_like(density))

    # return strds_points[indexs],pred_points[indexs]

    return strds_points

#converts a trimesh object into vertices, UV and normals 
def mesh_to_data(mesh):
    mesh_uv_u = mesh.metadata["_ply_raw"]["vertex"]["data"]["s"]
    mesh_uv_v = mesh.metadata["_ply_raw"]["vertex"]["data"]["t"]
    mesh_uv = np.column_stack((mesh_uv_u, mesh_uv_v)).astype(np.float32)
    mesh_v = mesh.vertices.astype(np.float32)
    mesh_f = mesh.faces.astype(np.uint32)

    return mesh_v, mesh_f, mesh_uv


class World2Local(torch.nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, strands_tbn, strands_positions, root_normals):
        return world_to_tbn_space(strands_tbn, strands_positions, root_normals)





