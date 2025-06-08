from OpenGL.GL import *
import numpy as np
from dataclasses import dataclass, field
import pyrr
from pyrr import Matrix44, Vector3
from pygltflib import GLTF2
import pyglet
from typing import Optional, List

@dataclass
class Node:
    name: str = ""                                             # 노드 이름
    parent_index: int = -1
    children_indices: List[int] = field(default_factory=list)  # 자식 노드 인덱스들
    joint_index: Optional[int] = None                          # skin.joints[] 내부에서의 인덱스
    inverse_bind_matrix: Optional[np.ndarray] = None           # 인버스 바인딩 매트릭스
    local_bind_matrix: Optional[np.ndarray] = None             # 우리가 계산한 로컬 바인딩 매트릭스
    local_matrix: Optional[np.ndarray] = None   
    world_matrix: Optional[np.ndarray] = None 

@dataclass     
class Mesh:
    positions: np.ndarray 
    normals: np.ndarray 
    texcoords: np.ndarray 
    joints: np.ndarray 
    weights: np.ndarray 
    indices: np.ndarray 
    texture_id: Optional[int] = None
    
def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v  # 또는 raise ValueError("Zero-length vector")
    return v / norm 

class Model:
    def __init__(self, mesh, nodes):
        self.mesh = mesh
        self.nodes = nodes
        self.jointCount = 0
        for node in self.nodes.values():
            if node.joint_index is not None:
                self.jointCount += 1
        
        self.VAO = GLuint()   
        glGenVertexArrays(1, ctypes.byref(self.VAO))
        glBindVertexArray(self.VAO)

        vertex_data = np.hstack([
            mesh.positions.astype(np.float32),       
            mesh.normals.astype(np.float32),
            mesh.texcoords.astype(np.float32),
            mesh.joints.astype(np.uint32).view(np.float32),  # float array로 정렬만 맞춤
            mesh.weights.astype(np.float32)
        ])

        stride = 64  # 16 floats
        VBO = GLuint()
        glGenBuffers(1, ctypes.byref(VBO))
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

        EBO = GLuint()
        glGenBuffers(1, ctypes.byref(EBO))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.nbytes, mesh.indices, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)

        glVertexAttribIPointer(3, 4, GL_UNSIGNED_INT, stride, ctypes.c_void_p(32))
        glEnableVertexAttribArray(3)

        glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(48))
        glEnableVertexAttribArray(4)
        
        
    def update(self, shader_program, joint_dir):
        glBindVertexArray(self.VAO)
        
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.mesh.texture_id)
        loc = glGetUniformLocation(shader_program, b"tex")
        glUniform1i(loc, 0)
        
        self.compute_world_transform(joint_dir)
        self.upload_skin_matrices(shader_program)
       
        glDrawElements(GL_TRIANGLES, len(self.mesh.indices), GL_UNSIGNED_SHORT, None)
        
        
        
        
    def compute_world_transform(self, joint_dir):
        for node in self.nodes.values():
            node.local_matrix = np.eye(4)
            
        def recurse(node, parent_matrix):
            if node.name in joint_dir:
                target_vec = joint_dir[node.name]['end']- joint_dir[node.name]['start']
                target_vec[2] = -target_vec[2]* 0.5
                target_vec = normalize(target_vec)
                base_matrix = parent_matrix @ node.local_bind_matrix
                base_rot = base_matrix[:3, :3]
                R = np.eye(4)
                
                if node.name == "mixamorig:Hips":
                    local_x = normalize(base_rot.T @ target_vec)
                    fwd_hint = np.array([0, 0, 1])
                    if abs(np.dot(local_x, fwd_hint)) > 0.99:
                        fwd_hint = np.array([0, 1, 0])
                    local_y = normalize(np.cross(local_x, fwd_hint))
                    local_z = normalize(np.cross(local_y, local_x))
                    
                    mid_pos = (joint_dir[node.name]['start'] + joint_dir[node.name]['end']) / 2
                    mid_pos[2] = 0
                    R[:3, 3] = mid_pos * 200
                else:
                    # 나머지는 local_y 기준
                    local_y = normalize(base_rot.T @ target_vec)
                    fwd_hint = np.array([0, 0, 1])
                    if abs(np.dot(local_y, fwd_hint)) > 0.99:
                        fwd_hint = np.array([1, 0, 0])
                    local_x = normalize(np.cross(local_y, fwd_hint))
                    local_z = normalize(np.cross(local_x,local_y))
                
                R[:3, 0] = local_x
                R[:3, 1] = local_y
                R[:3, 2] = local_z
                node.local_matrix = R
                
                
            # ParentWorld @ ChildLocal = ChildWorld
            node.world_matrix = parent_matrix @ (node.local_bind_matrix @ node.local_matrix)
            for child in node.children_indices:
                recurse(self.nodes[child], node.world_matrix)
                #recurse(self.nodes[child], parent_matrix @ node.local_bind_matrix)


        # 부모가 없는 루트 노드들 찾기
        root_nodes = [node for node in self.nodes.values() if node.parent_index  == -1]
        for root_node in root_nodes:
            recurse(root_node, np.eye(4, dtype=np.float32))


    def upload_skin_matrices(self, shader_program):
        skin_matrices = [None] * self.jointCount
        for node in self.nodes.values():
            if node.joint_index is None: continue
            final =  node.world_matrix @ node.inverse_bind_matrix
            skin_matrices[node.joint_index] = final.T

        loc = glGetUniformLocation(shader_program, b"boneMatrices")
        flat = np.array(skin_matrices, dtype=np.float32).reshape(-1) 
        glUniformMatrix4fv(loc, len(skin_matrices), GL_FALSE, flat)
        