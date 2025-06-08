from pygltflib import GLTF2
import pyglet
import numpy as np
from PIL import Image
from OpenGL.GL import *
import io
import pyrr
from pyrr import Matrix44, Vector3

from Model import Node, Mesh

class GLBLoader:
    def __init__(self, glb_path):
        self.gltf = GLTF2().load(glb_path)
        self.buffer_data = self.gltf.binary_blob()
        
        self.positions = None
        self.normals = None
        self.texcoords = None
        self.indices = None
        self.texture_id = None
        self.joints = None           # JOINTS_0 (정수형)
        self.weights = None          # WEIGHTS_0 (float형)
        
        self.nodes = {}
        self._load_geometry()
        self.mesh = Mesh(
            positions = self.positions,
            normals=self.normals,
            texcoords=self.texcoords,
            joints=self.joints,
            weights=self.weights,
            indices=self.indices,
            texture_id = self.texture_id
        )
        
        self._load_skinning()
        
    def _read_accessor_data(self, accessor_idx):
        accessor = self.gltf.accessors[accessor_idx]
        view = self.gltf.bufferViews[accessor.bufferView]
        dtype_map = {
            5120: np.int8, 5121: np.uint8,
            5122: np.int16, 5123: np.uint16,
            5125: np.uint32, 5126: np.float32,
        }
        type_count = {
            "SCALAR": 1, "VEC2": 2, "VEC3": 3,
            "VEC4": 4, "MAT4": 16
        }

        dtype = dtype_map[accessor.componentType]
        count = accessor.count
        components = type_count[accessor.type]

        byte_offset = (view.byteOffset or 0) + (accessor.byteOffset or 0)
        byte_length = count * components * np.dtype(dtype).itemsize

        data = np.frombuffer(self.buffer_data[byte_offset: byte_offset + byte_length], dtype=dtype)
        return data.reshape((count, components))

    def _upload_texture_from_image(self, pil_img):
        pil_img = pil_img.convert("RGBA")
        width, height = pil_img.size
        img_data = np.array(pil_img).flatten()

        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

        return tex_id

    def _load_geometry(self):
        mesh = self.gltf.meshes[0]
        primitive = mesh.primitives[0]

        self.positions = self._read_accessor_data(primitive.attributes.POSITION)
        self.normals = self._read_accessor_data(primitive.attributes.NORMAL)
        self.texcoords = self._read_accessor_data(primitive.attributes.TEXCOORD_0)
        self.indices = self._read_accessor_data(primitive.indices).flatten()

        # 버텍스 웨이트
        if primitive.attributes.JOINTS_0 is not None:
            self.joints = self._read_accessor_data(primitive.attributes.JOINTS_0)
        if primitive.attributes.WEIGHTS_0 is not None:
            self.weights = self._read_accessor_data(primitive.attributes.WEIGHTS_0)

        # 텍스처
        material_idx = primitive.material
        
        if material_idx is not None:
            mat = self.gltf.materials[material_idx]
            if mat.pbrMetallicRoughness and mat.pbrMetallicRoughness.baseColorTexture:
                texture_idx = mat.pbrMetallicRoughness.baseColorTexture.index
                image_idx = self.gltf.textures[texture_idx].source
                img = self.gltf.images[image_idx]

                if img.uri:
                    pil_img = Image.open(img.uri)
                    self.texture_id = self._upload_texture_from_image(pil_img)
                elif img.bufferView is not None:
                    view = self.gltf.bufferViews[img.bufferView]
                    start = view.byteOffset or 0
                    length = view.byteLength
                    tex_bytes = self.buffer_data[start:start+length]
                    pil_img = Image.open(io.BytesIO(tex_bytes))
                    self.texture_id = self._upload_texture_from_image(pil_img)
                    
    def _load_skinning(self):
        
        # 조인트 - 노드 인덱스, 인버스 바인드 연결
        inv_bind_mats = self._read_accessor_data(self.gltf.skins[0].inverseBindMatrices).reshape(-1, 4, 4)
        joint_nodes = self.gltf.skins[0].joints
        joint_map = {node_idx: (joint_idx, inv_bind) 
            for joint_idx, (node_idx, inv_bind) in enumerate(zip(joint_nodes, inv_bind_mats))}

        for idx, node in enumerate(self.gltf.nodes):
            new_node = Node()
            
            new_node.name = node.name
            
            if hasattr(node, "children") and node.children:
                new_node.children_indices = node.children
                
            if idx in joint_map:
                new_node.joint_index = joint_map[idx][0]
                new_node.inverse_bind_matrix = joint_map[idx][1].T
            else:
                new_node.inverse_bind_matrix = np.eye(4, dtype=np.float32)
                
            self.nodes[idx] = new_node
            
        for idx, node in self.nodes.items():
            for child_idx in node.children_indices:
                self.nodes[child_idx].parent_index = idx
                
        def recurse(node, parent_bind_inv):
            # ParentWorld @ ChildLocal = ChildWorld
            node.local_bind_matrix = parent_bind_inv @ np.linalg.inv(node.inverse_bind_matrix)
            
            for child in node.children_indices:
                recurse(self.nodes[child], node.inverse_bind_matrix)
              
        root_nodes = [node for node in self.nodes.values() if node.parent_index == -1]

        for root_node in root_nodes:
            recurse(root_node, np.eye(4, dtype=np.float32))