from pygltflib import GLTF2
import pyglet
import numpy as np
from PIL import Image
from OpenGL.GL import *
import pyrr
from pyrr import Matrix44, Vector3
import time
import pyglet

from PoseEstimator import PoseEstimator
from Model import Model, Node, Mesh
from GLBLoader import GLBLoader
from Shader import Shader

class Process():
    def __init__(self):
        print("oo")
        self.window = pyglet.window.Window(800, 600, "GLB Viewer", resizable=False)
        self.bQuit = False
        
        self.pose_estimator = PoseEstimator("sample.mp4")
        self.frame_duration = 1 / self.pose_estimator.videoFPS
        
        self.data = GLBLoader("pig.glb")
        self.model = Model(self.data.mesh, self.data.nodes)
        self.shader = Shader().shader
        
        self.view_mat = Matrix44.look_at(
            eye=Vector3([1.0, 2.0, 5.0]),
            target=Vector3([1.0, 2.0, 0.0]),
            up=Vector3([0.0, -1.0, 0.0])
        ).astype(np.float32)
        self.proj_mat = Matrix44.perspective_projection(
            fovy=45.0,
            aspect=800/600,
            near=0.01,
            far=10000.0
        ).astype(np.float32)
        self.model_mat = Matrix44.identity(dtype=np.float32)
        
        glEnable(GL_DEPTH_TEST)
        
        self.timer = 0
        self.curTime = time.time()
        
        
    def run(self):
        while self.bQuit is False:
            prevTime = self.curTime
            self.curTime = time.time()
            self.timer += self.curTime - prevTime
            if(self.timer < self.frame_duration): continue
            self.timer -= self.frame_duration
            
            self.window.switch_to()      # 컨텍스트 활성화
            self.window.dispatch_events()  # 이벤트 처리
            
            self.render()
            
            self.window.flip()
            

        
        
    def render(self):
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self.shader)

        # 카메라
        glUniformMatrix4fv(glGetUniformLocation(self.shader, b"model"), 1, GL_FALSE, self.model_mat)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, b"view"), 1, GL_FALSE, self.view_mat)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, b"projection"), 1, GL_FALSE, self.proj_mat)

        # 본행렬 계산 및 업로드
        self.bQuit = self.pose_estimator.update() 
        self.model.update(self.shader, self.pose_estimator.joints)
        
        


        
