from pygltflib import GLTF2
import pyglet
import numpy as np
from PIL import Image
from OpenGL.GL import *
import pyrr
from pyrr import Matrix44, Vector3

class Shader():
    def __init__(self, vs = "shader.vert", fs = "shader.frag"):
        vs = open(vs,'r', encoding="utf-8").read()
        fs = open(fs,'r', encoding="utf-8").read()
        self.shader = self.create_shader_program(vs, fs)
       
    def compile_shader(self, source, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(shader).decode())
        return shader
    
    def create_shader_program(self, vertex_src, fragment_src):
        vs = self.compile_shader(vertex_src, GL_VERTEX_SHADER)
        fs = self.compile_shader(fragment_src, GL_FRAGMENT_SHADER)
        program = glCreateProgram()
        glAttachShader(program, vs)
        glAttachShader(program, fs)
        glLinkProgram(program)
        if not glGetProgramiv(program, GL_LINK_STATUS):
            raise RuntimeError(glGetProgramInfoLog(program).decode())
        glDeleteShader(vs)
        glDeleteShader(fs)
        return program  
