#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in uvec4 aJoints;
layout(location = 4) in vec4 aWeights;

out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 boneMatrices[100]; // 충분히 큰 수로 설정

void main()
{
    vec4 normWeights = normalize(aWeights);
    mat4 skinMatrix = 
        normWeights.x * boneMatrices[aJoints.x] +
        normWeights.y * boneMatrices[aJoints.y] +
        normWeights.z * boneMatrices[aJoints.z] +
        normWeights.w * boneMatrices[aJoints.w];
        
    vec4 skinnedPos = skinMatrix * vec4(aPos, 1.0);
    gl_Position = projection * view * model * skinnedPos;

    TexCoord = aTexCoord;
}