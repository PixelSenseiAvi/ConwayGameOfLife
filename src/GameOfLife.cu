#include "Utility.h"

#include "GameOfLife.cuh"
#include <iostream>


bool GameOfLife::Init()
{
    GLint ret = true;  
    GLclampf Red = 0.0f, Green = 0.0f, Blue = 1.0f, Alpha = 0.0f;

    glClearColor(Red, Green, Blue, Alpha);
    glPointSize(m_pointSize);
    
    // // Init Buffer
    glGenBuffers(1, &m_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, m_widthX * m_widthY * sizeof(unsigned int), 0, GL_DYNAMIC_DRAW);

    // Attrib Pointer
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 1, GL_UNSIGNED_INT, GL_FALSE, 0, 0);

    // ShaderS
    m_shader = Shader("shaders/GameOfLife.shader");
    m_shader.Bind();
	m_shader.SetUniformUint("_uWidthX", m_widthX);
    m_shader.SetUniformUint("_uWidthY", m_widthY);
	m_shader.SetUniform4f("_uOnColor", 1., 1., 1., 1.);
	m_shader.SetUniform4f("_uOffColor", 0., 0., 0., 1.);
	m_shader.SetUniform4f("_uWindowXY", -1.0, 1.0, -1.0, 1.0);

    // CUDA graphics resource 
    cudaGraphicsGLRegisterBuffer(&m_resource, m_VBO, cudaGraphicsRegisterFlagsNone);

    unsigned int* m_DevState;
    m_BufferSize = m_widthX * m_widthY * sizeof(unsigned int);
	cudaGraphicsMapResources(1, &m_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&m_DevState, &m_BufferSize, m_resource);
	_random<<<m_blocks, m_threads>>>(m_DevState, 0.75f, 0);
	cudaGraphicsUnmapResources(1, &m_resource, 0);

    return ret;
}

void GameOfLife::Draw()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glDrawArrays(GL_POINTS, 0, m_widthX * m_widthY);
}

void GameOfLife::Update()
{
    unsigned int* m_DevState;
	unsigned int* m_DevNextState;
	cudaMalloc((void**)&m_DevNextState, m_BufferSize);

	cudaGraphicsMapResources(1, &m_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&m_DevState, &m_BufferSize, m_resource);

	_next<<<m_blocks, m_threads>>>(m_DevState, m_DevNextState);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaMemcpy(m_DevState, m_DevNextState, m_BufferSize, cudaMemcpyDeviceToDevice) );

	cudaGraphicsUnmapResources(1, &m_resource, 0);
	cudaFree(m_DevNextState);
}