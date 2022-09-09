#include <stdio.h>
#include <cstring>
#include <cmath>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Window.h"
#include "Camera.h"

#include "Utilities.cuh"


const float toRadians = 3.14159265f / 180.0f;

Window mainWindow;

std::vector<Shader> shaderList;

Camera camera;

//Texture dirtTexture;

GLfloat deltaTime = 0.0f;
GLfloat lastTime = 0.0f;
GLuint skyboxLocation;

// Vertex Shader
static const char* vShader = "Shaders/shader.vert";

// Fragment Shader
static const char* fShader = "Shaders/shader.frag";

/////////////////////////////////////////////
//Size of each cell (in pixels)
int pixelSize = 8;	//8 X 8 pixels for each cell

const unsigned int threadsPerBlockX = 20;
const unsigned int blockCountX = 6;

const unsigned int threadsPerBlockY = 20;
const unsigned int blockCountY = 6;

// 120x120
const unsigned int widthX = threadsPerBlockX * blockCountX; 
const unsigned int widthY = threadsPerBlockY * blockCountY;

dim3 threadSize(threadsPerBlockX, threadsPerBlockY);
dim3 blockSize(blockCountX, blockCountY);
//////////////////////////////////////////////

size_t m_BufferSize;
struct cudaGraphicsResource* m_resource;

GLuint m_VBO;


void CreateShaders()
{
	Shader *shader1 = new Shader();
	shader1->CreateFromFiles(vShader, fShader);
	shaderList.push_back(*shader1);
}


void Update()
{
	unsigned int* m_DevState;
	unsigned int* m_DevNextState;
	cudaMalloc((void**)&m_DevNextState, m_BufferSize);

	cudaGraphicsMapResources(1, &m_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&m_DevState, &m_BufferSize, m_resource);

	_next<<<blockSize, threadSize>>>(m_DevState, m_DevNextState);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaMemcpy(m_DevState, m_DevNextState, m_BufferSize, cudaMemcpyDeviceToDevice) );

	cudaGraphicsUnmapResources(1, &m_resource, 0);
	cudaFree(m_DevNextState);

}


void init()
{

	CreateShaders();

	glClearColor(0.0f, 0.0f, 1.0f, 0.0f);
    glPointSize(pixelSize);
    
    // // Init Buffer
    glGenBuffers(1, &m_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, widthX * widthY * sizeof(unsigned int), 0, GL_DYNAMIC_DRAW);

    // Attrib Pointer
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 1, GL_UNSIGNED_INT, GL_FALSE, 0, 0);

	shaderList[0].UseShader();

	shaderList[0].SetUniformUint("_uWidthX", widthX);
    shaderList[0].SetUniformUint("_uWidthY", widthY);
	shaderList[0].SetUniform4f("_uOnColor", 1., 1., 1., 1.);
	shaderList[0].SetUniform4f("_uOffColor", 0., 0., 0., 1.);
	shaderList[0].SetUniform4f("_uWindowXY", -1.0, 1.0, -1.0, 1.0);

	// CUDA graphics resource 
    cudaGraphicsGLRegisterBuffer(&m_resource, m_VBO, cudaGraphicsRegisterFlagsNone);

    unsigned int* m_DevState;
    m_BufferSize = widthX * widthY * sizeof(unsigned int);
	cudaGraphicsMapResources(1, &m_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&m_DevState, &m_BufferSize, m_resource);
	_random<<<blockSize, threadSize>>>(m_DevState, 0.75f, 0);
	cudaGraphicsUnmapResources(1, &m_resource, 0);

	shaderList[0].Validate();

}

void Draw()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glDrawArrays(GL_POINTS, 0, widthX * widthY);
}

int main() 
{
	mainWindow = Window(1366, 768);
	mainWindow.Initialise();

	init();
	//main loop
	while (!mainWindow.getShouldClose())
	{
		GLfloat now = glfwGetTime(); 
		deltaTime = now - lastTime; 
		lastTime = now;

		glfwPollEvents();
		if(mainWindow.RUN_SIMULATION)
			Update();
		if(mainWindow.SKIP_FORWARD)
			Update();
		mainWindow.SKIP_FORWARD = GL_FALSE;
		if(mainWindow.READ_INSERT)
		{
			//TODO:
		}
		Draw();
		mainWindow.swapBuffers();
	}

	return 0;
}

/*
 ~GameOfLife() 
      {
         glDisableVertexAttribArray(0);
      }
*/