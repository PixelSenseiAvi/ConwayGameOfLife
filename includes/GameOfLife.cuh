#pragma once

#include <GL/glew.h>
#include <stdio.h>
#include <cassert>

#include "cudaGL.h"
#include "cuda_gl_interop.h"
#include "vectors.h"

#include "Utilities.cuh"
#include "Shader.h"


class GameOfLife
{
   //private state variables
   public:
      VertexBuffer m_vb;
      VertexArray m_va;
      VertexBufferLayout m_layout;

      GLuint m_VBO;
      dim3 m_threads;
      dim3 m_blocks;
      unsigned int m_widthX;
      unsigned int m_widthY;
      unsigned int m_pointSize;
      size_t m_BufferSize;
      struct cudaGraphicsResource* m_resource;
      
      // Constructor
      GameOfLife(dim3 threads, dim3 blocks, 
                              unsigned int widthX, unsigned int widthY, 
                              unsigned int pointSize) 
         :  m_threads   ( threads ),
            m_blocks    ( blocks ),
            m_widthX    ( widthX ),
            m_widthY    ( widthY ),
            m_pointSize ( pointSize )
      {}

      // Destructor
      ~GameOfLife() 
      {
         glDisableVertexAttribArray(0);
      }

      bool Init();
      void Draw();
      void Update();
};