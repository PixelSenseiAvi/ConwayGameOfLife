#pragma once

#include <stdio.h>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>

#include <GL/glew.h>
#include <cassert>
#include <vector>


class Shader
{
public:
	Shader();

	void CreateFromString(const char* vertexCode, const char* fragmentCode);
	void CreateFromFiles(const char* vertexLocation, const char* fragmentLocation);
	void CreateFromFiles(const char* vertexLocation, const char* geometryLocation, const char* fragmentLocation);

	void Validate();

	std::string ReadFile(const char* fileLocation);

	void UseShader();
	void Unbind();
	void ClearShader();

	void SetUniform1f(const std::string& name, float value);
	void SetUniform1fv(const std::string& name, std::vector<float> data);
	void SetUniformUint(const std::string& name, unsigned int value);
	void SetUniform4f(const std::string& name, float f0, float f1, float f2, float f3);

	~Shader();

private:

	GLuint shaderID;
	/*, uniformProjection, uniformModel, uniformView, uniformEyePosition,
		uniformSpecularIntensity, uniformShininess, 
		uniformTexture, uniformFarPlane, skybox;
	*/

	void CompileShader(const char* vertexCode, const char* fragmentCode);
	void CompileShader(const char* vertexCode, const char* geometryCode, const char* fragmentCode);
	void AddShader(GLuint theProgram, const char* shaderCode, GLenum shaderType);

	void CompileProgram();

	int GetUniformLocation(const std::string& name);
};

